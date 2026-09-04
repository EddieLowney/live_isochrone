"""Persistence, built as an archive rather than a cache.

The operational system only needs the last few hundred cycles. A model trained
later needs everything, and anything not written at the moment of measurement is
gone for good — you cannot re-request last Tuesday's traffic. So the schema errs
toward keeping more than the running system reads.

What is kept and why:

* ``observation`` — every direct measurement, including the free-flow value and
  route distance the provider returned alongside it. Distance matters more than
  it looks: duration alone is not comparable across cells, but distance over
  duration is a speed, and speed is the quantity that actually generalises.
* ``failure`` — cells requested that came back empty, with the cycle. Feeds the
  quarantine and flags data-quality problems after the fact.
* ``cycle_log`` — one row per cycle: allowance, spend, holdout error, whether a
  refit happened. Without it there is no way to ask "did reconstruction get
  better after that change".
* ``baseline_history`` — append-only snapshots. The live baseline is an EWMA and
  overwrites itself, so its drift would otherwise be unrecoverable.
* ``cell_geometry`` and ``run_meta`` — where each cell was, keyed by grid
  fingerprint. Cell ids are axial coordinates and get reused when the grid
  moves, so without this an exported dataset cannot be located on a map.
* ``estimate`` — what the reconstructor predicted. Lets a future model be scored
  against what was actually deployed.

Schema changes go through ``_migrate``. The version lives in the file, so an
existing database is upgraded in place rather than silently mismatching.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from .registry import STORES
from .types import Measurement, utcnow

SCHEMA_VERSION = 4


def _epoch(dt: datetime) -> float:
    return dt.astimezone(timezone.utc).timestamp()


class Store(ABC):
    # -- baseline ---------------------------------------------------------
    @abstractmethod
    def set_baseline(self, values: dict[str, float], profile: str) -> None: ...

    @abstractmethod
    def get_baseline(self) -> dict[str, float]: ...

    # -- observations -----------------------------------------------------
    @abstractmethod
    def record(self, measurements: Iterable[Measurement], cycle: int) -> None: ...

    @abstractmethod
    def latest_observations(self) -> dict[str, Measurement]: ...

    @abstractmethod
    def history(self, max_cycles: int = 500) -> list[tuple[int, dict[str, float]]]: ...

    @abstractmethod
    def next_cycle(self) -> int: ...

    # -- archive ----------------------------------------------------------
    def record_failures(self, cell_ids: Iterable[str], cycle: int, provider: str,
                        reason: str = "no_result", when: Optional[datetime] = None) -> None:
        """Cells requested that returned nothing."""

    def record_cycle(self, report: dict) -> None:
        """One row per cycle of engine diagnostics."""

    def record_estimates(self, cycle: int, estimates: dict[str, float],
                         direct: Iterable[str], when: Optional[datetime] = None) -> None:
        """What the reconstructor predicted, for retrospective scoring."""

    def register_grid(self, fingerprint: str, origin, cells, grid_spec: dict) -> None:
        """Pin cell geometry so exported data can be located later."""

    def archive_stats(self) -> dict:
        return {}

    def export_rows(self, since_epoch: Optional[float] = None) -> tuple[list[str], list[tuple]]:
        return [], []

    # -- misc -------------------------------------------------------------
    @abstractmethod
    def put_surface(self, payload: dict) -> None: ...

    @abstractmethod
    def get_surface(self) -> Optional[dict]: ...

    @abstractmethod
    def get_state(self, key: str) -> Optional[dict]: ...

    @abstractmethod
    def put_state(self, key: str, value: dict) -> None: ...

    @abstractmethod
    def reset(self, keep_archive: bool = False) -> None:
        """Clear operational state. With `keep_archive`, measurement history and
        cycle logs survive — a grid change invalidates the baseline, not the
        record of what the roads were doing."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS run_meta (
    fingerprint TEXT PRIMARY KEY,
    origin_lat REAL, origin_lon REAL,
    grid_spec TEXT, first_seen TEXT, last_seen TEXT);

CREATE TABLE IF NOT EXISTS cell_geometry (
    fingerprint TEXT NOT NULL, cell_id TEXT NOT NULL,
    lat REAL NOT NULL, lon REAL NOT NULL, ring INTEGER,
    polygon TEXT,
    PRIMARY KEY (fingerprint, cell_id));

CREATE TABLE IF NOT EXISTS baseline (
    cell_id TEXT PRIMARY KEY, seconds REAL NOT NULL,
    profile TEXT NOT NULL, updated_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS baseline_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id TEXT NOT NULL, seconds REAL NOT NULL, profile TEXT,
    recorded_at TEXT NOT NULL, epoch REAL NOT NULL);
CREATE INDEX IF NOT EXISTS bh_cell ON baseline_history(cell_id, epoch);

CREATE TABLE IF NOT EXISTS observation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id TEXT NOT NULL, cycle INTEGER NOT NULL,
    observed_at TEXT NOT NULL, epoch REAL NOT NULL,
    seconds REAL NOT NULL,
    free_flow_seconds REAL, distance_m REAL,
    p10_seconds REAL, p50_seconds REAL, p90_seconds REAL, samples INTEGER,
    walk_seconds REAL, wait_seconds REAL, transfers INTEGER,
    next_jump_s REAL,
    provider TEXT NOT NULL, profile TEXT NOT NULL, elements REAL,
    fingerprint TEXT,
    UNIQUE (cell_id, cycle));
CREATE INDEX IF NOT EXISTS obs_cycle ON observation(cycle);
CREATE INDEX IF NOT EXISTS obs_cell_time ON observation(cell_id, epoch);
CREATE INDEX IF NOT EXISTS obs_epoch ON observation(epoch);

CREATE TABLE IF NOT EXISTS failure (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id TEXT NOT NULL, cycle INTEGER NOT NULL,
    observed_at TEXT NOT NULL, epoch REAL NOT NULL,
    provider TEXT, reason TEXT);
CREATE INDEX IF NOT EXISTS fail_cell ON failure(cell_id, epoch);

CREATE TABLE IF NOT EXISTS cycle_log (
    cycle INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL, epoch REAL NOT NULL,
    allowance INTEGER, elements_spent REAL,
    cells_requested INTEGER, cells_measured INTEGER,
    cells_masked INTEGER, cells_failed INTEGER,
    batches_sent INTEGER, batches_refused INTEGER,
    holdout_mae_s REAL, refit INTEGER, budget_stalled INTEGER,
    budget_remaining REAL, baseline_updated INTEGER, errors TEXT);
CREATE INDEX IF NOT EXISTS cyc_epoch ON cycle_log(epoch);

CREATE TABLE IF NOT EXISTS estimate (
    cycle INTEGER NOT NULL, cell_id TEXT NOT NULL,
    epoch REAL NOT NULL, estimated_s REAL, is_direct INTEGER,
    PRIMARY KEY (cycle, cell_id));

CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);

-- Training-ready join. Derived columns are computed rather than stored so a
-- change of definition does not require rewriting history.
CREATE VIEW IF NOT EXISTS observation_features AS
SELECT
    o.id, o.cell_id, o.cycle, o.observed_at, o.epoch,
    o.seconds, o.free_flow_seconds, o.distance_m,
    o.seconds - o.free_flow_seconds                              AS delta_s,
    o.p90_seconds - o.p10_seconds                                AS spread_s,
    o.p10_seconds, o.p50_seconds, o.p90_seconds, o.samples,
    o.walk_seconds, o.wait_seconds, o.transfers, o.next_jump_s,
    CASE WHEN o.free_flow_seconds > 0
         THEN o.seconds / o.free_flow_seconds END                AS ratio,
    CASE WHEN o.seconds > 0
         THEN o.distance_m / o.seconds * 3.6 END                 AS speed_kmh,
    CAST(strftime('%H', o.observed_at) AS INTEGER)               AS hour_utc,
    CAST(strftime('%w', o.observed_at) AS INTEGER)               AS dow,
    o.provider, o.profile, o.fingerprint,
    g.lat, g.lon, g.ring
FROM observation o
LEFT JOIN cell_geometry g
       ON g.cell_id = o.cell_id AND g.fingerprint = o.fingerprint;
"""


@STORES.register("sqlite")
class SqliteStore(Store):
    def __init__(self, path: str = "surface.db", record_estimates: bool = True, **_ignored):
        self.path = path
        self.record_estimates_enabled = bool(record_estimates)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        self._fingerprint: Optional[str] = None
        self._migrate()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _migrate(self) -> None:
        conn = self._conn()
        existing = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        v2 = "meta" in existing and "observation" in existing
        if v2:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(observation)")}
            for name in ("p10_seconds", "p50_seconds", "p90_seconds", "samples",
                         "walk_seconds", "wait_seconds", "transfers", "next_jump_s"):
                if name not in cols:
                    decl = "INTEGER" if name in ("samples", "transfers") else "REAL"
                    conn.execute(f"ALTER TABLE observation ADD COLUMN {name} {decl}")
            # The view is defined over the old column set; rebuild it.
            conn.execute("DROP VIEW IF EXISTS observation_features")
        legacy = "observation" in existing and "meta" not in existing
        if legacy:
            # v1 kept only cell_id/cycle/seconds/observed_at/provider/profile and
            # had no surrogate key. SQLite cannot ALTER in an AUTOINCREMENT
            # primary key, so the table is rebuilt and the rows copied across —
            # measurements already collected are carried forward, not discarded.
            conn.execute("ALTER TABLE observation RENAME TO observation_v1")
        conn.executescript(SCHEMA)
        if legacy:
            conn.execute(
                "INSERT INTO observation"
                " (cell_id, cycle, observed_at, epoch, seconds, provider, profile)"
                " SELECT cell_id, cycle, observed_at,"
                "        CAST(strftime('%s', observed_at) AS REAL),"
                "        seconds, provider, profile"
                " FROM observation_v1"
            )
            conn.execute("DROP TABLE observation_v1")
        conn.execute(
            "INSERT INTO meta(k, v) VALUES ('schema_version', ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (str(SCHEMA_VERSION),),
        )

    @property
    def schema_version(self) -> int:
        row = self._conn().execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()
        return int(row[0]) if row else 0

    # -- grid identity -----------------------------------------------------

    def register_grid(self, fingerprint, origin, cells, grid_spec) -> None:
        self._fingerprint = fingerprint
        now = utcnow().isoformat()
        conn = self._conn()
        conn.execute(
            "INSERT INTO run_meta(fingerprint, origin_lat, origin_lon, grid_spec,"
            " first_seen, last_seen) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(fingerprint) DO UPDATE SET last_seen=excluded.last_seen",
            (fingerprint, origin.lat, origin.lon, json.dumps(grid_spec), now, now),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO cell_geometry"
            "(fingerprint, cell_id, lat, lon, ring, polygon) VALUES (?,?,?,?,?,?)",
            [
                (
                    fingerprint, c.id, c.center.lat, c.center.lon, c.ring,
                    json.dumps([[p.lon, p.lat] for p in c.polygon]),
                )
                for c in cells
            ],
        )

    # -- baseline ----------------------------------------------------------

    def set_baseline(self, values, profile):
        now = utcnow()
        iso, ep = now.isoformat(), _epoch(now)
        conn = self._conn()
        conn.executemany(
            "INSERT INTO baseline(cell_id, seconds, profile, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(cell_id) DO UPDATE SET seconds=excluded.seconds, "
            "profile=excluded.profile, updated_at=excluded.updated_at",
            [(k, float(v), profile, iso) for k, v in values.items()],
        )
        # The live baseline is an EWMA that overwrites itself; keep the trail.
        conn.executemany(
            "INSERT INTO baseline_history(cell_id, seconds, profile, recorded_at, epoch)"
            " VALUES (?,?,?,?,?)",
            [(k, float(v), profile, iso, ep) for k, v in values.items()],
        )

    def get_baseline(self):
        return {
            r[0]: float(r[1])
            for r in self._conn().execute("SELECT cell_id, seconds FROM baseline")
        }

    # -- observations ------------------------------------------------------

    def record(self, measurements, cycle):
        rows = [
            (
                m.cell_id, cycle, m.observed_at.isoformat(), _epoch(m.observed_at),
                m.seconds, m.free_flow_seconds, m.distance_m,
                m.p10_seconds, m.p50_seconds, m.p90_seconds, m.samples,
                m.walk_seconds, m.wait_seconds, m.transfers, m.next_jump_s,
                m.provider, m.profile, m.elements, self._fingerprint,
            )
            for m in measurements
        ]
        if rows:
            self._conn().executemany(
                "INSERT INTO observation(cell_id, cycle, observed_at, epoch, seconds,"
                " free_flow_seconds, distance_m, p10_seconds, p50_seconds, p90_seconds,"
                " samples, walk_seconds, wait_seconds, transfers, next_jump_s,"
                " provider, profile, elements, fingerprint)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(cell_id, cycle) DO UPDATE SET"
                " seconds=excluded.seconds, free_flow_seconds=excluded.free_flow_seconds,"
                " distance_m=excluded.distance_m, p10_seconds=excluded.p10_seconds,"
                " p50_seconds=excluded.p50_seconds, p90_seconds=excluded.p90_seconds,"
                " samples=excluded.samples, walk_seconds=excluded.walk_seconds,"
                " wait_seconds=excluded.wait_seconds, transfers=excluded.transfers,"
                " next_jump_s=excluded.next_jump_s",
                rows,
            )

    def record_failures(self, cell_ids, cycle, provider, reason="no_result", when=None):
        w = when or utcnow()
        rows = [(c, cycle, w.isoformat(), _epoch(w), provider, reason) for c in cell_ids]
        if rows:
            self._conn().executemany(
                "INSERT INTO failure(cell_id, cycle, observed_at, epoch, provider, reason)"
                " VALUES (?,?,?,?,?,?)",
                rows,
            )

    def record_cycle(self, report):
        started = report.get("started_at")
        try:
            ep = _epoch(datetime.fromisoformat(started))
        except Exception:  # noqa: BLE001
            ep = _epoch(utcnow())
        self._conn().execute(
            "INSERT OR REPLACE INTO cycle_log(cycle, started_at, epoch, allowance,"
            " elements_spent, cells_requested, cells_measured, cells_masked, cells_failed,"
            " batches_sent, batches_refused, holdout_mae_s, refit, budget_stalled,"
            " budget_remaining, baseline_updated, errors)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                report.get("cycle"), started, ep, report.get("allowance"),
                report.get("elements_spent"), report.get("cells_requested"),
                report.get("cells_measured"), report.get("cells_masked"),
                report.get("cells_failed"), report.get("batches_sent"),
                report.get("batches_refused"), report.get("holdout_mae_s"),
                int(bool(report.get("refit"))), int(bool(report.get("budget_stalled"))),
                report.get("budget_remaining"), report.get("baseline_updated"),
                json.dumps(report.get("errors") or []),
            ),
        )

    def record_estimates(self, cycle, estimates, direct, when=None):
        if not self.record_estimates_enabled:
            return
        w = when or utcnow()
        ep = _epoch(w)
        direct = set(direct)
        self._conn().executemany(
            "INSERT OR REPLACE INTO estimate(cycle, cell_id, epoch, estimated_s, is_direct)"
            " VALUES (?,?,?,?,?)",
            [(cycle, k, ep, v, int(k in direct)) for k, v in estimates.items()],
        )

    def latest_observations(self):
        rows = self._conn().execute(
            "SELECT o.cell_id, o.seconds, o.observed_at, o.provider, o.profile,"
            " o.free_flow_seconds, o.distance_m, o.p10_seconds, o.p90_seconds,"
            " o.walk_seconds, o.wait_seconds, o.transfers, o.next_jump_s"
            " FROM observation o"
            " JOIN (SELECT cell_id, MAX(cycle) mc FROM observation GROUP BY cell_id) t"
            " ON o.cell_id=t.cell_id AND o.cycle=t.mc"
        )
        return {
            r[0]: Measurement(
                r[0], float(r[1]), datetime.fromisoformat(r[2]), r[3], r[4],
                free_flow_seconds=r[5], distance_m=r[6],
                p10_seconds=r[7], p90_seconds=r[8],
                walk_seconds=r[9], wait_seconds=r[10], transfers=r[11],
                next_jump_s=r[12],
            )
            for r in rows
        }

    def history(self, max_cycles: int = 500):
        cycles = [
            r[0]
            for r in self._conn().execute(
                "SELECT DISTINCT cycle FROM observation ORDER BY cycle DESC LIMIT ?",
                (max_cycles,),
            )
        ]
        if not cycles:
            return []
        marks = ",".join("?" * len(cycles))
        rows = self._conn().execute(
            f"SELECT cycle, cell_id, seconds FROM observation WHERE cycle IN ({marks})",
            cycles,
        )
        grouped: dict[int, dict[str, float]] = {}
        for cycle, cell_id, secs in rows:
            grouped.setdefault(cycle, {})[cell_id] = float(secs)
        return sorted(grouped.items())

    def next_cycle(self):
        row = self._conn().execute(
            "SELECT MAX(c) FROM (SELECT MAX(cycle) c FROM observation"
            " UNION SELECT MAX(cycle) FROM cycle_log)"
        ).fetchone()
        return (row[0] or 0) + 1

    # -- archive -----------------------------------------------------------

    def archive_stats(self):
        c = self._conn()
        one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
        span = c.execute(
            "SELECT MIN(observed_at), MAX(observed_at) FROM observation"
        ).fetchone()
        return {
            "schema_version": self.schema_version,
            "observations": one("SELECT COUNT(*) FROM observation"),
            "with_free_flow": one(
                "SELECT COUNT(*) FROM observation WHERE free_flow_seconds IS NOT NULL"
            ),
            "with_distance": one(
                "SELECT COUNT(*) FROM observation WHERE distance_m IS NOT NULL"
            ),
            "failures": one("SELECT COUNT(*) FROM failure"),
            "cycles_logged": one("SELECT COUNT(*) FROM cycle_log"),
            "estimates": one("SELECT COUNT(*) FROM estimate"),
            "baseline_snapshots": one("SELECT COUNT(*) FROM baseline_history"),
            "cells_known": one("SELECT COUNT(*) FROM cell_geometry"),
            "grids": one("SELECT COUNT(*) FROM run_meta"),
            "first_observation": span[0],
            "last_observation": span[1],
            "db_bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0,
        }

    def export_rows(self, since_epoch=None):
        q = "SELECT * FROM observation_features"
        args: tuple = ()
        if since_epoch is not None:
            q += " WHERE epoch >= ?"
            args = (since_epoch,)
        q += " ORDER BY epoch, cell_id"
        cur = self._conn().execute(q, args)
        return [d[0] for d in cur.description], cur.fetchall()

    # -- misc --------------------------------------------------------------

    def put_surface(self, payload):
        self._conn().execute(
            "INSERT INTO kv(k, v) VALUES ('surface', ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (json.dumps(payload),),
        )

    def get_surface(self):
        row = self._conn().execute("SELECT v FROM kv WHERE k='surface'").fetchone()
        return json.loads(row[0]) if row else None

    def get_state(self, key):
        row = self._conn().execute(
            "SELECT v FROM kv WHERE k=?", (f"state:{key}",)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_state(self, key, value):
        self._conn().execute(
            "INSERT INTO kv(k, v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (f"state:{key}", json.dumps(value)),
        )

    def reset(self, keep_archive: bool = False):
        """Default clears everything. `keep_archive` keeps the measurement record,
        which is what you want when only the grid geometry changed — the roads
        were doing what they were doing regardless of how you tiled them."""
        conn = self._conn()
        conn.executescript("DELETE FROM baseline; DELETE FROM kv;")
        if not keep_archive:
            conn.executescript(
                "DELETE FROM observation; DELETE FROM failure;"
                "DELETE FROM cycle_log; DELETE FROM estimate;"
                "DELETE FROM baseline_history;"
            )
        else:
            # Keep the rows but detach them from the cycle counter so a new grid
            # starts at cycle 1 without colliding with the archive.
            conn.execute("UPDATE observation SET cycle = -ABS(cycle) WHERE cycle > 0")
            conn.execute("UPDATE cycle_log SET cycle = -ABS(cycle) WHERE cycle > 0")
            conn.execute("UPDATE estimate SET cycle = -ABS(cycle) WHERE cycle > 0")


@STORES.register("memory")
class MemoryStore(Store):
    """Volatile. Used by tests and `simulate`, where persistence is noise."""

    def __init__(self, record_estimates: bool = True, **_ignored):
        self.record_estimates_enabled = bool(record_estimates)
        self._baseline: dict[str, float] = {}
        self._obs: dict[int, dict[str, Measurement]] = {}
        self._surface: Optional[dict] = None
        self._state: dict[str, dict] = {}
        self.failures: list[tuple] = []
        self.cycles: list[dict] = []
        self.estimates: dict[int, dict] = {}
        self.geometry: dict[str, tuple] = {}
        self.baseline_history: list[tuple] = []

    def set_baseline(self, values, profile):
        self._baseline.update({k: float(v) for k, v in values.items()})
        now = utcnow()
        self.baseline_history += [(k, float(v), profile, now) for k, v in values.items()]

    def get_baseline(self):
        return dict(self._baseline)

    def record(self, measurements, cycle):
        bucket = self._obs.setdefault(cycle, {})
        for m in measurements:
            bucket[m.cell_id] = m

    def record_failures(self, cell_ids, cycle, provider, reason="no_result", when=None):
        w = when or utcnow()
        self.failures += [(c, cycle, w, provider, reason) for c in cell_ids]

    def record_cycle(self, report):
        self.cycles.append(dict(report))

    def record_estimates(self, cycle, estimates, direct, when=None):
        if self.record_estimates_enabled:
            self.estimates[cycle] = dict(estimates)

    def register_grid(self, fingerprint, origin, cells, grid_spec):
        for c in cells:
            self.geometry[c.id] = (fingerprint, c.center.lat, c.center.lon, c.ring)

    def latest_observations(self):
        out: dict[str, Measurement] = {}
        for cycle in sorted(self._obs):
            out.update(self._obs[cycle])
        return out

    def history(self, max_cycles: int = 500):
        cycles = sorted(self._obs)[-max_cycles:]
        return [(c, {k: m.seconds for k, m in self._obs[c].items()}) for c in cycles]

    def next_cycle(self):
        return (max(self._obs) if self._obs else 0) + 1

    def archive_stats(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "observations": sum(len(v) for v in self._obs.values()),
            "failures": len(self.failures),
            "cycles_logged": len(self.cycles),
            "cells_known": len(self.geometry),
        }

    def put_surface(self, payload):
        self._surface = payload

    def get_surface(self):
        return self._surface

    def get_state(self, key):
        return self._state.get(key)

    def put_state(self, key, value):
        self._state[key] = value

    def reset(self, keep_archive: bool = False):
        self._baseline.clear()
        self._surface = None
        self._state.clear()
        if not keep_archive:
            self._obs.clear()
            self.failures.clear()
            self.cycles.clear()
            self.estimates.clear()
            self.baseline_history.clear()


def build_store(kind: str, **kwargs) -> Store:
    return STORES.build(kind, **kwargs)
