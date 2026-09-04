"""Quota enforcement.

The rule that makes "never exceed the free tier" a guarantee rather than an
intention: spend is decremented *before* the call, atomically, and the call only
happens if the decrement succeeded. A refund is issued only when a request fails
in a way the provider does not bill.

Two period shapes are supported because providers differ: Mapbox resets monthly,
TomTom daily. Pacing spreads a period's allowance across the cycles remaining in
it, so an idle morning does not license a runaway afternoon.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import Optional

from .registry import LEDGERS, PACERS
from .types import utcnow


def period_key(period: str, when: datetime) -> str:
    w = when.astimezone(timezone.utc)
    if period == "monthly":
        return w.strftime("%Y-%m")
    if period == "daily":
        return w.strftime("%Y-%m-%d")
    if period == "hourly":
        return w.strftime("%Y-%m-%dH%H")
    raise ValueError(f"unsupported period '{period}'")


def period_end(period: str, when: datetime) -> datetime:
    w = when.astimezone(timezone.utc)
    if period == "monthly":
        days = monthrange(w.year, w.month)[1]
        return w.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days)
    if period == "daily":
        return w.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if period == "hourly":
        return w.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    raise ValueError(f"unsupported period '{period}'")


class BudgetLedger(ABC):
    """Tracks units consumed per provider per period."""

    def __init__(self, quotas: dict[str, int], period: str = "monthly", safety: float = 0.9):
        #: Raw provider allowances, e.g. {"mapbox": 100_000}.
        self.quotas = dict(quotas)
        self.period = period
        #: Fraction of the raw quota the system is allowed to touch. Headroom
        #: covers retries, unroutable-but-billed cells, and calibration.
        self.safety = float(safety)

    def effective_quota(self, key: str) -> int:
        return int(self.quotas.get(key, 0) * self.safety)

    @abstractmethod
    def spent(self, key: str, when: Optional[datetime] = None) -> float: ...

    @abstractmethod
    def try_spend(self, key: str, units: float, when: Optional[datetime] = None) -> bool:
        """Atomically reserve `units`. False means refused; callers degrade."""

    @abstractmethod
    def refund(self, key: str, units: float, when: Optional[datetime] = None) -> None: ...

    def remaining(self, key: str, when: Optional[datetime] = None) -> float:
        return max(0.0, self.effective_quota(key) - self.spent(key, when))

    def snapshot(self, when: Optional[datetime] = None) -> dict:
        w = when or utcnow()
        return {
            "period": self.period,
            "period_key": period_key(self.period, w),
            "period_ends": period_end(self.period, w).isoformat(),
            "providers": {
                k: {
                    "quota": self.quotas.get(k, 0),
                    "effective_quota": self.effective_quota(k),
                    "spent": round(self.spent(k, w), 2),
                    "remaining": round(self.remaining(k, w), 2),
                }
                for k in self.quotas
            },
        }


@LEDGERS.register("memory")
class MemoryLedger(BudgetLedger):
    """Process-local. Fine for a single-process prototype, useless across
    workers — use the sqlite ledger the moment there is more than one."""

    def __init__(self, quotas: dict[str, int], period: str = "monthly", safety: float = 0.9):
        super().__init__(quotas, period, safety)
        self._lock = threading.Lock()
        self._spent: dict[tuple[str, str], float] = {}

    def spent(self, key: str, when: Optional[datetime] = None) -> float:
        pk = period_key(self.period, when or utcnow())
        with self._lock:
            return self._spent.get((key, pk), 0.0)

    def try_spend(self, key: str, units: float, when: Optional[datetime] = None) -> bool:
        pk = period_key(self.period, when or utcnow())
        cap = self.effective_quota(key)
        with self._lock:
            cur = self._spent.get((key, pk), 0.0)
            if cur + units > cap:
                return False
            self._spent[(key, pk)] = cur + units
            return True

    def refund(self, key: str, units: float, when: Optional[datetime] = None) -> None:
        pk = period_key(self.period, when or utcnow())
        with self._lock:
            self._spent[(key, pk)] = max(0.0, self._spent.get((key, pk), 0.0) - units)


@LEDGERS.register("sqlite")
class SqliteLedger(BudgetLedger):
    """Durable and safe across processes. The conditional UPDATE is the whole
    trick: the cap is enforced inside the write, so two workers racing cannot
    both succeed past the limit."""

    def __init__(
        self,
        quotas: dict[str, int],
        period: str = "monthly",
        safety: float = 0.9,
        path: str = "budget.db",
    ):
        super().__init__(quotas, period, safety)
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS ledger ("
                " key TEXT NOT NULL, period TEXT NOT NULL, spent REAL NOT NULL DEFAULT 0,"
                " PRIMARY KEY (key, period))"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
        return conn

    def spent(self, key: str, when: Optional[datetime] = None) -> float:
        pk = period_key(self.period, when or utcnow())
        row = self._conn().execute(
            "SELECT spent FROM ledger WHERE key=? AND period=?", (key, pk)
        ).fetchone()
        return float(row[0]) if row else 0.0

    def try_spend(self, key: str, units: float, when: Optional[datetime] = None) -> bool:
        pk = period_key(self.period, when or utcnow())
        cap = self.effective_quota(key)
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO ledger(key, period, spent) VALUES (?,?,0)", (key, pk)
            )
            cur = conn.execute(
                "UPDATE ledger SET spent = spent + ? "
                "WHERE key=? AND period=? AND spent + ? <= ?",
                (units, key, pk, units, cap),
            )
            ok = cur.rowcount > 0
            conn.execute("COMMIT")
            return ok
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def refund(self, key: str, units: float, when: Optional[datetime] = None) -> None:
        pk = period_key(self.period, when or utcnow())
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE ledger SET spent = MAX(0, spent - ?) WHERE key=? AND period=?",
            (units, key, pk),
        )
        conn.execute("COMMIT")


class PacingPolicy(ABC):
    """Converts remaining period budget into an allowance for one cycle."""

    @abstractmethod
    def allowance(
        self, ledger: BudgetLedger, key: str, when: datetime, cycle_seconds: float
    ) -> int: ...


@PACERS.register("proportional")
class ProportionalPacer(PacingPolicy):
    """Spread what is left evenly over the cycles left in the period.

    Self-correcting: underspend one hour and every later cycle gets slightly
    more. `burst` allows a bounded multiple of the even share so a rush-hour
    weighting can pull budget forward without the pacer fighting it.
    """

    def __init__(self, burst: float = 3.0, floor: int = 0, **_ignored):
        self.burst = float(burst)
        self.floor = int(floor)

    def allowance(self, ledger, key, when, cycle_seconds):
        remaining = ledger.remaining(key, when)
        secs_left = max(cycle_seconds, (period_end(ledger.period, when) - when).total_seconds())
        cycles_left = max(1.0, secs_left / cycle_seconds)
        even = remaining / cycles_left
        return max(self.floor, int(even * self.burst))


@PACERS.register("flat")
class FlatPacer(PacingPolicy):
    """Fixed units per cycle regardless of what is left. Predictable, and will
    stop early in a long month rather than tapering."""

    def __init__(self, units_per_cycle: int = 20, **_ignored):
        self.units_per_cycle = int(units_per_cycle)

    def allowance(self, ledger, key, when, cycle_seconds):
        return int(min(self.units_per_cycle, ledger.remaining(key, when)))


@PACERS.register("schedule")
class ScheduledPacer(PacingPolicy):
    """Time-of-day weighting. Delta variance is near zero overnight, so uniform
    cadence spends most of the quota confirming nothing is happening.

    `weights` maps UTC hour ranges to a multiplier on the even share.
    """

    def __init__(
        self,
        base: float = 1.0,
        windows: Optional[list[dict]] = None,
        floor: int = 1,
        **_ignored,
    ):
        self.base = float(base)
        self.floor = int(floor)
        self.windows = windows or []

    def _weight(self, when: datetime) -> float:
        hour = when.astimezone(timezone.utc).hour
        for w in self.windows:
            start, end = int(w["from_hour"]), int(w["to_hour"])
            inside = start <= hour < end if start <= end else (hour >= start or hour < end)
            if inside:
                return float(w.get("weight", 1.0))
        return self.base

    def allowance(self, ledger, key, when, cycle_seconds):
        remaining = ledger.remaining(key, when)
        secs_left = max(cycle_seconds, (period_end(ledger.period, when) - when).total_seconds())
        cycles_left = max(1.0, secs_left / cycle_seconds)
        even = remaining / cycles_left
        return max(self.floor, int(even * self._weight(when)))


def build_ledger(kind: str, **kwargs) -> BudgetLedger:
    return LEDGERS.build(kind, **kwargs)


def build_pacer(kind: str, **kwargs) -> PacingPolicy:
    return PACERS.build(kind, **kwargs)
