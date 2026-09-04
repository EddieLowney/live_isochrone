"""Walking layers: origin to stops, stop to stop, stops to cells.

RAPTOR itself needs only the first two, and both are small — a few dozen stops
near the origin, and stop pairs within a few hundred metres. The third is the
min-plus egress term and is the one that grows with output resolution.

Two sources:

* ``straight_line`` — haversine times a detour factor. Instant, roughly 10-15%
  optimistic, and wrong in the specific places that matter most: across the
  Charles, over the Mass Pike, through rail yards.
* ``otp`` — walk-only queries against the running OTP instance. Same street
  graph OTP uses for its own routing, which is what makes a RAPTOR-vs-OTP
  comparison meaningful: any disagreement is then the transit algorithm, not
  the pedestrian layer.

Everything is cached to ``.npz``. Sidewalks do not change between cycles, and
recomputing a constant 288 times a day is the waste this module exists to
remove.

A note on correctness: RAPTOR relaxes footpaths once per round, so the
transfer set must be transitively closed or a two-hop walk will be missed.
Both sources satisfy this automatically, because both are metrics obeying the
triangle inequality — the direct time between two stops is never worse than
going via a third. That holds only because all pairs within the cap are
computed, not just neighbouring ones.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from .gtfs import GtfsData
from .raptor import Footpaths

EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class WalkSource:
    """Walk seconds between coordinate pairs."""

    name = "base"

    def times(self, from_pts, to_pts) -> np.ndarray:
        raise NotImplementedError


class StraightLineWalk(WalkSource):
    name = "straight_line"

    def __init__(self, speed_ms: float = 1.33, detour: float = 1.35):
        self.speed_ms, self.detour = speed_ms, detour

    def times(self, from_pts, to_pts) -> np.ndarray:
        f, t = np.asarray(from_pts, float), np.asarray(to_pts, float)
        km = haversine_km(f[:, None, 0], f[:, None, 1], t[None, :, 0], t[None, :, 1])
        return (km * 1000.0 * self.detour / self.speed_ms).astype(np.int32)


class GraphWalk(WalkSource):
    """Shortest paths on a local OSM pedestrian network.

    Unlike `OtpWalk`, cost does not scale with the number of pairs: one
    bounded Dijkstra per source labels every destination at once. That is what
    makes the walk matrix cheap enough to rebuild whenever the grid changes.
    """

    name = "graph"

    def __init__(self, graph, max_seconds: float = 900.0,
                 max_snap_m: float = 250.0, chunk: int = 50):
        self.graph = graph
        self.max_seconds = float(max_seconds)
        self.max_snap_m = float(max_snap_m)
        self.chunk = int(chunk)

    def times(self, from_pts, to_pts) -> np.ndarray:
        f, t = np.asarray(from_pts, float), np.asarray(to_pts, float)
        w = self.graph.walk_matrix(
            f[:, 0], f[:, 1], t[:, 0], t[:, 1],
            max_seconds=self.max_seconds, chunk=self.chunk,
            max_snap_m=self.max_snap_m,
        )
        # Callers treat negative as "no route"; the graph uses MAX_TIME.
        return np.where(w >= 2**31 - 1, -1, w).astype(np.int32)


class OtpWalk(WalkSource):
    """Walk-only queries against a running OTP.

    Slow per pair, so callers must pre-filter with a straight-line bound: a pair
    whose straight-line walk already exceeds the cap cannot come in under it on
    the street network.
    """

    name = "otp"
    QUERY = (
        "query($f:InputCoordinates!,$t:InputCoordinates!,$d:String!,$tm:String!){"
        "plan(from:$f,to:$t,date:$d,time:$tm,transportModes:[{mode:WALK}],"
        "numItineraries:1){itineraries{duration}}}"
    )

    def __init__(self, base_url="http://127.0.0.1:8080", path="/otp/gtfs/v1",
                 concurrency=8, timeout_s=30.0, date="2026-08-18", time="12:00:00"):
        self.url = base_url.rstrip("/") + path
        self.concurrency, self.timeout_s = concurrency, timeout_s
        self.date, self.time = date, time
        self.failures = 0

    def _one(self, pair) -> int:
        (alat, alon), (blat, blon) = pair
        body = {"query": self.QUERY, "variables": {
            "f": {"lat": float(alat), "lon": float(alon)},
            "t": {"lat": float(blat), "lon": float(blon)},
            "d": self.date, "tm": self.time}}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.loads(r.read())
            its = ((out.get("data") or {}).get("plan") or {}).get("itineraries") or []
            return int(its[0]["duration"]) if its else -1
        except Exception:  # noqa: BLE001
            self.failures += 1
            return -1

    def pairs(self, pairs: list) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return np.fromiter(pool.map(self._one, pairs), dtype=np.int32,
                               count=len(pairs))


def build_footpaths(
    data: GtfsData,
    max_walk_s: int = 300,
    source: Optional[WalkSource] = None,
    cache: Optional[str] = None,
    same_station_s: int = 60,
) -> Footpaths:
    """Stop-to-stop transfers within `max_walk_s`.

    Platforms sharing a parent station are linked at a flat `same_station_s`
    rather than by street distance: walking between Red Line platforms at Park
    Street is an in-station move, and routing it over the sidewalk network
    would produce a nonsense detour up to the surface and back.
    """
    if cache and Path(cache).is_file():
        z = np.load(cache)
        return Footpaths(z["targets"], z["times"], z["ptr"])

    src = source or StraightLineWalk()
    n = data.n_stops
    pts = np.c_[data.stop_lat, data.stop_lon]
    # Official transfers override geometry entirely. An agency knows how long
    # it takes to cross Park Street underground; no street network does.
    official: dict[tuple[int, int], int] = {}
    for a, b, secs in zip(data.transfer_from, data.transfer_to, data.transfer_secs):
        official[(int(a), int(b))] = int(secs)
    # Straight-line bound first: an admissible filter, since street walking is
    # never faster than the crow flies.
    bound_km = max_walk_s * 1.33 / 1000.0
    targets, times, ptr = [], [], [0]
    for s in range(n):
        km = haversine_km(pts[s, 0], pts[s, 1], pts[:, 0], pts[:, 1])
        cand = np.flatnonzero((km <= bound_km) & (np.arange(n) != s))
        # No early exit when `cand` is empty: a stop with no geometric
        # neighbours can still have a declared transfer, and skipping the
        # official pass would silently drop it.
        if len(cand) == 0:
            secs = np.zeros(0, np.int32)
        elif isinstance(src, OtpWalk):
            secs = src.pairs([((pts[s, 0], pts[s, 1]), (pts[c, 0], pts[c, 1]))
                              for c in cand])
        else:
            secs = src.times(pts[s:s + 1], pts[cand])[0]
        seen = set()
        for c, t in zip(cand, secs):
            c = int(c)
            seen.add(c)
            declared = official.get((s, c))
            if declared is not None:
                if declared < 0:
                    continue          # the agency says this transfer is impossible
                # Declared, so exempt from the geometric cap.
                targets.append(c)
                times.append(int(declared))
                continue
            if data.parent[s] >= 0 and data.parent[s] == data.parent[c]:
                t = same_station_s
            if 0 <= t <= max_walk_s:
                targets.append(c)
                times.append(int(t))
        # Official transfers ignore `max_walk_s` entirely. The cap is a
        # computational bound on *candidate* footpaths; a declared transfer is
        # a fact about the network, and MBTA publishes some at 545 s against a
        # 450 s cap. Dropping those would delete real connections. The set is
        # small and bounded, so there is no cost to keeping all of them.
        for (a, b), declared in official.items():
            if a != s or b in seen or declared < 0:
                continue
            targets.append(int(b))
            times.append(int(declared))
        ptr.append(len(targets))

    fp = Footpaths(np.asarray(targets, np.int32), np.asarray(times, np.int32),
                   np.asarray(ptr, np.int64))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, targets=fp.targets, times=fp.times, ptr=fp.ptr)
    return fp


def build_access(
    data: GtfsData,
    origin_lat: float,
    origin_lon: float,
    max_walk_s: int = 900,
    source: Optional[WalkSource] = None,
    cache: Optional[str] = None,
) -> dict[int, int]:
    """Walk seconds from the origin to every stop within reach."""
    if cache and Path(cache).is_file():
        z = np.load(cache)
        return {int(k): int(v) for k, v in zip(z["stops"], z["secs"])}

    src = source or StraightLineWalk()
    pts = np.c_[data.stop_lat, data.stop_lon]
    km = haversine_km(origin_lat, origin_lon, pts[:, 0], pts[:, 1])
    cand = np.flatnonzero(km <= max_walk_s * 1.33 / 1000.0)
    if len(cand) == 0:
        return {}
    if isinstance(src, OtpWalk):
        secs = src.pairs([((origin_lat, origin_lon), (pts[c, 0], pts[c, 1]))
                          for c in cand])
    else:
        secs = src.times(np.array([[origin_lat, origin_lon]]), pts[cand])[0]
    out = {int(c): int(t) for c, t in zip(cand, secs) if 0 <= t <= max_walk_s}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, stops=np.array(list(out)),
                            secs=np.array(list(out.values())))
    return out


class EgressMatrix:
    """Static walk times from stops to output points — the ``W(s,p)`` term.

    Stored sparse, because a stop only reaches points within the egress cap and
    a dense matrix at any useful resolution would be mostly filler. This is the
    piece that makes resolution free: it is computed once, and each cycle then
    needs only the arrival vector.
    """

    def __init__(self, stop_idx: np.ndarray, point_idx: np.ndarray,
                 secs: np.ndarray, n_points: int):
        self.stop_idx, self.point_idx, self.secs = stop_idx, point_idx, secs
        self.n_points = n_points

    @classmethod
    def build(cls, data: GtfsData, points: np.ndarray, max_walk_s: int = 900,
              source: Optional[WalkSource] = None, cache: Optional[str] = None):
        if cache and Path(cache).is_file():
            z = np.load(cache)
            return cls(z["stop_idx"], z["point_idx"], z["secs"], int(z["n_points"]))
        src = source or StraightLineWalk()
        pts = np.c_[data.stop_lat, data.stop_lon]
        bound_km = max_walk_s * 1.33 / 1000.0
        si, pi, sec = [], [], []
        for s in range(data.n_stops):
            km = haversine_km(pts[s, 0], pts[s, 1], points[:, 0], points[:, 1])
            cand = np.flatnonzero(km <= bound_km)
            if not len(cand):
                continue
            if isinstance(src, OtpWalk):
                t = src.pairs([((pts[s, 0], pts[s, 1]),
                                (points[c, 0], points[c, 1])) for c in cand])
            else:
                t = src.times(pts[s:s + 1], points[cand])[0]
            keep = (t >= 0) & (t <= max_walk_s)
            si.append(np.full(keep.sum(), s, np.int32))
            pi.append(cand[keep].astype(np.int32))
            sec.append(t[keep].astype(np.int32))
        out = cls(
            np.concatenate(si) if si else np.zeros(0, np.int32),
            np.concatenate(pi) if pi else np.zeros(0, np.int32),
            np.concatenate(sec) if sec else np.zeros(0, np.int32),
            len(points),
        )
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, stop_idx=out.stop_idx, point_idx=out.point_idx,
                                secs=out.secs, n_points=out.n_points)
        return out

    def compose(self, stop_journey: np.ndarray, direct_walk: np.ndarray,
                unreachable: int) -> np.ndarray:
        """min-plus: T(p) = min(walk(o,p), min_s [journey(s) + W(s,p)]).

        One vectorised scatter-min over the sparse entries. This is the whole
        per-cycle cost of turning stop arrivals into a surface, and it does not
        grow with the number of stops or points beyond the nnz of W.
        """
        out = direct_walk.astype(np.int64).copy()
        j = stop_journey[self.stop_idx]
        cand = j + self.secs
        ok = j < unreachable
        np.minimum.at(out, self.point_idx[ok], cand[ok])
        return out

    def describe(self) -> dict:
        return {
            "nnz": int(len(self.secs)),
            "points": int(self.n_points),
            "stops_used": int(len(np.unique(self.stop_idx))) if len(self.stop_idx) else 0,
            "bytes": int(self.secs.nbytes + self.stop_idx.nbytes + self.point_idx.nbytes),
        }
