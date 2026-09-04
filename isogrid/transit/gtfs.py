"""GTFS to RAPTOR data structures.

RAPTOR is route-oriented, not graph-oriented, and that shapes everything here.
Its "route" is not a GTFS route: it is a maximal set of trips sharing an
identical stop sequence, so that scanning one route means walking a single stop
list once while tracking the earliest boardable trip. The MBTA's Route 1 becomes
several RAPTOR routes — one per distinct pattern and direction.

Everything ends up in flat numpy arrays with offset indices, because the inner
loop touches these millions of times and Python object access would dominate.
Parsing is cached to an ``.npz`` keyed by feed mtime and service date, so the
one-off cost is paid once per day rather than per cycle.

Times are seconds since midnight of the service day and legitimately exceed
86400: a trip departing 00:40 on a Saturday night belongs to Friday's service
day at 24:40:00. Dropping those would silently delete late-night service.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

MAX_TIME = np.int32(2**31 - 1)  # RAPTOR's "unreachable"


def parse_gtfs_time(s: str) -> int:
    """'25:13:00' -> 90780. Hours past 24 are normal, not errors."""
    if not s:
        return -1
    p = s.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])


def _rows(zf: zipfile.ZipFile, name: str) -> Iterator[dict]:
    try:
        raw = zf.read(name)
    except KeyError:
        return
    # utf-8-sig: GTFS feeds frequently carry a BOM, which corrupts the first
    # header name and makes every lookup on it fail silently.
    yield from csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))


@dataclass
class GtfsData:
    """RAPTOR's working set. All arrays, all index-addressed."""

    service_date: date
    #: stop_id strings, index == RAPTOR stop index
    stop_ids: np.ndarray
    stop_lat: np.ndarray
    stop_lon: np.ndarray
    stop_names: np.ndarray
    #: parent station index per stop, or -1. Transfers between platforms of one
    #: station should be free; without this they look like street walks.
    parent: np.ndarray

    #: route_stops[route_stop_ptr[r]:route_stop_ptr[r+1]] = stop indices in order
    route_stops: np.ndarray
    route_stop_ptr: np.ndarray
    #: route_trips[route_trip_ptr[r]:route_trip_ptr[r+1]] = trip indices,
    #: sorted by departure at the first stop
    route_trips: np.ndarray
    route_trip_ptr: np.ndarray
    #: arrival/departure seconds, flat: trip_time_ptr[t] + stop_index
    trip_arr: np.ndarray
    trip_dep: np.ndarray
    trip_time_ptr: np.ndarray
    trip_ids: np.ndarray
    trip_route: np.ndarray
    #: which RAPTOR routes serve a stop, and at which position along them
    stop_routes: np.ndarray
    stop_route_pos: np.ndarray
    stop_route_ptr: np.ndarray
    #: gtfs route short/long name per RAPTOR route, for diagnostics
    route_labels: np.ndarray
    #: Official transfers from transfers.txt, expanded to platform pairs.
    #: `transfer_secs` of -1 means the agency says the transfer is impossible;
    #: any other value overrides the geometric estimate. Agencies know things
    #: about station interiors that no street network contains.
    transfer_from: np.ndarray
    transfer_to: np.ndarray
    transfer_secs: np.ndarray

    @property
    def n_stops(self) -> int:
        return len(self.stop_ids)

    @property
    def n_routes(self) -> int:
        return len(self.route_stop_ptr) - 1

    @property
    def n_trips(self) -> int:
        return len(self.trip_ids)

    def index_of(self, stop_id: str) -> int:
        hit = np.where(self.stop_ids == stop_id)[0]
        return int(hit[0]) if len(hit) else -1

    def describe(self) -> dict:
        return {
            "service_date": self.service_date.isoformat(),
            "stops": self.n_stops,
            "raptor_routes": self.n_routes,
            "trips": self.n_trips,
            "stop_times": int(self.trip_time_ptr[-1]),
            "official_transfers": int(len(self.transfer_secs)),
            "forbidden_transfers": int((self.transfer_secs < 0).sum()),
        }


def active_services(zf: zipfile.ZipFile, on: date) -> set[str]:
    """Service ids running on a date, from calendar.txt plus exceptions."""
    weekday = ["monday", "tuesday", "wednesday", "thursday", "friday",
               "saturday", "sunday"][on.weekday()]
    ymd = on.strftime("%Y%m%d")
    active: set[str] = set()
    for r in _rows(zf, "calendar.txt"):
        if r.get(weekday) == "1" and r["start_date"] <= ymd <= r["end_date"]:
            active.add(r["service_id"])
    # calendar_dates overrides calendar, in both directions.
    for r in _rows(zf, "calendar_dates.txt"):
        if r["date"] != ymd:
            continue
        if r["exception_type"] == "1":
            active.add(r["service_id"])
        elif r["exception_type"] == "2":
            active.discard(r["service_id"])
    return active


def load_gtfs(
    path: str,
    on: Optional[date] = None,
    cache_dir: Optional[str] = None,
    rebuild: bool = False,
) -> GtfsData:
    """Parse a GTFS zip into RAPTOR arrays, caching the result."""
    on = on or datetime.now().date()
    src = Path(path)
    cache = None
    if cache_dir:
        key = (f"{src.stem}-{CACHE_FORMAT}-{on.isoformat()}"
               f"-{int(src.stat().st_mtime)}.npz")
        cache = Path(cache_dir) / key
        if cache.is_file() and not rebuild:
            return _from_npz(cache, on)

    data = _build(src, on)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _to_npz(cache, data)
    return data


def _build(src: Path, on: date) -> GtfsData:
    with zipfile.ZipFile(src) as zf:
        services = active_services(zf, on)

        stop_ids, lats, lons, names, parent_of = [], [], [], [], []
        idx: dict[str, int] = {}
        for r in _rows(zf, "stops.txt"):
            # location_type 1 is a station, 2+ are entrances and generic nodes.
            # Only boardable platforms belong in the RAPTOR stop set.
            if r.get("location_type") not in ("", "0", None):
                continue
            idx[r["stop_id"]] = len(stop_ids)
            stop_ids.append(r["stop_id"])
            lats.append(float(r["stop_lat"]))
            lons.append(float(r["stop_lon"]))
            names.append(r.get("stop_name", ""))
            parent_of.append(r.get("parent_station", "") or "")

        route_label: dict[str, str] = {}
        for r in _rows(zf, "routes.txt"):
            route_label[r["route_id"]] = (
                r.get("route_short_name") or r.get("route_long_name") or r["route_id"]
            )

        trip_service, trip_route_id = {}, {}
        for r in _rows(zf, "trips.txt"):
            if r["service_id"] in services:
                trip_service[r["trip_id"]] = r["service_id"]
                trip_route_id[r["trip_id"]] = r["route_id"]

        # stop_times is the large file — a single pass, grouped by trip.
        per_trip: dict[str, list[tuple[int, int, int, int]]] = {}
        for r in _rows(zf, "stop_times.txt"):
            tid = r["trip_id"]
            if tid not in trip_service:
                continue
            si = idx.get(r["stop_id"])
            if si is None:
                continue
            arr = parse_gtfs_time(r["arrival_time"] or r["departure_time"])
            dep = parse_gtfs_time(r["departure_time"] or r["arrival_time"])
            if arr < 0 or dep < 0:
                continue  # frequency-based trips need interpolation; skipped
            per_trip.setdefault(tid, []).append(
                (int(r["stop_sequence"]), si, arr, dep)
            )

    # Group trips into RAPTOR routes by identical stop pattern.
    patterns: dict[tuple, list[str]] = {}
    for tid, rows in per_trip.items():
        rows.sort()
        if len(rows) < 2:
            continue
        pattern = (trip_route_id[tid], tuple(r[1] for r in rows))
        patterns.setdefault(pattern, []).append(tid)

    route_stops, route_stop_ptr = [], [0]
    route_trips, route_trip_ptr = [], [0]
    trip_arr, trip_dep, trip_time_ptr = [], [], [0]
    trip_ids, trip_route, route_labels = [], [], []

    for r_idx, ((gtfs_route, stops), tids) in enumerate(sorted(patterns.items())):
        route_stops.extend(stops)
        route_stop_ptr.append(len(route_stops))
        route_labels.append(route_label.get(gtfs_route, gtfs_route))
        # Sorting by departure at the first stop is what lets the scan pick the
        # earliest boardable trip by walking forward and never backtracking.
        ordered = sorted(tids, key=lambda t: per_trip[t][0][3])
        for tid in ordered:
            rows = per_trip[tid]
            route_trips.append(len(trip_ids))
            trip_ids.append(tid)
            trip_route.append(r_idx)
            trip_arr.extend(x[2] for x in rows)
            trip_dep.extend(x[3] for x in rows)
            trip_time_ptr.append(len(trip_arr))
        route_trip_ptr.append(len(route_trips))

    n_stops = len(stop_ids)
    by_stop: dict[int, list[tuple[int, int]]] = {}
    for r in range(len(route_stop_ptr) - 1):
        for pos, s in enumerate(route_stops[route_stop_ptr[r]:route_stop_ptr[r + 1]]):
            by_stop.setdefault(s, []).append((r, pos))
    stop_routes, stop_route_pos, stop_route_ptr = [], [], [0]
    for s in range(n_stops):
        for r, pos in by_stop.get(s, ()):
            stop_routes.append(r)
            stop_route_pos.append(pos)
        stop_route_ptr.append(len(stop_routes))

    parent = np.full(n_stops, -1, dtype=np.int32)
    for i, p in enumerate(parent_of):
        if p and p in idx:
            parent[i] = idx[p]

    tr_from, tr_to, tr_secs = _load_transfers(src, idx, parent_of)

    a = lambda x, t=np.int32: np.asarray(x, dtype=t)  # noqa: E731
    return GtfsData(
        service_date=on,
        stop_ids=np.asarray(stop_ids, dtype=object),
        stop_lat=np.asarray(lats, dtype=np.float64),
        stop_lon=np.asarray(lons, dtype=np.float64),
        stop_names=np.asarray(names, dtype=object),
        parent=parent,
        route_stops=a(route_stops), route_stop_ptr=a(route_stop_ptr),
        route_trips=a(route_trips), route_trip_ptr=a(route_trip_ptr),
        trip_arr=a(trip_arr), trip_dep=a(trip_dep),
        trip_time_ptr=a(trip_time_ptr, np.int64),
        trip_ids=np.asarray(trip_ids, dtype=object),
        trip_route=a(trip_route),
        stop_routes=a(stop_routes), stop_route_pos=a(stop_route_pos),
        stop_route_ptr=a(stop_route_ptr),
        route_labels=np.asarray(route_labels, dtype=object),
        transfer_from=a(tr_from), transfer_to=a(tr_to), transfer_secs=a(tr_secs),
    )


def _load_transfers(src: Path, idx: dict[str, int], parent_of: list[str]):
    """Read transfers.txt and expand it to platform-level pairs.

    Two wrinkles worth knowing:

    * Agencies often reference a **station**, not a platform. Those rows are
      expanded to every child platform pair, since the RAPTOR stop set holds
      only boardable platforms.
    * A stop pair can appear more than once, for instance with route-specific
      qualifiers that RAPTOR's stop-to-stop footpaths cannot represent. The
      most permissive row wins: the smallest time, and a pair is treated as
      forbidden only when every row says so.

    transfer_type: 0 recommended, 1 timed (vehicle waits), 2 needs
    min_transfer_time, 3 not possible, and 4/5 in-seat transfers where the
    rider stays aboard as the vehicle continues under a new trip id.

    Types 4 and 5 are deliberately ignored. They link *trips*, not stops, so a
    stop-to-stop footpath cannot express them — writing one would let any trip
    at that stop transfer freely, which is far more permissive than the feed
    intends. Ignoring them is pessimistic by at most one RAPTOR round.
    """
    children: dict[str, list[int]] = {}
    for sid, i in idx.items():
        children.setdefault(sid, []).append(i)
    for sid, i in idx.items():
        par = parent_of[i]
        if par:
            children.setdefault(par, []).append(i)

    best: dict[tuple[int, int], int] = {}
    with zipfile.ZipFile(src) as zf:
        for r in _rows(zf, "transfers.txt"):
            a_ids = children.get((r.get("from_stop_id") or "").strip(), [])
            b_ids = children.get((r.get("to_stop_id") or "").strip(), [])
            if not a_ids or not b_ids:
                continue
            ttype = (r.get("transfer_type") or "0").strip() or "0"
            if ttype in ("4", "5"):
                continue          # in-seat, trip-linked: see the docstring
            raw = (r.get("min_transfer_time") or "").strip()
            if not raw.isdigit():
                # MBTA also publishes min_walk_time; min_transfer_time is the
                # binding figure when both are present.
                raw = (r.get("min_walk_time") or "").strip()
            if ttype == "3":
                secs = -1
            elif ttype == "1":
                secs = 0          # timed: the connecting vehicle waits
            elif raw.isdigit():
                secs = int(raw)
            else:
                continue          # recommended with no time: leave to geometry
            for a in a_ids:
                for b in b_ids:
                    if a == b:
                        continue
                    cur = best.get((a, b))
                    if cur is None:
                        best[(a, b)] = secs
                    elif cur < 0:
                        best[(a, b)] = secs      # any allowed row beats a ban
                    elif secs >= 0:
                        best[(a, b)] = min(cur, secs)
    if not best:
        return [], [], []
    keys = sorted(best)
    return ([k[0] for k in keys], [k[1] for k in keys], [best[k] for k in keys])


_ARRAYS = ("stop_ids", "stop_lat", "stop_lon", "stop_names", "parent",
           "route_stops", "route_stop_ptr", "route_trips", "route_trip_ptr",
           "trip_arr", "trip_dep", "trip_time_ptr", "trip_ids", "trip_route",
           "stop_routes", "stop_route_pos", "stop_route_ptr", "route_labels",
           "transfer_from", "transfer_to", "transfer_secs")

#: Bumped when the array set changes, so old caches are rebuilt rather than
#: failing on a missing key.
CACHE_FORMAT = "v2"


def _to_npz(path: Path, d: GtfsData) -> None:
    np.savez_compressed(path, **{k: getattr(d, k) for k in _ARRAYS})


def _from_npz(path: Path, on: date) -> GtfsData:
    z = np.load(path, allow_pickle=True)
    return GtfsData(service_date=on, **{k: z[k] for k in _ARRAYS})
