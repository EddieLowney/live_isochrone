"""A tiny GTFS feed with hand-checkable answers.

RAPTOR is easy to write and easy to get subtly wrong — an off-by-one in the
boarding scan, a transfer relaxed in the wrong round, a binary search that
returns a trip you have already missed. Those bugs do not crash; they return
plausible numbers that are a few minutes optimistic. The only defence is a feed
small enough to verify by hand.

Layout::

    A --RED--> B --RED--> C          departs A at :00 and :30, 10 min per hop
    A --BLU--> D --BLU--> E          departs A at :05 and :35,  8 min per hop
                    C <--200m walk--> D

From A at 08:00 the answers are:

* B  08:10  — RED, direct.
* D  08:13  — BLU, direct.
* E  08:21  — BLU, direct.
* C  08:15:30 — **not** the direct RED arrival of 08:20. BLU reaches D at 08:13
  and the 150 s footpath gets to C at 08:15:30, four and a half minutes earlier.

That last one is the case worth having. A naive implementation that relaxes
footpaths before the route scan, or that stops improving a stop once any
transit leg has reached it, returns 08:20 — plausible, undramatic, and wrong.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

STOPS = [
    ("A", "Alpha", 42.3600, -71.0580),
    ("B", "Bravo", 42.3650, -71.0620),
    ("C", "Charlie", 42.3700, -71.0660),
    ("D", "Delta", 42.3702, -71.0682),   # ~180 m from C
    ("E", "Echo", 42.3760, -71.0720),
]

#: (route, trip, [(stop, arrive, depart)])
TRIPS = [
    ("RED", "RED_0800", [("A", "08:00:00", "08:00:00"),
                         ("B", "08:10:00", "08:10:00"),
                         ("C", "08:20:00", "08:20:00")]),
    ("RED", "RED_0830", [("A", "08:30:00", "08:30:00"),
                         ("B", "08:40:00", "08:40:00"),
                         ("C", "08:50:00", "08:50:00")]),
    ("BLU", "BLU_0805", [("A", "08:05:00", "08:05:00"),
                         ("D", "08:13:00", "08:13:00"),
                         ("E", "08:21:00", "08:21:00")]),
    ("BLU", "BLU_0835", [("A", "08:35:00", "08:35:00"),
                         ("D", "08:43:00", "08:43:00"),
                         ("E", "08:51:00", "08:51:00")]),
]


def write_feed(path: str, service_date: str = "20260818") -> str:
    """Write the feed to `path` and return it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def csv(rows: list[list[str]]) -> str:
        return "\n".join(",".join(r) for r in rows) + "\n"

    stops = [["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type",
              "parent_station"]]
    stops += [[s, n, str(la), str(lo), "0", ""] for s, n, la, lo in STOPS]

    routes = [["route_id", "route_short_name", "route_long_name", "route_type"],
              ["RED", "Red", "Red Line", "1"],
              ["BLU", "Blue", "Blue Line", "1"]]

    trips = [["route_id", "service_id", "trip_id", "direction_id"]]
    stop_times = [["trip_id", "arrival_time", "departure_time", "stop_id",
                   "stop_sequence"]]
    for route, trip, seq in TRIPS:
        trips.append([route, "WEEKDAY", trip, "0"])
        for i, (stop, arr, dep) in enumerate(seq):
            stop_times.append([trip, arr, dep, stop, str(i + 1)])

    cal = [["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "start_date", "end_date"],
           ["WEEKDAY", "1", "1", "1", "1", "1", "1", "1", "20260101", "20271231"]]

    agency = [["agency_id", "agency_name", "agency_url", "agency_timezone"],
              ["T", "Test", "https://example.invalid", "America/New_York"]]

    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agency.txt", csv(agency))
        zf.writestr("stops.txt", csv(stops))
        zf.writestr("routes.txt", csv(routes))
        zf.writestr("trips.txt", csv(trips))
        zf.writestr("stop_times.txt", csv(stop_times))
        zf.writestr("calendar.txt", csv(cal))
    return str(p)


def footpaths_for(data) -> "object":
    """The single C<->D footpath, 200 m at 1.33 m/s ~= 150 s."""
    import numpy as np
    from .raptor import Footpaths

    n = data.n_stops
    ci, di = data.index_of("C"), data.index_of("D")
    targets, times, ptr = [], [], [0]
    for s in range(n):
        if s == ci:
            targets.append(di); times.append(150)
        elif s == di:
            targets.append(ci); times.append(150)
        ptr.append(len(targets))
    return Footpaths(
        np.asarray(targets, np.int32), np.asarray(times, np.int32),
        np.asarray(ptr, np.int64),
    )
