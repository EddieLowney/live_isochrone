"""RAPTOR: round-based earliest-arrival search over public transit.

One search from one origin labels **every stop in the network** with its
earliest arrival time. That property is the whole reason this exists — OTP's
`plan` runs an equivalent search and then discards everything except the single
destination asked about, so building an isochrone through it means paying for
the same search once per destination.

The algorithm has no priority queue and no graph traversal. Round *k* computes
journeys using exactly *k* transit legs, and consists of two phases:

1. **Route scan.** For each route touched in round *k−1*, walk its stop list
   once, tracking the earliest boardable trip. Boarding can only improve as you
   move along the route, so the scan never backtracks — that is what keeps the
   cost linear in (routes + stop_times) per round rather than combinatorial.
2. **Transfers.** Relax footpaths from every stop improved in phase 1.

Bounding rounds at K bounds transfers at K−1. Four rounds covers essentially
every real journey.

Range-RAPTOR sweeps a departure window in **descending** order so labels from
a later departure remain valid upper bounds for an earlier one. That yields the
full arrival-time profile for close to the cost of a single run, which matters
because transit travel time is a step function of departure time and one sample
is a snapshot of an arbitrary moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .gtfs import MAX_TIME, GtfsData


@dataclass
class Footpaths:
    """Walkable stop-to-stop connections, in CSR layout.

    RAPTOR requires these to be transitively closed for correctness: if a->b
    and b->c are both walkable, a->c must exist too, because the transfer phase
    relaxes only once per round and will not chain two hops.
    """

    targets: np.ndarray   # flat neighbour stop indices
    times: np.ndarray     # walk seconds, parallel to targets
    ptr: np.ndarray       # ptr[s]:ptr[s+1] slices the two above

    @classmethod
    def empty(cls, n_stops: int) -> "Footpaths":
        return cls(
            np.zeros(0, np.int32), np.zeros(0, np.int32),
            np.zeros(n_stops + 1, np.int64),
        )


@dataclass
class RaptorResult:
    """Earliest arrival per stop, plus how it was reached."""

    arrival: np.ndarray          # seconds since service-day midnight, or MAX_TIME
    rounds_used: np.ndarray      # transit legs taken; 0 means walked from origin
    departure: int

    def reachable(self) -> np.ndarray:
        return self.arrival < MAX_TIME

    def journey_seconds(self) -> np.ndarray:
        """Arrival minus departure. MAX_TIME stays MAX_TIME."""
        out = self.arrival.astype(np.int64) - self.departure
        out[self.arrival >= MAX_TIME] = MAX_TIME
        return out


def raptor(
    data: GtfsData,
    access: dict[int, int],
    departure: int,
    footpaths: Optional[Footpaths] = None,
    max_rounds: int = 4,
    delays: Optional[np.ndarray] = None,
    min_transfer_time: int = 0,
    skipped: Optional[np.ndarray] = None,
) -> RaptorResult:
    """Earliest arrival at every stop, leaving the origin at `departure`.

    `access` maps stop index -> walking seconds from the origin.
    `delays` is a second offset from GTFS-RT, either per trip or per stop-time.
    `skipped` marks stop-times the vehicle will not serve, parallel to
    `trip_arr`. Those cannot be boarded or alighted at — a skipped stop is not
    a late stop, and treating it as one would route passengers onto a train
    that sails past them.
    """
    n = data.n_stops
    best = np.full(n, MAX_TIME, dtype=np.int64)      # best over all rounds
    prev = np.full(n, MAX_TIME, dtype=np.int64)      # best as of round k-1
    rounds_used = np.full(n, -1, dtype=np.int8)

    for s, walk in access.items():
        t = departure + int(walk)
        if t < best[s]:
            best[s] = prev[s] = t
            rounds_used[s] = 0

    marked = np.zeros(n, dtype=bool)
    for s in access:
        marked[s] = True
    if footpaths is not None:
        _relax(best, prev, rounds_used, marked, footpaths, 0)

    # Delays arrive either as one value per trip or, preferably, one per
    # (trip, stop) — delay grows along a run, and a single scalar smears that.
    per_stoptime = delays is not None and len(delays) == len(data.trip_arr)

    trip_dep, trip_arr, tptr = data.trip_dep, data.trip_arr, data.trip_time_ptr
    rstops, rsptr = data.route_stops, data.route_stop_ptr
    rtrips, rtptr = data.route_trips, data.route_trip_ptr
    sroutes, spos, ssptr = data.stop_routes, data.stop_route_pos, data.stop_route_ptr

    for k in range(1, max_rounds + 1):
        if not marked.any():
            break
        # Collect routes touching a marked stop, along with the earliest
        # position at which we could board them.
        board_pos: dict[int, int] = {}
        for s in np.flatnonzero(marked):
            for j in range(ssptr[s], ssptr[s + 1]):
                r, p = int(sroutes[j]), int(spos[j])
                if r not in board_pos or p < board_pos[r]:
                    board_pos[r] = p
        marked[:] = False
        improved = np.zeros(n, dtype=bool)

        for r, start in board_pos.items():
            stops = rstops[rsptr[r]:rsptr[r + 1]]
            trips = rtrips[rtptr[r]:rtptr[r + 1]]
            if len(trips) == 0:
                continue
            cur = -1          # index into `trips` of the trip we are riding
            board_at = -1     # position where we boarded, for the delay offset
            for pos in range(start, len(stops)):
                s = int(stops[pos])
                if cur >= 0:
                    # Normally we alight from the trip we boarded. If that trip
                    # skips this stop, the passenger simply waits for the next
                    # one that serves it — trips are FIFO-ordered, so any later
                    # trip was equally boardable where we got on.
                    idx = cur
                    limit = len(trips)
                    while (
                        idx < limit and skipped is not None
                        and skipped[tptr[int(trips[idx])] + pos]
                    ):
                        idx += 1
                    if idx < limit:
                        t = int(trips[idx])
                        off = tptr[t] + pos
                        arr = int(trip_arr[off])
                        if delays is not None:
                            arr += int(delays[off] if per_stoptime else delays[t])
                        if arr < best[s]:
                            best[s] = arr
                            rounds_used[s] = k
                            improved[s] = True
                # Can we catch something earlier by boarding here instead?
                ready = prev[s]
                if ready >= MAX_TIME:
                    continue
                if cur >= 0:
                    ready += min_transfer_time
                earliest = _earliest_trip(
                    trips, trip_dep, tptr, pos, ready, delays, cur, per_stoptime
                )
                # The earliest trip may not stop here. Advance to one that does
                # rather than treating a skipped stop as boardable.
                while (
                    earliest >= 0 and skipped is not None
                    and earliest < (cur if cur >= 0 else len(trips))
                    and skipped[tptr[int(trips[earliest])] + pos]
                ):
                    earliest += 1
                    if earliest >= (cur if cur >= 0 else len(trips)):
                        earliest = -1
                if earliest >= 0 and earliest != cur:
                    cur, board_at = earliest, pos
        # A stop improved this round can seed further boardings next round.
        prev = best.copy()
        marked |= improved
        if footpaths is not None and improved.any():
            _relax(best, prev, rounds_used, marked, footpaths, k)

    arrival = best.astype(np.int64)
    return RaptorResult(arrival=arrival, rounds_used=rounds_used, departure=departure)


def _earliest_trip(
    trips: np.ndarray, trip_dep: np.ndarray, tptr: np.ndarray,
    pos: int, ready: int, delays: Optional[np.ndarray], cur: int,
    per_stoptime: bool = False,
) -> int:
    """First trip on this route departing `pos` at or after `ready`.

    Trips are sorted by first departure, so a binary search is valid. Bounded
    above by `cur`, since a later trip than the one already boarded can never
    help.
    """
    hi = cur if cur >= 0 else len(trips)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        t = int(trips[mid])
        dep = int(trip_dep[tptr[t] + pos])
        if delays is not None:
            dep += int(delays[tptr[t] + pos] if per_stoptime else delays[t])
        if dep >= ready:
            hi = mid
        else:
            lo = mid + 1
    return lo if lo < (cur if cur >= 0 else len(trips)) else -1


def _relax(best, prev, rounds_used, marked, fp: Footpaths, k: int) -> None:
    """Walk from every improved stop to its footpath neighbours."""
    for s in np.flatnonzero(marked.copy()):
        base = best[s]
        if base >= MAX_TIME:
            continue
        for j in range(fp.ptr[s], fp.ptr[s + 1]):
            t = int(fp.targets[j])
            cand = base + int(fp.times[j])
            if cand < best[t]:
                best[t] = prev[t] = cand
                rounds_used[t] = k
                marked[t] = True


@dataclass
class Profile:
    """Arrival-time distribution per stop across a departure window."""

    departures: np.ndarray            # departure seconds sampled
    journey: np.ndarray               # (n_departures, n_stops) journey seconds
    stop_ids: np.ndarray

    def percentile(self, p: float) -> np.ndarray:
        """Percentile of journey time per stop, ignoring departures from which
        the stop was unreachable.

        Kept in integer seconds throughout: masking with NaN would force a
        float cast that `MAX_TIME` does not survive, and a stop reachable from
        only some departures is a real and common case near the end of service.
        """
        out = np.full(self.journey.shape[1], MAX_TIME, dtype=np.int64)
        for s in range(self.journey.shape[1]):
            col = self.journey[:, s]
            good = col[col < MAX_TIME]
            if good.size:
                out[s] = int(np.percentile(good, p, method="linear"))
        return out

    def now(self) -> np.ndarray:
        """Journey time leaving at the *first* sampled departure.

        This is the headline number for a live map. The median over a departure
        window answers a planning question — "if I leave sometime in the next
        half hour, typically how long" — and deliberately erases the thing this
        map exists to show. Someone who has just missed a bus is genuinely 15
        minutes worse off, and saying otherwise is smoothing away the signal.
        """
        return self.journey[0].copy()

    def time_to_cliff(self, jump_threshold_s: int = 120) -> np.ndarray:
        """Seconds until the journey time jumps by more than the threshold.

        Journey time as a function of departure is a sawtooth: it falls with
        slope -1 while you wait for a departure you can still catch, then leaps
        by roughly a headway the moment you can no longer catch it. The height
        of that leap is the headway; the distance to it is how long you have to
        get moving.

        MAX_TIME means no cliff inside the sampled window — either service is
        frequent enough that missing one barely matters, or the cell is walk-
        dominated and transit is irrelevant to it.
        """
        n_dep, n_stops = self.journey.shape
        out = np.full(n_stops, MAX_TIME, dtype=np.int64)
        if n_dep < 2:
            return out
        base = self.journey[0]
        for k in range(1, n_dep):
            gap = int(self.departures[k] - self.departures[0])
            jumped = (
                (self.journey[k] < MAX_TIME) & (base < MAX_TIME)
                & (self.journey[k] - base > jump_threshold_s)
                & (out == MAX_TIME)
            )
            out[jumped] = gap
            # Leaving later cannot help, so once the sawtooth resets the new
            # low becomes the reference for the next cliff.
            base = np.minimum(base, self.journey[k])
        return out

    def coverage(self) -> np.ndarray:
        """Fraction of sampled departures from which each stop is reachable.

        Below 1.0 means service is sparse enough that whether you get there at
        all depends on when you leave — which the median alone hides.
        """
        return (self.journey < MAX_TIME).mean(axis=0)

    def median(self) -> np.ndarray:
        return self.percentile(50)

    def spread(self) -> np.ndarray:
        """p90 minus p10: the headway penalty for bad timing."""
        hi, lo = self.percentile(90), self.percentile(10)
        out = hi - lo
        out[(hi >= MAX_TIME) | (lo >= MAX_TIME)] = MAX_TIME
        return out

    def reachable(self) -> np.ndarray:
        return (self.journey < MAX_TIME).any(axis=0)


def range_raptor(
    data: GtfsData,
    access: dict[int, int],
    window_start: int,
    window_seconds: int = 1800,
    samples: int = 6,
    footpaths: Optional[Footpaths] = None,
    max_rounds: int = 4,
    delays: Optional[np.ndarray] = None,
    min_transfer_time: int = 0,
    skipped: Optional[np.ndarray] = None,
) -> Profile:
    """Sweep a departure window and return the arrival profile per stop.

    Departures are processed newest-first, which is what makes the sweep cheap:
    a journey optimal for a later departure remains feasible for an earlier one,
    so each run starts from labels that are already close.
    """
    step = window_seconds / max(1, samples)
    departures = np.array(
        [int(window_start + step * i) for i in range(samples)], dtype=np.int64
    )
    journey = np.full((samples, data.n_stops), MAX_TIME, dtype=np.int64)
    for i in range(samples - 1, -1, -1):
        res = raptor(
            data, access, int(departures[i]), footpaths,
            max_rounds, delays, min_transfer_time, skipped,
        )
        journey[i] = res.journey_seconds()
    return Profile(departures=departures, journey=journey, stop_ids=data.stop_ids)
