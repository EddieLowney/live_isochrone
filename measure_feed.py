#!/usr/bin/env python3
"""How often do MBTA vehicle positions actually update?

Three different numbers get conflated when people say "the feed updates every
30 seconds", and interpolation depends on all three:

1. **Publish cadence** — how often the `.pb` file itself changes. Sets how
   often it is worth polling, and nothing else.
2. **Per-vehicle fix interval** — how often an individual vehicle's own
   timestamp advances. This is what an interpolation buffer has to span, and
   it is not the same as the publish cadence: the feed can republish every
   5 seconds while any given bus reports every 45.
3. **Arrival lag** — how old a position already is when it reaches you. A
   30 second display buffer on positions that arrive 20 seconds stale means
   showing the world 50 seconds late.

Sizing the buffer from (1) rather than (2) plus (3) is the mistake this script
exists to prevent.

Run it for five minutes:  python3 measure_feed.py
Longer is better:         python3 measure_feed.py --minutes 15
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
import sys
import time
import zipfile
from collections import defaultdict

sys.path.insert(0, ".")

from isogrid.transit.realtime import VehicleFetcher  # noqa: E402

URL = "https://cdn.mbta.com/realtime/VehiclePositions.pb"
GTFS = "otp/mbta.gtfs.zip"
ROUTE_TYPE = {"0": "light rail", "1": "subway", "2": "commuter rail",
              "3": "bus", "4": "ferry"}


def route_kinds(path: str) -> dict:
    """route_id -> readable mode, so the breakdown means something."""
    try:
        with zipfile.ZipFile(path) as zf:
            rows = csv.DictReader(io.StringIO(
                zf.read("routes.txt").decode("utf-8-sig")))
            return {r["route_id"]: ROUTE_TYPE.get(r.get("route_type", ""), "other")
                    for r in rows}
    except Exception:
        return {}


def haversine_m(a_lat, a_lon, b_lat, b_lon) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(min(1.0, h)))


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, int(len(s) * p / 100))
    return s[k]


def summarise(name, xs, unit="s"):
    if not xs:
        print(f"  {name:<16} no data")
        return
    print(f"  {name:<16} n={len(xs):<6} median {statistics.median(xs):6.1f}{unit}"
          f"  p90 {pct(xs,90):6.1f}{unit}  p95 {pct(xs,95):6.1f}{unit}"
          f"  max {max(xs):6.1f}{unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--poll", type=float, default=2.0,
                    help="seconds between polls; conditional, so a 304 is cheap")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    kinds = route_kinds(GTFS)
    print(f"polling {args.url}")
    print(f"every {args.poll:.0f}s for {args.minutes:.0f} min "
          f"({'routes.txt loaded' if kinds else 'no routes.txt — modes unavailable'})\n")

    fetcher = VehicleFetcher(args.url)
    last_ts: dict = {}
    last_pos: dict = {}
    intervals: dict = defaultdict(list)
    distances: dict = defaultdict(list)
    lags: dict = defaultdict(list)
    change_times: list = []
    polls = 0
    end = time.time() + args.minutes * 60

    try:
        while time.time() < end:
            polls += 1
            t_poll = time.time()
            try:
                vehicles = fetcher.vehicles()
            except RuntimeError as exc:
                print(f"  fetch failed: {exc}")
                time.sleep(args.poll)
                continue

            advanced = 0
            for v in vehicles:
                kind = kinds.get(v.route_id or "", "unknown")
                if v.timestamp:
                    lags[kind].append(t_poll - v.timestamp)
                prev = last_ts.get(v.id)
                if v.timestamp and prev and v.timestamp > prev:
                    intervals[kind].append(v.timestamp - prev)
                    advanced += 1
                    p = last_pos.get(v.id)
                    if p:
                        distances[kind].append(haversine_m(p[0], p[1], v.lat, v.lon))
                if v.timestamp:
                    last_ts[v.id] = v.timestamp
                last_pos[v.id] = (v.lat, v.lon)
            if advanced:
                change_times.append(t_poll)

            left = end - time.time()
            print(f"\r  poll {polls:>4}  vehicles {len(vehicles):>4}  "
                  f"advanced {advanced:>4}  {left:>5.0f}s left", end="", flush=True)
            time.sleep(max(0.0, args.poll - (time.time() - t_poll)))
    except KeyboardInterrupt:
        print("\n  stopped early")

    print("\n" + "=" * 68)
    d = fetcher.describe()
    print(f"polls {polls}   real fetches {d['fetches']}   304s {d['not_modified']}"
          f"   {d['mb_downloaded']} MB")
    if len(change_times) > 1:
        gaps = [b - a for a, b in zip(change_times, change_times[1:])]
        print(f"\n1. PUBLISH CADENCE — how often the feed content changed")
        summarise("feed change", gaps)

    print(f"\n2. PER-VEHICLE FIX INTERVAL — what the buffer must span")
    allint = []
    for kind in sorted(intervals, key=lambda k: -len(intervals[k])):
        summarise(kind, intervals[kind])
        allint += intervals[kind]
    summarise("ALL", allint)

    print(f"\n3. ARRIVAL LAG — how stale a position already is on arrival")
    alllag = []
    for kind in sorted(lags, key=lambda k: -len(lags[k])):
        summarise(kind, lags[kind])
        alllag += lags[kind]
    summarise("ALL", alllag)

    print(f"\n4. DISTANCE MOVED BETWEEN FIXES")
    for kind in sorted(distances, key=lambda k: -len(distances[k])):
        summarise(kind, distances[kind], unit="m")

    if allint:
        buf = pct(allint, 95)
        lag = statistics.median(alllag) if alllag else 0.0
        print("\n" + "=" * 68)
        print(f"buffer suggestion: {buf:.0f}s  (p95 of fix intervals)")
        print(f"  plus median arrival lag {lag:.0f}s")
        print(f"  -> the map would show the world about {buf + lag:.0f}s behind")
        under = sum(1 for x in allint if x > buf) / len(allint) * 100
        print(f"  underrun rate at that buffer: {under:.1f}% of fixes")
        if pct(allint, 50) * 3 < pct(allint, 95):
            print("\n  note: p95 is far above the median, so the interval is very "
                  "uneven.\n  A buffer sized to p95 costs everyone latency to "
                  "cover a few stragglers —\n  consider sizing to p90 and easing "
                  "through the underruns instead.")


if __name__ == "__main__":
    main()
