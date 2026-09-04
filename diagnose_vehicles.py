#!/usr/bin/env python3
"""Trace a few vehicles through the whole interpolation pipeline.

Prints, per fix: the raw feed values, what the projection made of them, which
guard (if any) suppressed the movement, and — the part that matters — **whether
each published leg begins where the previous one ended**.

That last column is the one to read. The browser plays legs back to back and
jumps straight to the start of the next one, so a non-zero join gap is not a
rounding artefact: it is exactly the distance the marker teleports. A run with
gaps is a run that rubber-bands, whatever the code intends.

The summary also reports leg-seconds published against wall-clock seconds
elapsed. Those should track: below 1.0 means playback runs dry and halts,
above 1.0 means legs arrive faster than they can be played and the browser
falls further behind every poll until the queue overflows.

  python3 diagnose_vehicles.py                 # 2 min, 6 rail vehicles
  python3 diagnose_vehicles.py --minutes 4 --vehicles 3 --modes bus
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import time
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, ".")

from isogrid.transit.realtime import VehicleFetcher  # noqa: E402
from isogrid.transit.shapes import ROUTE_KIND  # noqa: E402
from isogrid.transit.vehicles import (  # noqa: E402
    ShapeIndex, VehicleTracker, _local_scale,
)

URL = "https://cdn.mbta.com/realtime/VehiclePositions.pb"
GTFS = "otp/mbta.gtfs.zip"


def route_kinds(path):
    with zipfile.ZipFile(path) as zf:
        rows = csv.DictReader(io.StringIO(zf.read("routes.txt").decode("utf-8-sig")))
        return {r["route_id"]: ROUTE_KIND.get(r.get("route_type", ""), "other")
                for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--vehicles", type=int, default=6)
    # Spelled as the config spells it; VehicleTracker normalises. "light rail"
    # not matching "light_rail" is what silently hid the entire Green Line.
    ap.add_argument("--modes", default="subway,light rail")
    ap.add_argument("--origin", default="42.343291,-71.090291")
    args = ap.parse_args()

    lat0, lon0 = (float(x) for x in args.origin.split(","))
    keep = {m.strip() for m in args.modes.split(",")}
    kinds = route_kinds(GTFS)

    print("indexing shapes...", flush=True)
    index = ShapeIndex.build(GTFS, lat0, lon0, 12.0,
                             cache="data/raptor/diag-shapeindex.npz")
    print(f"  {index.describe()}\n")

    tracker = VehicleTracker(index, max_age_s=180.0,
                             route_kinds=kinds, keep_kinds=keep)
    fetcher = VehicleFetcher(URL)
    mx, my = _local_scale(lat0)

    watch: list = []
    history: dict = defaultdict(list)   # vid -> rows
    legs_seen: dict = defaultdict(dict)  # vid -> {leg_id: leg}
    guards = Counter()
    gaps: list = []                    # (metres, vid, leg_id) for every handover
    end = time.time() + args.minutes * 60
    polls = 0

    while time.time() < end:
        now = time.time()
        try:
            vehicles = fetcher.vehicles()
        except RuntimeError as exc:
            print(f"fetch failed: {exc}")
            break
        polls += 1
        want = [v for v in vehicles if tracker.keep(v)]
        if not watch:
            want.sort(key=lambda v: math.hypot(v.lat - lat0, v.lon - lon0))
            watch = [v.id for v in want[:args.vehicles]]
            print(f"watching {len(watch)} of {len(want)} {'/'.join(sorted(keep))} "
                  f"vehicles\n")

        before = {v.id: (tracker.tracks[v.id].last_s
                         if v.id in tracker.tracks else None) for v in want}
        tracker.update(want, now)
        guards.update(tracker.stats)
        segs = {f["id"]: f for f in tracker.segments(want, now)}

        # Leg continuity, over every vehicle rather than just the watched few:
        # a 1 km teleport on a train nobody happened to be watching is still a
        # 1 km teleport.
        for vid, feat in segs.items():
            known = legs_seen[vid]
            for leg in feat["legs"]:
                prev_id = max(known) if known else None
                if leg["id"] in known:
                    if known[leg["id"]]["path"] != leg["path"]:
                        guards["MUTATED_LEG"] += 1
                    continue
                if prev_id is not None and leg["id"] > prev_id:
                    p = known[prev_id]["path"][-1]
                    q = leg["path"][0]
                    gaps.append((math.hypot((q[0] - p[0]) * mx,
                                            (q[1] - p[1]) * my), vid, leg["id"]))
                known[leg["id"]] = leg

        for v in want:
            if v.id not in watch:
                continue
            tk = tracker.tracks.get(v.id)
            if tk is None:
                continue
            seg = segs.get(v.id)
            newest = seg["legs"][-1] if seg and seg["legs"] else None
            history[v.id].append({
                "wall": now, "ts": v.timestamp, "trip": v.trip_id,
                "dir": v.direction_id, "status": v.status,
                "s": tk.last_s, "prev_s": before.get(v.id),
                "off": tk.off_shape_m, "shape": tk.shape_id,
                "nlegs": len(seg["legs"]) if seg else 0,
                "dur": newest["dur"] if newest else None,
                "pts": len(newest["path"]) if newest else 0,
                "on_track": newest["on_track"] if newest else None,
            })
        time.sleep(max(0.0, args.poll - (time.time() - now)))

    # ---- per-vehicle trace -------------------------------------------------
    for vid in watch:
        rows = history.get(vid, [])
        if not rows:
            continue
        print("=" * 82)
        print(f"{vid}   shape={rows[-1]['shape']}")
        print(f"{'wall':>6} {'age':>5} {'status':>10} {'s (m)':>9} {'d s':>7} "
              f"{'off':>5} {'legs':>4} {'dur':>5} {'pts':>4} {'trk':>4} trip")
        t0 = rows[0]["wall"]
        for r in rows:
            age = "" if r["ts"] is None else f"{r['wall'] - r['ts']:.0f}s"
            s = "" if r["s"] is None else f"{r['s']:.0f}"
            ds = ("" if r["s"] is None or r["prev_s"] is None
                  else f"{r['s'] - r['prev_s']:+.0f}")
            dur = "" if r["dur"] is None else f"{r['dur']:.0f}s"
            trk = {True: "yes", False: "NO", None: ""}[r["on_track"]]
            print(f"{r['wall']-t0:>6.0f} {age:>5} {str(r['status']):>10} "
                  f"{s:>9} {ds:>7} {r['off']:>5.0f} {r['nlegs']:>4} "
                  f"{dur:>5} {r['pts']:>4} {trk:>4} "
                  f"{(r['trip'] or '')[-10:]}")
        trips = {r["trip"] for r in rows}
        if len(trips) > 1:
            print(f"  !! trip_id changed {len(trips)} times — each change "
                  f"clears the track, and the marker jumps to the new position")

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 82)
    print(f"tracker: {tracker.describe()}")
    print("\nguards fired (summed over polls):")
    for k in ("on_shape", "off_shape", "no_shape", "held", "trip_change",
              "reshape", "MUTATED_LEG"):
        print(f"  {k:12s} {guards.get(k, 0)}")
    if guards.get("no_shape"):
        print("  -> no_shape means straight-line interpolation: the vehicle "
              "cuts every corner.\n     Usually an ADDED trip whose route or "
              "direction did not resolve either.")
    if guards.get("MUTATED_LEG"):
        print("  -> a published leg CHANGED. The past moved under a vehicle "
              "midway through\n     playing it, which the browser can only "
              "render as a bounce.")

    if gaps:
        bad = [g for g in gaps if g[0] > 5]
        print(f"\nleg handovers: {len(gaps)}   with a join gap > 5 m: "
              f"{len(bad)} ({100*len(bad)/len(gaps):.1f}%)")
        for d, vid, lid in sorted(bad, reverse=True)[:8]:
            print(f"   {d:8.0f} m  {vid}  at leg {lid}")
        if bad:
            print("  -> each gap is a teleport of that distance: the browser "
                  "jumps to the\n     start of the next leg with nothing to "
                  "interpolate across.")

    durs = [l["dur"] for v in legs_seen.values() for l in v.values()]
    if durs:
        durs.sort()
        vehicle_seconds = polls * args.poll * max(1, len(legs_seen))
        print(f"\nlegs published: {len(durs)}   duration median "
              f"{durs[len(durs)//2]:.0f}s  p90 {durs[9*len(durs)//10]:.0f}s  "
              f"max {durs[-1]:.0f}s")
        ratio = sum(durs) / vehicle_seconds
        print(f"leg-seconds {sum(durs):.0f} vs vehicle-seconds elapsed "
              f"{vehicle_seconds:.0f}  (ratio {ratio:.2f})")
        # Short runs sit below 1.0 for an honest reason: a vehicle's first fix
        # closes no leg, and vehicles with stale timestamps close none at all.
        if ratio < 0.75:
            print("  -> playback runs dry: the marker halts at the last known "
                  "point and waits.")
        elif ratio > 1.15:
            print("  -> legs outrun playback: the browser falls further behind "
                  "every poll\n     until the queue overflows and drops one, "
                  "which is a jump forward.")
        ontrack = sum(1 for v in legs_seen.values() for l in v.values()
                      if l["on_track"])
        print(f"legs on real track geometry: {ontrack}/{len(durs)} "
              f"= {100*ontrack/len(durs):.0f}%")


if __name__ == "__main__":
    main()
