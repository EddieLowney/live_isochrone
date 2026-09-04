#!/usr/bin/env python3
"""Cross-check RAPTOR against OTP, and the surface against itself.

Nothing so far has compared this system to anything external. The unit tests
show the algorithm does what the algorithm is supposed to do; they cannot show
that the numbers on the map are true. OTP is an independent implementation
reading the same feeds, so disagreement between them is the cheapest available
evidence.

Interpreting the gap:

* **RAPTOR consistently faster by 5-20%** — expected, and not a bug. RAPTOR is
  configured with `walk_source: straight_line`, which understates walking
  distance. Switching to `otp` should collapse most of it.
* **Scattered large disagreements, either direction** — worth investigating.
  Likely candidates are transfer handling, the egress walk cap, or stops
  reachable in one engine and not the other.
* **RAPTOR slower** — suspicious. RAPTOR should never lose to OTP on the same
  data, since optimistic walking can only help it.

Run with both servers up:  python3 validate.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

MAP_URL = "http://127.0.0.1:8001"
OTP_URL = "http://127.0.0.1:8080/otp/gtfs/v1"
TZ = ZoneInfo("America/New_York")
N_SAMPLES = 25

PLAN = """
query($f:InputCoordinates!,$t:InputCoordinates!,$d:String!,$tm:String!){
  plan(from:$f,to:$t,date:$d,time:$tm,
       transportModes:[{mode:WALK},{mode:TRANSIT}],numItineraries:1){
    itineraries{ duration walkTime waitingTime legs{ mode duration } }
  }
}"""


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def otp_plan(origin, dest, when):
    body = {"query": PLAN, "variables": {
        "f": {"lat": origin[0], "lon": origin[1]},
        "t": {"lat": dest[0], "lon": dest[1]},
        "d": when.strftime("%Y-%m-%d"), "tm": when.strftime("%H:%M:%S")}}
    req = urllib.request.Request(
        OTP_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    its = ((out.get("data") or {}).get("plan") or {}).get("itineraries") or []
    return min((i["duration"] for i in its), default=None)


def main():
    try:
        gj = get(f"{MAP_URL}/api/geojson")
        cfg = get(f"{MAP_URL}/api/config")
    except Exception as exc:
        sys.exit(f"cannot reach the map at {MAP_URL}: {exc}")

    meta = gj.get("metadata", {})
    origin = (meta["origin"]["lat"], meta["origin"]["lon"])
    cells = [f["properties"] for f in gj["features"]]
    live = [c for c in cells if c.get("live_s") and not c.get("masked")]
    print(f"origin        {origin[0]:.5f}, {origin[1]:.5f}")
    print(f"cells         {len(cells)}  ({len(live)} with a travel time)")
    print(f"walk source   {cfg['provider'].get('walk_source')}")
    rt = cfg["provider"].get("stat_realtime", {})
    print(f"realtime      {rt.get('trips_with_delay')} trips, "
          f"{rt.get('skipped_stops')} skipped stops, "
          f"median delay {rt.get('median_delay_s')}s")

    # -- structural self-checks: cheap, and catch whole classes of error -----
    print("\n--- internal consistency ---")
    bad_walk = [c for c in live if c["baseline_s"] and c["live_s"] > c["baseline_s"] + 1]
    print(f"{'cells slower than walking':38} {len(bad_walk)}  (want 0)")
    far = sorted(live, key=lambda c: c["live_s"])[-1]
    near = sorted(live, key=lambda c: c["live_s"])[0]
    print(f"{'fastest / slowest cell':38} {near['live_s']/60:.1f} / {far['live_s']/60:.1f} min")
    jumps = [c for c in live if c.get("next_jump_s") is not None]
    print(f"{'cells with a cliff in the window':38} {len(jumps)} of {len(live)}")

    # -- cross-validation against OTP ---------------------------------------
    print(f"\n--- RAPTOR vs OTP, {N_SAMPLES} random cells ---")
    when = datetime.now(TZ)
    random.seed(0)
    sample = random.sample(live, min(N_SAMPLES, len(live)))
    rows, fails = [], 0
    for c in sample:
        try:
            otp = otp_plan(origin, (c["lat"], c["lon"]), when)
        except Exception:
            fails += 1
            continue
        if otp is None:
            continue
        rows.append((c["id"], c["live_s"], otp))

    if not rows:
        sys.exit("no comparable cells — is OTP running on 8080?")

    diffs = [(r - o) / o * 100 for _, r, o in rows]
    print(f"{'cell':>10} {'raptor':>9} {'otp':>9} {'diff':>8}")
    for cid, r, o in sorted(rows, key=lambda x: (x[1] - x[2]) / x[2])[:6]:
        print(f"{cid:>10} {r/60:8.1f}m {o/60:8.1f}m {(r-o)/o*100:+7.1f}%")
    print(f"{'...':>10}")
    for cid, r, o in sorted(rows, key=lambda x: (x[1] - x[2]) / x[2])[-4:]:
        print(f"{cid:>10} {r/60:8.1f}m {o/60:8.1f}m {(r-o)/o*100:+7.1f}%")

    print(f"\ncompared      {len(rows)} cells ({fails} query failures)")
    print(f"median diff   {statistics.median(diffs):+.1f}%")
    print(f"mean diff     {statistics.mean(diffs):+.1f}%")
    print(f"within 20%    {sum(1 for d in diffs if abs(d) <= 20)}/{len(diffs)}")
    worse = sum(1 for d in diffs if d > 10)
    print(f"raptor slower by >10%: {worse}  (expect ~0; optimistic walking "
          f"should never lose to OTP)")


if __name__ == "__main__":
    main()
