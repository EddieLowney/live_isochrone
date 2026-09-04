"""Why do some TripUpdates yield no stop predictions? Read the raw fields."""
import collections, urllib.request
from datetime import date
from isogrid.transit.gtfs import load_gtfs
from isogrid.transit.realtime import (
    F_ENTITY, F_STOP_TIME_UPDATE, F_TRIP, F_TRIP_ID, F_TRIP_UPDATE,
    parse_message, parse_trip_updates,
)

raw = urllib.request.urlopen(
    "https://cdn.mbta.com/realtime/TripUpdates.pb", timeout=30).read()
d = load_gtfs("otp/mbta.gtfs.zip", on=date.today(), cache_dir="data/raptor")
preds = parse_trip_updates(raw)
known = {str(x) for x in d.trip_ids}

empty = {p.trip_id for p in preds.trips
         if not p.canceled and p.trip_id in known and not p.stops}
print(f"trips with zero parsed stop predictions: {len(empty)}\n")

tu_fields, stu_fields, ev_fields, n_stu = (collections.Counter() for _ in range(3)), None, None, []
tu_fields = collections.Counter(); stu_fields = collections.Counter()
ev_fields = collections.Counter(); n_stu = []
shown = 0
for ent_b in parse_message(raw).get(F_ENTITY, []):
    ent = parse_message(ent_b)
    tu_b = ent.get(F_TRIP_UPDATE, [None])[0]
    if tu_b is None:
        continue
    tu = parse_message(tu_b)
    trip = parse_message(tu.get(F_TRIP, [b""])[0])
    tid_raw = trip.get(F_TRIP_ID, [None])[0]
    if tid_raw is None or tid_raw.decode("utf-8", "replace") not in empty:
        continue
    tu_fields.update(tu.keys())
    stus = tu.get(F_STOP_TIME_UPDATE, [])
    n_stu.append(len(stus))
    for s_b in stus:
        s = parse_message(s_b)
        stu_fields.update(s.keys())
        for f in (2, 3):
            if f in s:
                ev_fields.update(parse_message(s[f][0]).keys())
    if shown < 2 and stus:
        print(f"sample trip {tid_raw.decode()}: {len(stus)} stop_time_updates")
        for s_b in stus[:3]:
            s = parse_message(s_b)
            desc = {k: (v[0] if not isinstance(v[0], bytes) else v[0][:12]) for k, v in s.items()}
            print("   fields:", desc)
        shown += 1

print(f"\nstop_time_update count: min={min(n_stu) if n_stu else 0} "
      f"max={max(n_stu) if n_stu else 0} zero={sum(1 for x in n_stu if x==0)}")
print("TripUpdate field numbers seen:      ", dict(tu_fields))
print("StopTimeUpdate field numbers seen:  ", dict(stu_fields))
print("StopTimeEvent field numbers seen:   ", dict(ev_fields))
print("\nlegend: STU 1=stop_sequence 2=arrival 3=departure 4=stop_id 5=schedule_rel")
print("        EVENT 1=delay 2=time 3=uncertainty")
