"""What does MBTA's transfers.txt actually contain?"""
import csv, io, zipfile, collections
with zipfile.ZipFile("otp/mbta.gtfs.zip") as zf:
    names = [n for n in zf.namelist() if n.endswith(".txt")]
    if "transfers.txt" not in names:
        raise SystemExit("no transfers.txt in this feed")
    rows = list(csv.DictReader(io.StringIO(
        zf.read("transfers.txt").decode("utf-8-sig"))))
print(f"rows      {len(rows):,}")
print(f"columns   {list(rows[0].keys())}\n")
types = collections.Counter(r.get("transfer_type", "") for r in rows)
print("transfer_type  (0=recommended 1=timed 2=needs min_time 3=not possible)")
for k, v in sorted(types.items()):
    print(f"  {k or '(blank)':<10} {v:>7,}")
mt = [int(r["min_transfer_time"]) for r in rows
      if r.get("min_transfer_time", "").strip().isdigit()]
print(f"\nrows with min_transfer_time: {len(mt):,}")
if mt:
    mt.sort()
    print(f"  min {mt[0]}s  median {mt[len(mt)//2]}s  p90 {mt[int(len(mt)*.9)]}s  max {mt[-1]}s")
extra = [c for c in rows[0] if c not in
         ("from_stop_id","to_stop_id","transfer_type","min_transfer_time")]
print(f"\nnon-standard columns: {extra or 'none'}")
print("\nsample rows:")
for r in rows[:4]:
    print("  ", {k: v for k, v in r.items() if v})
