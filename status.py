"""Where is everything? Run before starting or after any folder surgery.

Reads state only — costs no transactions.
"""
import os
from isogrid import Config

cfg = Config.load(os.environ.get("ISOGRID_CONFIG", "config/tomtom.yaml"))
c = cfg.build()
base, snap = c.store.get_baseline(), c.ledger.snapshot()
key = os.environ.get(cfg.section("render").get("basemap", {}).get("key_env", "TOMTOM_API_KEY"))

print(f"config          {os.environ.get('ISOGRID_CONFIG', 'config/tomtom.yaml')}")
print(f"api key         {'set' if key else 'NOT SET — export TOMTOM_API_KEY'}")
print(f"baseline cells  {len(base)} of {len(c.grid)}")
print(f"cycles run      {c.store.next_cycle() - 1}")
print(f"period          {snap['period_key']}  resets {snap['period_ends'][:16]}Z")
for k, v in snap["providers"].items():
    if v["spent"] or k.startswith("tomtom"):
        pct = 100 * v["spent"] / max(1, v["effective_quota"])
        print(f"  {k:14} {v['spent']:>7.0f} / {v['effective_quota']:<7} ({pct:.1f}%)")

if not key:
    print("\nACTION  export TOMTOM_API_KEY, then re-run this")
elif len(base) < len(c.grid) * 0.5:
    print("\nACTION  no baseline — run `seed` (37 transactions), then `serve`")
else:
    print("\nACTION  ready — run `serve`")
