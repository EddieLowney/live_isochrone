"""Command line entry points.

    python -m isogrid.cli baseline    # measure the free-flow reference once
    python -m isogrid.cli run         # one cycle
    python -m isogrid.cli simulate    # many cycles in virtual time
    python -m isogrid.cli bench       # compare reconstructors on held-out cells
    python -m isogrid.cli serve       # HTTP API plus the renderer
    python -m isogrid.cli budget      # what the ledger thinks you have left
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from .config import Config
from .engine import Engine
from .types import BudgetExhausted


def _load(args) -> Config:
    overrides: dict = {}
    if args.provider:
        overrides["provider"] = {"kind": args.provider}
    if args.ephemeral:
        overrides["store"] = {"kind": "memory"}
        overrides["budget"] = {"ledger": {"kind": "memory"}}
    cfg = Config.load(args.config, overrides=overrides or None)
    for spec in args.set or []:
        path, _, value = spec.partition("=")
        node = cfg.raw
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            node[keys[-1]] = json.loads(value)
        except json.JSONDecodeError:
            node[keys[-1]] = value
    for section in ("store", "budget"):
        block = cfg.raw.get(section, {})
        p = block.get("path") or block.get("ledger", {}).get("path")
        if p:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return cfg


def cmd_baseline(args) -> None:
    engine = Engine(_load(args).build())
    n = engine.bootstrap_baseline(force=args.force)
    print(f"baseline: {n} cells measured on the '{engine.c.provider.baseline_profile}' profile")


def cmd_run(args) -> None:
    """Run cycles in real time, sleeping between them.

    The sleep is not cosmetic. Without it, `--cycles 10` would fire ten cycles
    back to back and spend ten cycles' worth of quota in seconds. Use
    `--interval 0` only with the synthetic provider.
    """
    import time

    cfg = _load(args)
    engine = Engine(cfg.build())
    if not engine.c.store.get_baseline():
        raise SystemExit(
            "no baseline yet. Run `seed` (providers returning free-flow) or "
            "`baseline` (everyone else) first."
        )
    interval = cfg.cycle_seconds if args.interval is None else args.interval
    for i in range(args.cycles):
        if i:
            time.sleep(interval)
        _s, report = engine.run_cycle()
        print(json.dumps(report.as_dict()), flush=True)


def cmd_simulate(args) -> None:
    cfg = _load(args)
    engine = Engine(cfg.build())
    start = datetime.now(timezone.utc).replace(hour=4, minute=0, second=0, microsecond=0)
    if engine.c.store.get_baseline() == {}:
        engine.bootstrap_baseline(when=start)
    reports = engine.simulate(args.cycles, start=start)
    warm = [r for r in reports[len(reports) // 3 :] if r.holdout_mae_s is not None]
    spent = sum(r.elements_spent for r in reports)
    print(f"cycles              {len(reports)}")
    print(f"elements spent      {spent:.0f}")
    print(f"elements per cycle  {spent / max(1, len(reports)):.1f}")
    if warm:
        mae = sum(r.holdout_mae_s for r in warm) / len(warm)
        print(f"holdout MAE         {mae:.1f} s   ({mae / 60:.2f} min)")
    print(f"reconstructor       {json.dumps(engine.c.reconstructor.diagnostics())}")
    print(f"budget              {json.dumps(engine.c.ledger.snapshot())}")


def cmd_bench(args) -> None:
    """Compare reconstruction strategies under an identical measurement stream."""
    rows = []
    for kind in args.reconstructors.split(","):
        cfg = _load(args)
        cfg.raw["store"] = {"kind": "memory"}
        cfg.raw["budget"]["ledger"] = {"kind": "memory"}
        cfg.raw["reconstruct"]["kind"] = kind.strip()
        engine = Engine(cfg.build())
        start = datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)
        engine.bootstrap_baseline(when=start)
        reports = engine.simulate(args.cycles, start=start)
        warm = [r.holdout_mae_s for r in reports[len(reports) // 3 :] if r.holdout_mae_s is not None]
        rows.append((kind.strip(), sum(warm) / len(warm) if warm else float("nan")))
    width = max(len(r[0]) for r in rows)
    best = min(r[1] for r in rows)
    print(f"{'reconstructor'.ljust(width)}   holdout MAE")
    for name, mae in rows:
        flag = "  <-- best" if abs(mae - best) < 1e-9 else ""
        print(f"{name.ljust(width)}   {mae:8.1f} s{flag}")


def cmd_seed(args) -> None:
    """Seed the baseline from one live sweep, for providers that return
    free-flow with every measurement."""
    cfg = _load(args)
    if args.reset:
        # Build the components directly: Engine refuses to construct when the
        # stored grid no longer matches, which is exactly when --reset is needed.
        comps = cfg.build()
        comps.store.reset(keep_archive=not args.wipe_archive)
        print("cleared baseline" + ("" if args.wipe_archive else "; measurement archive kept"))
    engine = Engine(cfg.build())
    try:
        n = engine.seed_baseline_from_live()
        print(f"baseline seeded from {n} live cells (free-flow returned by provider)")
    except RuntimeError as exc:
        if "no free-flow values" not in str(exc):
            raise
        # Only TomTom returns free-flow alongside the live value. Everyone else
        # needs a separate sweep on the baseline profile — which is what the
        # caller wanted anyway, so do it rather than making them read an error.
        prof = engine.c.provider.baseline_profile
        print(f"provider does not return free-flow; sweeping the '{prof}' profile instead")
        n = engine.bootstrap_baseline()
        print(f"baseline: {n} cells measured on the '{prof}' profile")


def cmd_preflight(args) -> None:
    """Cheapest possible end-to-end check before committing a real run.

    Costs two transactions. Verifies auth, response shape, free-flow
    availability, and whether the configured cadence fits the quota.
    """
    cfg = _load(args)
    c = cfg.build()
    prov, grid, led = c.provider, c.grid, c.ledger
    key = prov.budget_key
    ok = True

    print(f"provider        {prov.describe()}")
    print(f"grid            {len(grid)} cells")

    per_sweep = sum(b.elements for b in prov.plan(grid.cells))
    naive = prov.cost_of(len(grid)) if hasattr(prov, "cost_of") else per_sweep
    cycles_per_day = 86400 / cfg.cycle_seconds
    quota = led.effective_quota(key)
    print(f"cost per sweep  {per_sweep} units  ({len(prov.plan(grid.cells))} requests)")
    if naive > per_sweep:
        print(f"                unbatched would cost {naive} — batching saves {naive - per_sweep}")
    print(f"cycles per day  {cycles_per_day:.0f} at {cfg.cycle_seconds:.0f}s")
    print(f"usable quota    {quota} per period ({led.period}) ({led.quotas.get(key)} x {led.safety})")

    allowance = c.pacer.allowance(led, key, datetime.now(timezone.utc), cfg.cycle_seconds)
    daily = allowance * cycles_per_day
    verdict = "fits" if daily <= quota else "OVER BUDGET"
    if daily > quota:
        ok = False
    print(f"planned spend   {allowance}/cycle -> {daily:.0f}/day  [{verdict}]")
    if daily > quota:
        print(f"                reduce pacer units to {int(quota / cycles_per_day)} "
              f"or raise cycle_seconds")

    # Show exactly what will go on the wire, so a 400 can be reproduced with
    # curl without guessing.
    probe = list(grid.cells)[1:3]
    batch = prov.plan(probe)[0]
    if hasattr(prov, "_options"):
        body = {
            "origins": [{"point": {"latitude": cfg.origin.lat,
                                   "longitude": cfg.origin.lon}}],
            "destinations": [
                {"point": {"latitude": c.center.lat, "longitude": c.center.lon}}
                for c in batch.cells
            ],
            "options": prov._options(prov.live_profile, None),
        }
        print("\nrequest body")
        print(json.dumps(body, indent=2))
    if args.dry_run:
        print("\ndry run — nothing sent")
        return

    print("\nlive request (2 cells, ~2 transactions)...")
    # Spend through the ledger so the printed budget is honest.
    if not led.try_spend(key, batch.elements):
        print("  FAILED  ledger refused the spend; quota already used")
        raise SystemExit(1)
    try:
        got = prov.fetch(cfg.origin, batch, prov.live_profile, None)
    except Exception as exc:  # noqa: BLE001
        led.refund(key, batch.elements)
        print(f"  FAILED  {type(exc).__name__}: {exc}")
        print("\n  403 -> key not entitled to Matrix Routing v2, or quota gone")
        print("  400 -> request rejected; the body above is what was sent")
        print("  no key -> export TOMTOM_API_KEY=... first")
        raise SystemExit(1)
    if not got:
        print("  FAILED  no measurements returned — check grid coverage for this region")
        raise SystemExit(1)
    for m in got:
        ff = f", free-flow {m.free_flow_seconds:.0f}s" if m.free_flow_seconds else ""
        print(f"  {m.cell_id}  {m.seconds:.0f}s{ff}")
    has_ff = all(m.free_flow_seconds is not None for m in got)
    print(f"\nfree-flow returned   {has_ff}")
    if has_ff:
        print("  -> use `seed`, and keep engine.baseline_from_measurements: true")
    else:
        print("  -> use `baseline`, which costs one extra sweep")
    print(f"budget after preflight  {led.snapshot()}")
    print("\nPREFLIGHT OK" if ok else "\nPREFLIGHT: fix the budget warning above")


def cmd_mask(args) -> None:
    """Show which cells are excluded and why, before spending anything."""
    c = _load(args).build()
    c.mask.prepare(c.grid.cells, c.store.get_state("mask"))
    decision = c.mask.evaluate(c.grid.cells)
    total, dropped = len(c.grid), len(decision.excluded)
    print(f"mask        {json.dumps(c.mask.describe())}")
    print(f"grid        {total} cells")
    print(f"excluded    {dropped} ({dropped / max(1, total):.0%})")
    print(f"active      {total - dropped}")
    by_reason: dict = {}
    for cid, reason in decision.excluded.items():
        by_reason.setdefault(reason, []).append(cid)
    for reason, ids in sorted(by_reason.items()):
        print(f"\n  {reason} ({len(ids)})")
        for cid in sorted(ids)[: args.limit]:
            cell = c.grid.get(cid)
            print(f"    {cid:14} {cell.center.lat:.5f},{cell.center.lon:.5f}")
        if len(ids) > args.limit:
            print(f"    ... {len(ids) - args.limit} more")
    if dropped:
        per_sweep = sum(b.elements for b in c.provider.plan(c.grid.cells))
        saved = per_sweep - sum(
            b.elements
            for b in c.provider.plan(
                [x for x in c.grid.cells if not decision.is_excluded(x.id)]
            )
        )
        print(f"\nsaves ~{saved} units per full sweep")


def cmd_export(args) -> None:
    """Dump the measurement archive as a training-ready table.

    Emits the `observation_features` view: raw seconds plus free-flow, distance,
    derived delta / ratio / speed, hour and weekday, and the cell's coordinates.
    """
    import csv
    from datetime import timedelta

    c = _load(args).build()
    since = None
    if args.since_days:
        since = (datetime.now(timezone.utc) - timedelta(days=args.since_days)).timestamp()
    cols, rows = c.store.export_rows(since)
    if not rows:
        print("nothing to export yet")
        return
    if args.format == "parquet":
        try:
            import pandas as pd
        except ImportError:
            raise SystemExit("parquet needs pandas: pip install pandas pyarrow")
        pd.DataFrame(rows, columns=cols).to_parquet(args.out, index=False)
    else:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows)
    print(f"exported {len(rows)} rows x {len(cols)} columns -> {args.out}")


def cmd_archive(args) -> None:
    """What has been collected so far."""
    c = _load(args).build()
    for k, v in c.store.archive_stats().items():
        print(f"  {k:22} {v}")


def cmd_budget(args) -> None:
    engine = Engine(_load(args).build())
    print(json.dumps(engine.c.ledger.snapshot(), indent=2))


def cmd_serve(args) -> None:
    from .server import serve

    engine = Engine(_load(args).build())
    if not engine.c.store.get_baseline():
        print("no baseline yet — measuring it now")
        engine.bootstrap_baseline()
    try:
        engine.run_cycle()
    except BudgetExhausted as exc:
        print(f"warning: {exc}")
    serve(engine, host=args.host, port=args.port, autorun=not args.no_autorun)


def cmd_publish(args) -> None:
    """Run the pipeline as a writer of static files instead of a server."""
    from .publish import MBTA_GTFS_URL, DirSink, FeedRefresher, R2Sink, run

    cfg = _load(args)
    sinks = []
    if args.out:
        sinks.append(DirSink(args.out))
    if args.r2:
        # Before the engine exists: missing credentials should fail in one
        # second, not after a baseline.
        sinks.append(R2Sink.from_env(gzip_bodies=not args.no_gzip))
    if not sinks:
        raise SystemExit("nothing to publish to: pass --out DIR, --r2, or both")
    feed = None
    gtfs = (cfg.raw.get("provider") or {}).get("gtfs_path")
    if gtfs and not args.no_feed_refresh:
        feed = FeedRefresher(gtfs, url=args.feed_url or MBTA_GTFS_URL,
                             every_s=args.feed_check_hours * 3600)
    engine = Engine(cfg.build())
    # Feed refresh, baseline and first cycle happen inside `run`, in that order.
    pub = run(engine, sinks, vehicles_s=args.vehicles_seconds, feed=feed, once=args.once)
    if args.once:
        print(json.dumps(pub.status_payload(), indent=2))


def cmd_publish_check(args) -> None:
    from .publish import check

    raise SystemExit(0 if check(args.url, args.origin) else 1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="isogrid", description=__doc__)
    p.add_argument("--config", default="config/prototype.yaml")
    p.add_argument("--provider", help="override provider kind (synthetic|mapbox)")
    p.add_argument("--ephemeral", action="store_true", help="memory store and ledger")
    p.add_argument("--set", action="append", metavar="a.b=value", help="dotted config override")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("baseline")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_baseline)
    s = sub.add_parser("run")
    s.add_argument("--cycles", type=int, default=1)
    s.add_argument("--interval", type=float, default=None,
                   help="seconds between cycles (default: engine.cycle_seconds)")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("simulate"); s.add_argument("--cycles", type=int, default=300); s.set_defaults(fn=cmd_simulate)
    s = sub.add_parser("bench")
    s.add_argument("--cycles", type=int, default=300)
    s.add_argument("--reconstructors", default="passthrough,idw,factor")
    s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("seed")
    s.add_argument("--reset", action="store_true",
                   help="clear the baseline first (use after changing the grid)")
    s.add_argument("--wipe-archive", action="store_true",
                   help="also delete collected measurements — usually not what you want")
    s.set_defaults(fn=cmd_seed)
    s = sub.add_parser("mask")
    s.add_argument("--limit", type=int, default=12)
    s.set_defaults(fn=cmd_mask)
    s = sub.add_parser("preflight")
    s.add_argument("--dry-run", action="store_true",
                   help="print the request body without sending it")
    s.set_defaults(fn=cmd_preflight)
    s = sub.add_parser("export")
    s.add_argument("--out", default="observations.csv")
    s.add_argument("--format", choices=["csv", "parquet"], default="csv")
    s.add_argument("--since-days", type=float, default=None)
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("archive"); s.set_defaults(fn=cmd_archive)
    s = sub.add_parser("budget"); s.set_defaults(fn=cmd_budget)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-autorun", action="store_true")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("publish", help="write the map to static files every cycle")
    s.add_argument("--out", metavar="DIR", help="write to a local directory")
    s.add_argument("--r2", action="store_true",
                   help="upload to Cloudflare R2 (credentials from R2_* env vars)")
    s.add_argument("--no-gzip", action="store_true", help="upload R2 bodies uncompressed")
    s.add_argument("--vehicles-seconds", type=float, default=10.0)
    s.add_argument("--once", action="store_true",
                   help="publish one cycle, print status, and exit")
    s.add_argument("--feed-url", default=None, help="static GTFS to keep current")
    s.add_argument("--feed-check-hours", type=float, default=6.0)
    s.add_argument("--no-feed-refresh", action="store_true")
    s.set_defaults(fn=cmd_publish)
    s = sub.add_parser("publish-check",
                       help="fetch published files the way a browser would")
    s.add_argument("--url", required=True, help="public base URL of the bucket")
    s.add_argument("--origin", default="https://eddielowney.github.io")
    s.set_defaults(fn=cmd_publish_check)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
