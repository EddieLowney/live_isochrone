"""Cycle orchestration.

One cycle is: ask the pacer what it can afford, ask the sampler what to buy with
it, spend through the ledger, measure, reconstruct, publish.

The ordering matters in one specific way — the ledger is decremented *before*
the provider call, and refunded only on a failure the provider would not have
billed. Any other ordering makes the quota advisory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .config import Components
from .masks import MaskDecision
from .reconstruct import Reconstructor
from .sampling import SamplingContext
from .types import (
    BudgetExhausted,
    CellState,
    Measurement,
    ProviderError,
    Surface,
    utcnow,
)


@dataclass
class CycleReport:
    cycle: int
    started_at: datetime
    allowance: int
    elements_spent: float = 0.0
    cells_requested: int = 0
    cells_measured: int = 0
    batches_sent: int = 0
    batches_refused: int = 0
    errors: list[str] = field(default_factory=list)
    holdout_mae_s: Optional[float] = None
    budget_remaining: float = 0.0
    refit: bool = False
    baseline_updated: int = 0
    cells_masked: int = 0
    cells_failed: int = 0
    budget_stalled: bool = False
    archived: bool = True

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["started_at"] = self.started_at.isoformat()
        if self.holdout_mae_s is not None:
            d["holdout_mae_s"] = round(self.holdout_mae_s, 2)
        return d


class Engine:
    def __init__(self, components: Components):
        self.c = components
        self.cfg = components.config
        eng = self.cfg.section("engine")
        self.refit_every = int(eng["refit_every"])
        self.history_cycles = int(eng["history_cycles"])
        self.validation_holdout = int(eng["validation_holdout"])
        self.max_batches = int(eng["max_batches_per_cycle"])
        # When the provider returns free-flow alongside the live value, the
        # baseline can be maintained continuously instead of sampled once.
        # EWMA rather than last-value: free-flow estimates carry noise.
        self.baseline_from_measurements = bool(eng.get("baseline_from_measurements", False))
        self.baseline_ewma_alpha = float(eng.get("baseline_ewma_alpha", 0.05))
        # Display cadence and archive cadence are different questions. At 30 s
        # cycles a row per cell per cycle is millions a day for history nobody
        # needs at that resolution.
        self.archive_every = max(1, int(eng.get("archive_every", 1)))
        self._last_fit_cycle = -10**9
        self._check_grid_fingerprint()
        self.mask = components.mask
        self.mask.prepare(components.grid.cells, components.store.get_state("mask"))
        self._decision: MaskDecision = self.mask.evaluate(components.grid.cells)

    # -- grid identity -----------------------------------------------------

    def grid_fingerprint(self) -> str:
        """Hash of the grid's geometry.

        Cell ids are axial coordinates, so they stay identical when the origin
        or cell size changes. Without this check a stored baseline would be
        silently reused for cells that now sit somewhere else entirely, and
        every delta would be wrong with nothing to show for it.
        """
        spec = {
            "origin": [round(self.cfg.origin.lat, 7), round(self.cfg.origin.lon, 7)],
            "grid": self.c.grid.describe(),
        }
        blob = json.dumps(spec, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _check_grid_fingerprint(self) -> None:
        store = self.c.store
        current = self.grid_fingerprint()
        saved = (store.get_state("grid") or {}).get("fingerprint")
        if saved and saved != current and store.get_baseline():
            raise RuntimeError(
                "the grid geometry changed but a baseline from the old grid is "
                f"still stored (was {saved}, now {current}). Those cells are in "
                "different places now, so every delta would be wrong.\n"
                "Re-seed with:  isogrid.cli ... seed --reset"
            )
        store.put_state("grid", {"fingerprint": current})
        store.register_grid(
            current, self.cfg.origin, self.c.grid.cells, self.c.grid.describe()
        )

    def reset_measurements(self, keep_archive: bool = True) -> None:
        """Drop the baseline for a changed grid. The measurement archive is kept
        by default — the roads were doing what they were doing regardless of how
        the area happened to be tiled."""
        self.c.store.reset(keep_archive=keep_archive)
        self.c.store.put_state("grid", {"fingerprint": self.grid_fingerprint()})

    # -- baseline ---------------------------------------------------------

    def seed_baseline_from_live(self, when: Optional[datetime] = None) -> int:
        """Populate the baseline from one live sweep, for providers that return
        free-flow alongside the live value. Costs one live sweep instead of a
        separate historical one, and the two numbers agree by construction."""
        provider, ledger = self.c.provider, self.c.ledger
        derived: dict[str, float] = {}
        live_cells = [c for c in self.c.grid.cells if not self._decision.is_excluded(c.id)]
        for batch in provider.plan(live_cells):
            if not ledger.try_spend(provider.budget_key, batch.elements, when):
                raise BudgetExhausted("not enough allowance to seed the baseline")
            try:
                for m in provider.fetch(self.cfg.origin, batch, provider.live_profile, when):
                    if m.free_flow_seconds is not None:
                        derived[m.cell_id] = m.free_flow_seconds
            except ProviderError:
                ledger.refund(provider.budget_key, batch.elements, when)
                raise
        if not derived:
            raise RuntimeError(
                f"provider '{provider.budget_key}' returned no free-flow values; "
                "use bootstrap_baseline() instead"
            )
        self.c.store.set_baseline(derived, provider.live_profile)
        return len(derived)

    def bootstrap_baseline(self, force: bool = False, when: Optional[datetime] = None) -> int:
        """Measure every cell once on the baseline profile.

        Uses the same provider as the live layer by design: a baseline from a
        different engine would inject a spatial bias into every delta that never
        cancels out.
        """
        existing = self.c.store.get_baseline()
        todo = [
            c
            for c in self.c.grid.cells
            if (force or c.id not in existing) and not self._decision.is_excluded(c.id)
        ]
        if not todo:
            return 0
        provider, ledger = self.c.provider, self.c.ledger
        measured: dict[str, float] = {}
        for batch in provider.plan(todo):
            if not ledger.try_spend(provider.budget_key, batch.elements, when):
                raise BudgetExhausted(
                    f"baseline needs {batch.elements} more units of "
                    f"'{provider.budget_key}' but the period allowance is spent"
                )
            try:
                for m in provider.fetch(
                    self.c.config.origin, batch, provider.baseline_profile, when
                ):
                    measured[m.cell_id] = m.seconds
            except ProviderError:
                ledger.refund(provider.budget_key, batch.elements, when)
                raise
        self.c.store.set_baseline(measured, provider.baseline_profile)
        return len(measured)

    # -- one cycle --------------------------------------------------------

    def run_cycle(self, when: Optional[datetime] = None) -> tuple[Surface, CycleReport]:
        now = when or utcnow()
        store, grid, provider, ledger = self.c.store, self.c.grid, self.c.provider, self.c.ledger

        cycle = store.next_cycle()
        baseline = store.get_baseline()
        if not baseline:
            raise RuntimeError("no baseline recorded — run bootstrap_baseline first")

        allowance = self.c.pacer.allowance(
            ledger, provider.budget_key, now, self.cfg.cycle_seconds
        )
        report = CycleReport(cycle=cycle, started_at=now, allowance=allowance)

        if hasattr(self.mask, "tick"):
            self.mask.tick(cycle)
        self._decision = self.mask.evaluate(grid.cells)
        report.cells_masked = len(self._decision.excluded)

        history = store.history(self.history_cycles)
        ctx = SamplingContext(
            grid=grid,
            baseline=baseline,
            latest=store.latest_observations(),
            history=history,
            cycle=cycle,
            now=now,
            excluded=set(self._decision.excluded),
        )

        if allowance <= 0:
            # Distinct from a refused batch: the pacer never even proposed a
            # request. Without this the map simply goes quiet with no reason
            # attached anywhere.
            report.budget_stalled = True
            report.errors.append(
                f"allowance is 0 — '{provider.budget_key}' quota exhausted for "
                f"this {ledger.period} period"
            )

        wanted = self.c.sampler.select(ctx, allowance)
        report.cells_requested = len(wanted)

        measurements: list[Measurement] = []
        for batch in provider.plan(wanted)[: self.max_batches]:
            if not ledger.try_spend(provider.budget_key, batch.elements, now):
                report.batches_refused += 1
                break
            try:
                got = provider.fetch(self.cfg.origin, batch, provider.live_profile, now)
                measurements.extend(got)
                report.batches_sent += 1
                report.elements_spent += batch.elements
                # Requested but not returned means unroutable upstream. Feeding
                # that back is what lets the quarantine mask learn.
                returned = {m.cell_id for m in got}
                missing = [c.id for c in batch.cells if c.id not in returned]
                # OTP and the transit simulator name their own unreachable
                # cells; trust that over inference when it is available.
                named = getattr(provider, "last_unreachable", None)
                if named:
                    missing = [c for c in named if c not in returned]
                self.mask.observe([c.id for c in batch.cells], returned)
                if missing:
                    store.record_failures(missing, cycle, provider.budget_key, when=now)
                report.cells_failed += len(missing)
            except ProviderError as exc:
                ledger.refund(provider.budget_key, batch.elements, now)
                report.errors.append(str(exc))

        report.cells_measured = len(measurements)
        report.archived = cycle % self.archive_every == 0
        if report.archived:
            store.record(measurements, cycle)

        if self.baseline_from_measurements:
            derived = {
                m.cell_id: m.free_flow_seconds
                for m in measurements
                if m.free_flow_seconds is not None
            }
            if derived:
                a = self.baseline_ewma_alpha
                blended = {
                    cid: (a * v + (1 - a) * baseline[cid]) if cid in baseline else v
                    for cid, v in derived.items()
                }
                store.set_baseline(blended, provider.baseline_profile)
                baseline.update(blended)
                report.baseline_updated = len(blended)

        if cycle - self._last_fit_cycle >= self.refit_every:
            self.c.reconstructor.fit(
                store.history(self.history_cycles), baseline, grid.cells
            )
            self._last_fit_cycle = cycle
            report.refit = True

        surface = self._reconstruct(
            measurements, baseline, now, report, cycle
        )
        store.put_state("mask", self.mask.state())
        report.budget_remaining = ledger.remaining(provider.budget_key, now)
        store.record_cycle(report.as_dict())
        surface.notes["cycle_report"] = report.as_dict()
        surface.notes["budget"] = ledger.snapshot(now)
        surface.notes["reconstructor"] = self.c.reconstructor.diagnostics()
        surface.notes["mask"] = self.mask.describe()
        store.put_surface(surface.as_dict())
        return surface, report

    def _reconstruct(
        self,
        measurements: list[Measurement],
        baseline: dict[str, float],
        now: datetime,
        report: CycleReport,
        cycle: int,
    ) -> Surface:
        observed_ratio: dict[str, float] = {}
        observed_secs: dict[str, float] = {}
        # Transit measurements carry a departure-window profile. Keep it keyed
        # by cell so the surface can report spread alongside the median.
        detail: dict[str, Measurement] = {m.cell_id: m for m in measurements}
        for m in measurements:
            b = baseline.get(m.cell_id)
            if b is None:
                continue
            observed_secs[m.cell_id] = m.seconds
            # A zero baseline means the origin cell; ratio is defined as 1.
            observed_ratio[m.cell_id] = (m.seconds / b) if b else 1.0

        # Hold out a few measured cells so reconstruction quality is measured
        # against ground truth the model did not see. Works against any
        # provider, not just the synthetic one.
        holdout: dict[str, float] = {}
        if self.validation_holdout > 0 and len(observed_ratio) > self.validation_holdout + 2:
            keys = sorted(observed_ratio)
            step = max(1, len(keys) // self.validation_holdout)
            for cid in keys[::step][: self.validation_holdout]:
                holdout[cid] = observed_ratio.pop(cid)

        estimates = self.c.reconstructor.estimate(
            self.c.grid.cells, observed_ratio, baseline
        )
        self.c.store.record_estimates(
            cycle,
            {
                k: (baseline[k] * v)
                for k, v in estimates.items()
                if baseline.get(k) is not None
            },
            observed_secs.keys(),
            when=now,
        )

        if holdout:
            errs = [
                abs(estimates.get(cid, 1.0) - true) * baseline.get(cid, 0.0)
                for cid, true in holdout.items()
            ]
            report.holdout_mae_s = sum(errs) / len(errs)
            observed_ratio.update(holdout)

        latest = self.c.store.latest_observations()
        cells = []
        for cell in self.c.grid.cells:
            if self._decision.is_excluded(cell.id):
                # Rendered as a hole, not as an estimate. Pretending to know a
                # travel time for open water would be worse than a gap.
                cells.append(
                    CellState(
                        cell_id=cell.id,
                        center=cell.center,
                        masked=True,
                        mask_reason=self._decision.reason(cell.id),
                    )
                )
                continue
            base = baseline.get(cell.id)
            last = latest.get(cell.id)
            est_ratio = estimates.get(cell.id, 1.0)
            d = detail.get(cell.id) or last
            cells.append(
                CellState(
                    cell_id=cell.id,
                    center=cell.center,
                    baseline_s=base,
                    observed_s=observed_secs.get(cell.id),
                    observed_at=last.observed_at if last else None,
                    estimated_s=(base * est_ratio) if base is not None else None,
                    is_direct=cell.id in observed_secs,
                    p10_s=getattr(d, "p10_seconds", None) if d else None,
                    p90_s=getattr(d, "p90_seconds", None) if d else None,
                    walk_s=getattr(d, "walk_seconds", None) if d else None,
                    wait_s=getattr(d, "wait_seconds", None) if d else None,
                    transfers=getattr(d, "transfers", None) if d else None,
                    next_jump_s=getattr(d, "next_jump_s", None) if d else None,
                )
            )

        return Surface(
            origin=self.cfg.origin,
            generated_at=now,
            cells=cells,
            directly_measured=len(observed_secs),
            elements_spent=int(report.elements_spent),
        )

    # -- convenience ------------------------------------------------------

    def simulate(
        self,
        cycles: int,
        start: Optional[datetime] = None,
        step_seconds: Optional[float] = None,
    ) -> list[CycleReport]:
        """Fast-forward many cycles in virtual time.

        The only way to evaluate a sampling or reconstruction change without
        waiting days for real data, and with the synthetic provider it costs
        nothing.
        """
        step = timedelta(seconds=step_seconds or self.cfg.cycle_seconds)
        t = start or utcnow()
        reports = []
        for _ in range(cycles):
            _surface, rep = self.run_cycle(t)
            reports.append(rep)
            t += step
        return reports

    def geojson(self) -> dict:
        """Cell polygons as GeoJSON, joined to the latest surface. Consumed by
        the renderer; also drops straight into deck.gl or QGIS."""
        surface = self.c.store.get_surface() or {}
        by_id = {c["id"]: c for c in surface.get("cells", [])}
        features = []
        for cell in self.c.grid.cells:
            props = dict(by_id.get(cell.id, {"id": cell.id}))
            features.append(
                {
                    "type": "Feature",
                    "id": cell.id,
                    "properties": props,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[p.lon, p.lat] for p in cell.polygon]
                            + [[cell.polygon[0].lon, cell.polygon[0].lat]]
                        ],
                    },
                }
            )
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {k: v for k, v in surface.items() if k != "cells"},
        }
