"""Which cells to spend this cycle's budget on.

A Sampler sees the grid, everything measured so far, and an allowance in
billable units. It returns an ordered list of cells. Order matters: the engine
truncates to whatever the budget actually permits, so the most valuable cells
must come first.

The interesting strategies allocate non-uniformly. Under a fixed budget,
sampling frequency proportional to a cell's delta volatility minimises total
squared staleness — Neyman allocation, with sigma estimated from the
observation log.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence

from .grid import Grid
from .registry import SAMPLERS
from .types import Cell, Measurement, utcnow


class SamplingContext:
    """Everything a sampler is allowed to look at."""

    def __init__(
        self,
        grid: Grid,
        baseline: dict[str, float],
        latest: dict[str, Measurement],
        history: Sequence[tuple[int, dict[str, float]]],
        cycle: int,
        now: Optional[datetime] = None,
        excluded: Optional[set] = None,
    ):
        self.grid = grid
        self.baseline = baseline
        self.latest = latest
        self.history = history
        self.cycle = cycle
        self.now = now or utcnow()
        self.excluded = excluded or set()

    @property
    def active_cells(self) -> list:
        """Cells worth spending on. Samplers must use this rather than
        `grid.cells`, or masked cells silently consume the allowance."""
        return [c for c in self.grid.cells if c.id not in self.excluded]

    def staleness_cycles(self, cell_id: str) -> float:
        m = self.latest.get(cell_id)
        if m is None:
            return float("inf")
        return max(0.0, (self.now - m.observed_at).total_seconds() / 60.0)

    def volatility(self, cell_id: str, min_obs: int = 3) -> Optional[float]:
        """Standard deviation of the congestion ratio for one cell."""
        base = self.baseline.get(cell_id)
        if not base:
            return None
        vals = [snap[cell_id] / base for _, snap in self.history if cell_id in snap]
        if len(vals) < min_obs:
            return None
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(var)


class Sampler(ABC):
    @abstractmethod
    def select(self, ctx: SamplingContext, allowance: int) -> list[Cell]:
        """Ordered, at most `allowance` cells."""

    def describe(self) -> dict:
        return {"type": getattr(self, "registry_name", type(self).__name__)}


@SAMPLERS.register("round_robin")
class RoundRobinSampler(Sampler):
    """Rotate through every cell in a fixed order. Simple, fair, and the right
    default while bootstrapping — it builds the history the smarter samplers
    need before they can do anything."""

    def __init__(self, **_ignored):
        pass

    def select(self, ctx, allowance):
        cells = ctx.active_cells
        if not cells or allowance <= 0:
            return []
        n = len(cells)
        start = (ctx.cycle * allowance) % n
        return [cells[(start + i) % n] for i in range(min(allowance, n))]


@SAMPLERS.register("staleness")
class StalenessSampler(Sampler):
    """Always measure whatever was measured longest ago. Guarantees a bound on
    worst-case age; ignores the fact that some cells barely move.

    Degenerate but correct when the allowance covers every cell, which is the
    situation with a local router: it simply returns all of them."""

    def __init__(self, **_ignored):
        pass

    def select(self, ctx, allowance):
        if allowance <= 0:
            return []
        ranked = sorted(ctx.active_cells, key=lambda c: -ctx.staleness_cycles(c.id))
        return ranked[:allowance]


@SAMPLERS.register("volatility")
class VolatilitySampler(Sampler):
    """Neyman allocation with a staleness term.

    Priority is sigma_i * age_i: a cell earns measurement by being both
    unpredictable and out of date. Cells without enough history get
    `unknown_priority`, which should stay high so new cells get explored rather
    than starved.
    """

    def __init__(
        self,
        unknown_priority: float = 10.0,
        staleness_weight: float = 1.0,
        min_obs: int = 3,
        floor_sigma: float = 0.005,
        **_ignored,
    ):
        self.unknown_priority = float(unknown_priority)
        self.staleness_weight = float(staleness_weight)
        self.min_obs = int(min_obs)
        self.floor_sigma = float(floor_sigma)

    def _priority(self, ctx: SamplingContext, cell: Cell) -> float:
        sigma = ctx.volatility(cell.id, self.min_obs)
        age = ctx.staleness_cycles(cell.id)
        if sigma is None or math.isinf(age):
            return self.unknown_priority + min(age, 1e6) * 1e-6
        return max(sigma, self.floor_sigma) * (1.0 + self.staleness_weight * age)

    def select(self, ctx, allowance):
        if allowance <= 0:
            return []
        ranked = sorted(ctx.active_cells, key=lambda c: -self._priority(ctx, c))
        return ranked[:allowance]


@SAMPLERS.register("probe")
class ProbeSampler(Sampler):
    """Fixed high-information probe set, refreshed every cycle, with leftover
    budget rotating through everything else.

    This is the shape that fits a small quota: the probes identify the factor
    structure the reconstructor uses to estimate unmeasured cells, and the
    rotation slowly refreshes the rest so the model can be validated against
    ground truth it did not train on.
    """

    def __init__(
        self,
        probe_ids: Optional[Sequence[str]] = None,
        n_probes: int = 12,
        selection: str = "volatility",
        refresh_every: int = 288,
        max_probe_fraction: float = 0.7,
        **_ignored,
    ):
        self.probe_ids = list(probe_ids) if probe_ids else []
        self.n_probes = int(n_probes)
        self.selection = selection
        self.refresh_every = int(refresh_every)
        #: Ceiling on the share of each cycle's allowance the fixed probes may
        #: take. Without it, a probe set as large as the allowance starves the
        #: rotation, no non-probe cell is ever measured, and the reconstructor
        #: has no history for the cells it is supposed to predict.
        self.max_probe_fraction = float(max_probe_fraction)
        self._chosen: list[str] = []
        self._chosen_at = -1

    def _choose_probes(self, ctx: SamplingContext) -> list[str]:
        if self.probe_ids:
            return [c for c in self.probe_ids if ctx.grid.has(c)]
        scored: list[tuple[float, str]] = []
        for cell in ctx.active_cells:
            sigma = ctx.volatility(cell.id, min_obs=3)
            scored.append((sigma if sigma is not None else 0.0, cell.id))
        scored.sort(reverse=True)
        picked = [cid for _, cid in scored[: self.n_probes]]
        if len(picked) < self.n_probes:
            # Not enough history yet: spread the remainder over rings so the
            # probe set has geometric coverage from the first cycle.
            spare = [c.id for c in ctx.active_cells if c.id not in picked]
            stride = max(1, len(spare) // max(1, self.n_probes - len(picked)))
            picked += spare[::stride][: self.n_probes - len(picked)]
        return picked

    def select(self, ctx, allowance):
        if allowance <= 0:
            return []
        if not self._chosen or ctx.cycle - self._chosen_at >= self.refresh_every:
            self._chosen = self._choose_probes(ctx)
            self._chosen_at = ctx.cycle
        cap = max(1, int(allowance * self.max_probe_fraction))
        probes = [
            ctx.grid.get(cid)
            for cid in self._chosen[:cap]
            if ctx.grid.has(cid) and cid not in ctx.excluded
        ]
        spare = allowance - len(probes)
        if spare <= 0:
            return probes
        rest = [c for c in ctx.active_cells if c.id not in self._chosen]
        if not rest:
            return probes
        start = (ctx.cycle * spare) % len(rest)
        rotation = [rest[(start + i) % len(rest)] for i in range(min(spare, len(rest)))]
        return probes + rotation

    def describe(self):
        d = super().describe()
        d.update({"n_probes": self.n_probes, "probes": list(self._chosen)})
        return d


def build_sampler(kind: str, **kwargs) -> Sampler:
    return SAMPLERS.build(kind, **kwargs)
