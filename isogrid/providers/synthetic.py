"""Offline provider that fabricates plausible travel times.

This is not a stub. It generates congestion from a small number of latent
factors driving corridor-shaped loadings, which is the structure the PCA
reconstructor is supposed to exploit. If a sampling or reconstruction change
does not improve results here, it will not improve them against Mapbox either.

It also matches the real provider's batching and cost model, so budget
accounting is exercised identically in test and production.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Optional, Sequence

from ..registry import PROVIDERS
from ..types import Cell, Coord, Measurement, ProviderError, utcnow
from .base import Batch, TravelTimeProvider


@PROVIDERS.register("synthetic")
class SyntheticProvider(TravelTimeProvider):
    """Deterministic given (cell, timestamp) and seed.

    Congestion model::

        live = freeflow * (1 + sum_k loading[cell, k] * factor_k(t) + noise)

    with `n_factors` corridors whose loadings fall off by angular distance from
    a corridor bearing. Low rank by construction.
    """

    budget_key = "synthetic"

    def __init__(
        self,
        seed: int = 7,
        n_factors: int = 4,
        base_speed_kmh: float = 42.0,
        corridor_bonus: float = 0.55,
        noise: float = 0.02,
        failure_rate: float = 0.0,
        dead_cells: Optional[Sequence[str]] = None,
        max_coords_per_request: int = 25,
        peak_hours: Sequence[float] = (8.0, 17.5),
        **_ignored,
    ) -> None:
        self.seed = int(seed)
        self.n_factors = int(n_factors)
        self.base_speed_kmh = float(base_speed_kmh)
        self.corridor_bonus = float(corridor_bonus)
        self.noise = float(noise)
        self.failure_rate = float(failure_rate)
        # Cells the router cannot reach — open water, closed campuses, bad
        # geometry. Requested and billed, but nothing comes back.
        self.dead_cells = set(dead_cells or ())
        self.max_coords_per_request = int(max_coords_per_request)
        self.peak_hours = tuple(peak_hours)
        rng = random.Random(self.seed)
        # Corridor bearings and how sharply loading decays away from them.
        self._corridors = [
            (rng.uniform(0, 360), rng.uniform(1.5, 3.2), rng.uniform(0.35, 0.95))
            for _ in range(self.n_factors)
        ]

    # -- ground truth ----------------------------------------------------

    def freeflow_seconds(self, origin: Coord, cell: Cell) -> float:
        """Free-flow duration. Anisotropic: arterials radiate outward, so travel
        along a corridor bearing is faster per kilometre than across it."""
        d = origin.haversine_km(cell.center)
        if d < 1e-6:
            return 0.0
        bearing = origin.bearing_deg(cell.center)
        speedup = 1.0
        for b, _sharp, _amp in self._corridors:
            ang = math.radians(min((bearing - b) % 360, (b - bearing) % 360))
            speedup += self.corridor_bonus * math.cos(ang) ** 8 if ang < math.pi / 2 else 0.0
        # Detour factor: road distance exceeds straight-line, more so nearby.
        detour = 1.35 - 0.15 * min(d / 20.0, 1.0)
        hours = (d * detour) / (self.base_speed_kmh * speedup)
        return hours * 3600.0 + 45.0

    def factor_values(self, when: datetime) -> list[float]:
        """Latent congestion factors at a moment. Smooth in time, peaking at the
        configured rush hours, with each factor phase-shifted."""
        t = when.astimezone(timezone.utc)
        hour = t.hour + t.minute / 60.0 + t.second / 3600.0
        vals = []
        for k, (_b, sharp, amp) in enumerate(self._corridors):
            level = 0.0
            for peak in self.peak_hours:
                shift = peak + 0.7 * k - 0.35 * len(self.peak_hours)
                level += math.exp(-((hour - shift) ** 2) / (2 * 1.15**2))
            slow = 0.15 * math.sin(2 * math.pi * (t.timetuple().tm_yday + hour / 24) / 7.0 + k)
            vals.append(amp * (level + slow) * (sharp / 2.5))
        return vals

    def loadings(self, origin: Coord, cell: Cell) -> list[float]:
        """How exposed a cell is to each factor. Corridor-shaped, scaled by
        distance because congestion accumulates over a longer path."""
        d = origin.haversine_km(cell.center)
        if d < 1e-6:
            return [0.0] * self.n_factors
        bearing = origin.bearing_deg(cell.center)
        reach = min(d / 12.0, 1.0)
        out = []
        for b, sharp, _amp in self._corridors:
            ang = math.radians(min((bearing - b) % 360, (b - bearing) % 360))
            out.append(reach * math.exp(-((ang * sharp) ** 2)))
        return out

    def _noise_for(self, cell: Cell, when: datetime) -> float:
        if self.noise <= 0:
            return 0.0
        bucket = int(when.timestamp() // 300)
        rng = random.Random(f"{self.seed}:{cell.id}:{bucket}")
        return rng.gauss(0.0, self.noise)

    def true_seconds(self, origin: Coord, cell: Cell, when: datetime, profile: str) -> float:
        base = self.freeflow_seconds(origin, cell)
        if profile == self.baseline_profile:
            return base
        factors = self.factor_values(when)
        load = self.loadings(origin, cell)
        mult = 1.0 + sum(l * f for l, f in zip(load, factors)) + self._noise_for(cell, when)
        return base * max(0.85, mult)

    # -- provider interface ----------------------------------------------

    def fetch(
        self,
        origin: Coord,
        batch: Batch,
        profile: str,
        depart_at: Optional[datetime] = None,
    ) -> list[Measurement]:
        when = depart_at or utcnow()
        if self.failure_rate > 0:
            rng = random.Random(f"{self.seed}:fail:{when.timestamp()}:{len(batch)}")
            if rng.random() < self.failure_rate:
                raise ProviderError("synthetic transport failure")
        per_cell = batch.elements / max(1, len(batch.cells))
        return [
            Measurement(
                cell_id=c.id,
                seconds=self.true_seconds(origin, c, when, profile),
                observed_at=when,
                provider=self.budget_key,
                profile=profile,
                elements=per_cell,
                # Mirrors TomTom, which reports free-flow with every live value.
                free_flow_seconds=self.freeflow_seconds(origin, c),
                distance_m=origin.haversine_km(c.center) * 1000 * 1.35,
            )
            for c in batch.cells
            if c.id not in self.dead_cells
        ]
