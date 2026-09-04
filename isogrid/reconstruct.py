"""Estimating the cells you did not measure.

Every reconstructor works in *ratio space* — live seconds divided by baseline
seconds — rather than raw seconds. That removes the dominant spatial gradient
(far cells simply take longer) and leaves the congestion signal, which is the
part that is actually correlated across cells and therefore predictable.

Reconstructors are fitted offline against the observation log and applied online
against whatever subset was measured this cycle. Swapping one for another
requires no re-measurement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from .registry import RECONSTRUCTORS
from .types import Cell, Coord


class Reconstructor(ABC):
    """Maps a sparse set of observed ratios to a dense field."""

    #: Set True by fit() when the model is usable.
    fitted: bool = False

    def fit(
        self,
        history: Sequence[tuple[int, dict[str, float]]],
        baseline: dict[str, float],
        cells: Sequence[Cell],
    ) -> None:
        """Optional. Stateless reconstructors ignore this."""
        self.fitted = True

    @abstractmethod
    def estimate(
        self,
        cells: Sequence[Cell],
        observed_ratio: dict[str, float],
        baseline: dict[str, float],
    ) -> dict[str, float]:
        """Return an estimated ratio for every cell in `cells`."""

    def diagnostics(self) -> dict:
        return {"type": getattr(self, "registry_name", type(self).__name__), "fitted": self.fitted}


@RECONSTRUCTORS.register("passthrough")
class PassthroughReconstructor(Reconstructor):
    """Last observed value, or 1.0 (free-flow) if never observed. The honest
    baseline to benchmark everything else against."""

    def __init__(self, **_ignored):
        self.fitted = True

    def estimate(self, cells, observed_ratio, baseline):
        return {c.id: observed_ratio.get(c.id, 1.0) for c in cells}


@RECONSTRUCTORS.register("idw")
class IDWReconstructor(Reconstructor):
    """Inverse-distance weighting over observed neighbours.

    Assumes congestion is spatially smooth. True near a jam, false across a
    river or a highway boundary — which is exactly where it will look worst.
    """

    def __init__(self, power: float = 2.0, k: int = 6, max_km: float = 25.0, **_ignored):
        self.power = float(power)
        self.k = int(k)
        self.max_km = float(max_km)
        self.fitted = True

    def estimate(self, cells, observed_ratio, baseline):
        obs = [(c, observed_ratio[c.id]) for c in cells if c.id in observed_ratio]
        if not obs:
            return {c.id: 1.0 for c in cells}
        out: dict[str, float] = {}
        for cell in cells:
            if cell.id in observed_ratio:
                out[cell.id] = observed_ratio[cell.id]
                continue
            scored = []
            for other, val in obs:
                d = cell.center.haversine_km(other.center)
                if d <= self.max_km:
                    scored.append((max(d, 1e-3), val))
            if not scored:
                out[cell.id] = 1.0
                continue
            scored.sort()
            near = scored[: self.k]
            wts = [1.0 / (d**self.power) for d, _ in near]
            out[cell.id] = sum(w * v for w, (_, v) in zip(wts, near)) / sum(wts)
        return out


@RECONSTRUCTORS.register("factor")
class FactorReconstructor(Reconstructor):
    """PCA on the historical ratio surface, solved from a sparse cross-section.

    Congestion propagates along corridors, so the cross-section of ratios is
    low-rank. Fit components once offline; each cycle, regress the observed
    ratios on the components restricted to observed cells to recover the factor
    scores, then project back out to the full grid.

    Falls back to IDW until enough history exists to fit.
    """

    def __init__(
        self,
        n_components: int = 4,
        min_cycles: int = 20,
        min_coverage: float = 0.25,
        ridge: float = 1e-3,
        impute_iters: int = 6,
        clip: tuple[float, float] = (0.7, 4.0),
        **_ignored,
    ):
        self.n_components = int(n_components)
        self.min_cycles = int(min_cycles)
        self.min_coverage = float(min_coverage)
        self.ridge = float(ridge)
        self.impute_iters = int(impute_iters)
        self.clip = tuple(clip)
        self.fitted = False
        self._cells: list[str] = []
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None
        self._explained: list[float] = []
        self._fallback = IDWReconstructor()
        self._n_cycles = 0

    def fit(self, history, baseline, cells):
        self.fitted = False
        if len(history) < self.min_cycles:
            self._n_cycles = len(history)
            return
        ids = [c.id for c in cells if baseline.get(c.id)]  # zero-baseline origin excluded
        idx = {cid: j for j, cid in enumerate(ids)}
        M = np.full((len(history), len(ids)), np.nan)
        for i, (_cycle, snap) in enumerate(history):
            for cid, secs in snap.items():
                j = idx.get(cid)
                b = baseline.get(cid)
                if j is not None and b:
                    M[i, j] = secs / b

        coverage = np.mean(~np.isnan(M), axis=0)
        keep = coverage >= self.min_coverage
        if keep.sum() < self.n_components + 2:
            self._n_cycles = len(history)
            return
        M = M[:, keep]
        self._cells = [cid for cid, k in zip(ids, keep) if k]

        # Iterative mean imputation. Crude next to full probabilistic PCA, but
        # it converges fast and keeps the dependency surface small.
        col_mean = np.nanmean(M, axis=0)
        col_mean = np.where(np.isnan(col_mean), 1.0, col_mean)
        missing = np.isnan(M)
        M = np.where(missing, col_mean, M)
        comps = None
        for _ in range(self.impute_iters):
            mean = M.mean(axis=0)
            X = M - mean
            _u, s, vt = np.linalg.svd(X, full_matrices=False)
            k = min(self.n_components, vt.shape[0])
            comps = vt[:k]
            recon = mean + (X @ comps.T) @ comps
            if not missing.any():
                break
            M = np.where(missing, recon, M)

        self._mean = M.mean(axis=0)
        self._components = comps
        total = float(np.sum(s**2)) or 1.0
        self._explained = [float(v) for v in (s[: comps.shape[0]] ** 2) / total]
        self._n_cycles = len(history)
        self.fitted = True

    def estimate(self, cells, observed_ratio, baseline):
        if not self.fitted or self._components is None or self._mean is None:
            return self._fallback.estimate(cells, observed_ratio, baseline)

        pos = {cid: j for j, cid in enumerate(self._cells)}
        rows = [(pos[cid], val) for cid, val in observed_ratio.items() if cid in pos]
        if len(rows) < self._components.shape[0]:
            return self._fallback.estimate(cells, observed_ratio, baseline)

        sel = np.array([r for r, _ in rows])
        y = np.array([v for _, v in rows]) - self._mean[sel]
        B = self._components[:, sel].T
        # Ridge-regularised least squares: the observed subset can be poorly
        # conditioned when probes cluster on one corridor.
        A = B.T @ B + self.ridge * np.eye(B.shape[1])
        scores = np.linalg.solve(A, B.T @ y)
        full = self._mean + self._components.T @ scores
        full = np.clip(full, self.clip[0], self.clip[1])

        out = {}
        for cell in cells:
            if cell.id in observed_ratio:
                out[cell.id] = observed_ratio[cell.id]
            elif cell.id in pos:
                out[cell.id] = float(full[pos[cell.id]])
            else:
                out[cell.id] = 1.0
        return out

    def diagnostics(self):
        d = super().diagnostics()
        d.update(
            {
                "n_components": self.n_components,
                "history_cycles": self._n_cycles,
                "modelled_cells": len(self._cells),
                "explained_variance_ratio": [round(v, 4) for v in self._explained],
            }
        )
        return d


def build_reconstructor(kind: str, **kwargs) -> Reconstructor:
    return RECONSTRUCTORS.build(kind, **kwargs)
