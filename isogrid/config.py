"""Configuration.

Every pluggable component is named by a string plus a bag of options, so
changing the sampling algorithm, the grid scale, or the provider is a YAML edit.
Unknown options are tolerated by the registry, which means one config file can
carry settings for several alternatives at once while you compare them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import Coord

DEFAULTS: dict[str, Any] = {
    "origin": {"lat": 42.3519, "lon": -71.0552, "label": "South Station, Boston"},
    "grid": {"kind": "hex", "cell_km": 2.0, "rings": 3},
    "provider": {"kind": "synthetic", "seed": 7, "n_factors": 4},
    "store": {"kind": "sqlite", "path": "data/surface.db", "record_estimates": True},
    "budget": {
        "ledger": {"kind": "sqlite", "path": "data/budget.db"},
        "period": "monthly",
        "safety": 0.9,
        "quotas": {"mapbox": 100000, "synthetic": 100000, "tomtom_tiles": 50000},
        "pacer": {"kind": "proportional", "burst": 3.0, "floor": 1},
    },
    "mask": {
        "kind": "composite",
        "masks": [{"kind": "quarantine", "failures_before_quarantine": 3}],
    },
    "sampling": {"kind": "probe", "n_probes": 12, "max_probe_fraction": 0.7, "refresh_every": 288},
    "reconstruct": {"kind": "factor", "n_components": 4, "min_cycles": 20, "min_coverage": 0.05},
    "engine": {
        "cycle_seconds": 300,
        "refit_every": 12,
        "history_cycles": 500,
        "validation_holdout": 3,
        "max_batches_per_cycle": 8,
    },
    "render": {
        "mode": "delta",
        "absolute_min_minutes": 0,
        "absolute_max_minutes": 45,
        "delta_range_minutes": [-3, 15],
        "stale_after_seconds": 1800,
        "poll_seconds": 20,
        "initial_zoom": 15,
        "spread_max_minutes": 20,
        "cliff_max_minutes": 15,
        "basemap": {
            "enabled": True,
            "style": "night",
            "tile_size": 512,
            "cache_dir": "data/tiles",
            "key_env": "TOMTOM_API_KEY",
            "budget_key": "tomtom_tiles",
            "language": "en-US",
            "labels": True,
            "labels_min_zoom": 13,
            "labels_opacity": 0.65,
            "base_opacity": 1.0,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Config:
    raw: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[dict] = None) -> "Config":
        data: dict = {}
        if path:
            import yaml

            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        merged = _deep_merge(DEFAULTS, data)
        if overrides:
            merged = _deep_merge(merged, overrides)
        return cls(raw=merged)

    # -- accessors --------------------------------------------------------

    def section(self, name: str) -> dict:
        return copy.deepcopy(self.raw.get(name, {}))

    @property
    def origin(self) -> Coord:
        o = self.raw["origin"]
        return Coord(float(o["lat"]), float(o["lon"]))

    @property
    def origin_label(self) -> str:
        return str(self.raw["origin"].get("label", ""))

    def kind_and_options(self, section: str) -> tuple[str, dict]:
        block = self.section(section)
        kind = block.pop("kind", None)
        if kind is None:
            raise KeyError(f"config section '{section}' needs a 'kind'")
        return kind, block

    @property
    def cycle_seconds(self) -> float:
        return float(self.raw["engine"]["cycle_seconds"])

    def build(self) -> "Components":
        from .budget import build_ledger, build_pacer
        from .grid import build_grid
        from .masks import build_mask
        from .providers.base import build_provider
        from .reconstruct import build_reconstructor
        from .sampling import build_sampler
        from .store import build_store

        gk, gopts = self.kind_and_options("grid")
        pk, popts = self.kind_and_options("provider")
        sk, sopts = self.kind_and_options("store")
        smk, smopts = self.kind_and_options("sampling")
        rk, ropts = self.kind_and_options("reconstruct")
        mk, mopts = self.kind_and_options("mask")

        budget = self.section("budget")
        lk = budget["ledger"].pop("kind")
        pacer_cfg = budget.get("pacer", {})
        pacer_kind = pacer_cfg.pop("kind", "proportional")

        return Components(
            config=self,
            grid=build_grid(gk, origin=self.origin, **gopts),
            provider=build_provider(pk, **popts),
            store=build_store(sk, **sopts),
            sampler=build_sampler(smk, **smopts),
            reconstructor=build_reconstructor(rk, **ropts),
            mask=build_mask(mk, **mopts),
            ledger=build_ledger(
                lk,
                quotas=budget["quotas"],
                period=budget["period"],
                safety=budget["safety"],
                **budget["ledger"],
            ),
            pacer=build_pacer(pacer_kind, **pacer_cfg),
        )


@dataclass
class Components:
    """Fully constructed system. The engine takes one of these."""

    config: Config
    grid: Any
    provider: Any
    store: Any
    sampler: Any
    reconstructor: Any
    mask: Any
    ledger: Any
    pacer: Any

    def describe(self) -> dict:
        return {
            "origin": {"label": self.config.origin_label, **self.config.section("origin")},
            "grid": self.grid.describe(),
            "provider": self.provider.describe(),
            "sampler": self.sampler.describe(),
            "reconstructor": self.reconstructor.diagnostics(),
            "mask": self.mask.describe(),
            "render": self.config.section("render"),
        }
