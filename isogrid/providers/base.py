"""Routing provider interface.

Batching lives here rather than in the engine because the rules are provider
specific and they drive cost. Mapbox bills per element and caps coordinates per
request; TomTom bills on matrix dimensions and rewards a different shape
entirely. Both express that through `plan()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from ..registry import PROVIDERS
from ..types import Cell, Coord, Measurement


@dataclass(frozen=True)
class Batch:
    """One outbound request and what it will cost."""

    cells: tuple[Cell, ...]
    elements: int

    def __len__(self) -> int:
        return len(self.cells)


class TravelTimeProvider(ABC):
    """One origin, many destinations, live durations back."""

    #: Ledger key. Distinct providers must not share a budget pool.
    budget_key: str = "default"
    #: Coordinates the upstream API accepts per request, origin included.
    max_coords_per_request: int = 25
    #: Profile used for live measurement.
    live_profile: str = "driving-traffic"
    #: Profile used to build the free-flow baseline. Must be the same engine as
    #: `live_profile`, or the delta carries a provider bias that never cancels.
    baseline_profile: str = "driving"

    @property
    def max_destinations(self) -> int:
        return max(1, self.max_coords_per_request - 1)

    def plan(self, cells: Sequence[Cell]) -> list[Batch]:
        """Split destinations into billable requests. Default is the Mapbox
        shape: one source against N destinations, billed N elements."""
        out: list[Batch] = []
        step = self.max_destinations
        for i in range(0, len(cells), step):
            chunk = tuple(cells[i : i + step])
            out.append(Batch(cells=chunk, elements=self.cost_of(len(chunk))))
        return out

    def cost_of(self, n_destinations: int) -> int:
        """Billable units for a 1 x n request."""
        return max(1, n_destinations)

    @abstractmethod
    def fetch(
        self,
        origin: Coord,
        batch: Batch,
        profile: str,
        depart_at: Optional[datetime] = None,
    ) -> list[Measurement]:
        """Execute one request. Raise ProviderError on transport failure so the
        engine can refund the ledger."""

    def describe(self) -> dict:
        return {
            "name": getattr(self, "registry_name", type(self).__name__),
            "budget_key": self.budget_key,
            "max_destinations": self.max_destinations,
            "live_profile": self.live_profile,
            "baseline_profile": self.baseline_profile,
        }


def build_provider(kind: str, **kwargs) -> TravelTimeProvider:
    return PROVIDERS.build(kind, **kwargs)
