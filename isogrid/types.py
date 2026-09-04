"""Value types shared across the system.

Everything here is a plain dataclass with no behaviour beyond arithmetic on its
own fields. Modules depend on these types rather than on each other, which is
what keeps providers, samplers and reconstructors independently swappable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

EARTH_RADIUS_KM = 6371.0088


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Coord:
    """A WGS84 point. Latitude first, matching every mapping convention except
    Mapbox's URL format, which the provider adapter handles on its own."""

    lat: float
    lon: float

    def offset_km(self, north_km: float, east_km: float) -> "Coord":
        """Local flat-earth offset. Accurate enough over the tens of kilometres
        a single-origin travel-time grid spans."""
        dlat = math.degrees(north_km / EARTH_RADIUS_KM)
        dlon = math.degrees(east_km / (EARTH_RADIUS_KM * math.cos(math.radians(self.lat))))
        return Coord(self.lat + dlat, self.lon + dlon)

    def haversine_km(self, other: "Coord") -> float:
        p1, p2 = math.radians(self.lat), math.radians(other.lat)
        dp = p2 - p1
        dl = math.radians(other.lon - self.lon)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

    def bearing_deg(self, other: "Coord") -> float:
        """Initial compass bearing toward another point, in [0, 360)."""
        p1, p2 = math.radians(self.lat), math.radians(other.lat)
        dl = math.radians(other.lon - self.lon)
        y = math.sin(dl) * math.cos(p2)
        x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


@dataclass(frozen=True)
class Cell:
    """One sampling location. `polygon` is the drawable outline; `center` is the
    point actually sent to the routing provider."""

    id: str
    center: Coord
    polygon: tuple[Coord, ...] = ()
    ring: int = 0


@dataclass(frozen=True)
class Measurement:
    """A single observed travel time. `elements` records what the call cost so
    the ledger and the measurement history stay reconcilable."""

    cell_id: str
    seconds: float
    observed_at: datetime
    provider: str
    profile: str
    elements: int = 1
    #: Free-flow duration, when the provider returns it alongside the live
    #: value. TomTom does; Mapbox does not. When present it gives a baseline
    #: from the same request, so the delta carries no cross-call drift.
    free_flow_seconds: Optional[float] = None
    #: Route distance. Duration alone is not comparable across cells; distance
    #: over duration is a speed, which is the quantity that generalises.
    distance_m: Optional[float] = None
    #: Departure-window profile. Transit travel time is a step function of when
    #: you leave — miss a bus on a 15-minute headway and the answer changes by
    #: 15 minutes. A single departure is a snapshot of one arbitrary moment, so
    #: transit providers sample a window and report the distribution.
    p10_seconds: Optional[float] = None
    p50_seconds: Optional[float] = None
    p90_seconds: Optional[float] = None
    samples: Optional[int] = None
    #: Journey composition, for diagnosing *why* a cell is slow.
    walk_seconds: Optional[float] = None
    wait_seconds: Optional[float] = None
    transfers: Optional[int] = None
    #: Seconds until the journey time jumps by more than a threshold — the
    #: "leave now or wait for the next one" cliff. None when the sampled window
    #: contains no cliff.
    next_jump_s: Optional[float] = None

    @property
    def spread_seconds(self) -> Optional[float]:
        """p90 minus p10. The headway penalty: how much worse it gets if you
        arrive at the stop at a bad moment."""
        if self.p10_seconds is None or self.p90_seconds is None:
            return None
        return self.p90_seconds - self.p10_seconds


@dataclass
class CellState:
    """Everything the renderer needs about one cell."""

    cell_id: str
    center: Coord
    baseline_s: Optional[float] = None
    observed_s: Optional[float] = None
    observed_at: Optional[datetime] = None
    estimated_s: Optional[float] = None
    is_direct: bool = False
    #: Excluded before any quota was spent: water, out of area, or unroutable.
    masked: bool = False
    mask_reason: Optional[str] = None
    #: Transit extras, carried through to the renderer.
    p10_s: Optional[float] = None
    p90_s: Optional[float] = None
    walk_s: Optional[float] = None
    wait_s: Optional[float] = None
    transfers: Optional[int] = None
    next_jump_s: Optional[float] = None

    @property
    def spread_s(self) -> Optional[float]:
        if self.p10_s is None or self.p90_s is None:
            return None
        return self.p90_s - self.p10_s

    @property
    def delta_s(self) -> Optional[float]:
        """Live minus free-flow. The quantity the map is really about."""
        live = self.estimated_s if self.estimated_s is not None else self.observed_s
        if live is None or self.baseline_s is None:
            return None
        return live - self.baseline_s

    @property
    def ratio(self) -> Optional[float]:
        live = self.estimated_s if self.estimated_s is not None else self.observed_s
        if live is None or self.baseline_s is None:
            return None
        if self.baseline_s == 0:
            # The origin cell. Zero travel time to itself, so no congestion.
            return 1.0
        return live / self.baseline_s

    def staleness_s(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.observed_at is None:
            return None
        return ((now or utcnow()) - self.observed_at).total_seconds()


@dataclass
class Surface:
    """A full reconstructed field, ready to serialise to the renderer."""

    origin: Coord
    generated_at: datetime
    cells: list[CellState] = field(default_factory=list)
    directly_measured: int = 0
    elements_spent: int = 0
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "origin": {"lat": self.origin.lat, "lon": self.origin.lon},
            "generated_at": self.generated_at.isoformat(),
            "directly_measured": self.directly_measured,
            "elements_spent": self.elements_spent,
            "notes": self.notes,
            "cells": [
                {
                    "id": c.cell_id,
                    "lat": c.center.lat,
                    "lon": c.center.lon,
                    "baseline_s": c.baseline_s,
                    "live_s": c.estimated_s if c.estimated_s is not None else c.observed_s,
                    "delta_s": c.delta_s,
                    "ratio": c.ratio,
                    "direct": c.is_direct,
                    "masked": c.masked,
                    "mask_reason": c.mask_reason,
                    "p10_s": c.p10_s,
                    "p90_s": c.p90_s,
                    "spread_s": c.spread_s,
                    "walk_s": c.walk_s,
                    "wait_s": c.wait_s,
                    "transfers": c.transfers,
                    "next_jump_s": c.next_jump_s,
                    "staleness_s": c.staleness_s(self.generated_at),
                }
                for c in self.cells
            ],
        }


class BudgetExhausted(RuntimeError):
    """Raised when a spend is refused. Callers degrade, they do not retry."""


class ProviderError(RuntimeError):
    """Transport or upstream failure. Distinguished from BudgetExhausted so the
    engine knows whether a refund is owed."""
