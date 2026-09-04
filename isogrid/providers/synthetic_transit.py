"""Offline stand-in for OTP, with a real headway model.

The point is not to fake plausible numbers. It is to reproduce the one property
that makes transit different from driving: **travel time is a step function of
departure time**. Leave at 14:04 and catch the 14:05; leave at 14:06 and wait
until 14:20. Any code path that handles departure-window profiling, spread
reporting, or the reconstruction question has to be exercised against that
discontinuity, and a smooth simulator would hide every bug that matters.

Model: radial lines from the origin, each with a headway and a vehicle speed.
Journey time to a cell is::

    min( walk_direct,
         min over stops s of [ wait(s, t) + ride(s) + walk(s -> cell) ] )

where `wait` depends on where the departure time falls in the headway cycle.
"""

from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from ..registry import PROVIDERS
from ..types import Cell, Coord, Measurement, ProviderError
from .base import Batch, TravelTimeProvider


class Line:
    """One transit line: a bearing, a headway, and stops along it."""

    def __init__(self, bearing: float, headway_s: float, speed_kmh: float,
                 stop_spacing_km: float, n_stops: int, offset_s: float):
        self.bearing = bearing
        self.headway_s = headway_s
        self.speed_kmh = speed_kmh
        self.stop_spacing_km = stop_spacing_km
        self.n_stops = n_stops
        self.offset_s = offset_s

    def stops(self, origin: Coord) -> list[tuple[Coord, float]]:
        """(location, ride time from origin) for each stop."""
        out = []
        for i in range(1, self.n_stops + 1):
            d = i * self.stop_spacing_km
            north = d * math.cos(math.radians(self.bearing))
            east = d * math.sin(math.radians(self.bearing))
            # Dwell time at each intermediate stop is real and adds up.
            ride = (d / self.speed_kmh) * 3600.0 + i * 25.0
            out.append((origin.offset_km(north, east), ride))
        return out

    def wait_at(self, t: datetime) -> float:
        secs = t.timestamp() + self.offset_s
        return self.headway_s - (secs % self.headway_s)


@PROVIDERS.register("synthetic_transit")
class SyntheticTransitProvider(TravelTimeProvider):
    budget_key = "synthetic_transit"
    live_profile = "transit"
    baseline_profile = "walk"

    def __init__(
        self,
        seed: int = 11,
        n_lines: int = 5,
        walk_speed_ms: float = 1.33,
        walk_detour: float = 1.35,
        access_walk_max_km: float = 0.8,
        profile_window_minutes: float = 30.0,
        profile_samples: int = 5,
        delay_noise_s: float = 45.0,
        dead_cells: Optional[Sequence[str]] = None,
        failure_rate: float = 0.0,
        **_ignored,
    ) -> None:
        self.seed = int(seed)
        self.walk_speed_ms = float(walk_speed_ms)
        self.walk_detour = float(walk_detour)
        self.access_walk_max_km = float(access_walk_max_km)
        self.profile_window_minutes = float(profile_window_minutes)
        self.profile_samples = max(1, int(profile_samples))
        self.delay_noise_s = float(delay_noise_s)
        self.dead_cells = set(dead_cells or ())
        self.failure_rate = float(failure_rate)
        rng = random.Random(self.seed)
        self.lines = [
            Line(
                bearing=rng.uniform(0, 360),
                # Rapid transit through to infrequent bus.
                headway_s=rng.choice([300, 480, 600, 900, 1200]),
                speed_kmh=rng.uniform(18, 40),
                stop_spacing_km=rng.uniform(0.5, 1.2),
                n_stops=rng.randint(5, 9),
                offset_s=rng.uniform(0, 600),
            )
            for _ in range(int(n_lines))
        ]
        self.last_unreachable: list[str] = []

    # -- ground truth -----------------------------------------------------

    def walk_seconds(self, a: Coord, b: Coord) -> float:
        km = a.haversine_km(b) * self.walk_detour
        return km * 1000.0 / self.walk_speed_ms

    def _stops(self, origin: Coord) -> list[tuple[Coord, float, Line]]:
        out = []
        for line in self.lines:
            for coord, ride in line.stops(origin):
                out.append((coord, ride, line))
        return out

    def journey(self, origin: Coord, cell: Cell, depart: datetime) -> Optional[dict]:
        """Best itinerary leaving at `depart`. None if nothing reaches the cell."""
        direct = self.walk_seconds(origin, cell.center)
        best = {
            "duration": direct, "walkTime": direct, "waitingTime": 0.0,
            "transfers": 0, "walkDistance": origin.haversine_km(cell.center) * 1000,
        }
        for coord, ride, line in self._stops(origin):
            egress = self.walk_seconds(coord, cell.center)
            if egress > self.access_walk_max_km * 1000 / self.walk_speed_ms * 2:
                continue
            wait = line.wait_at(depart)
            if self.delay_noise_s:
                bucket = int(depart.timestamp() // 60)
                rng = random.Random(f"{self.seed}:{id(line)}:{bucket}")
                wait = max(0.0, wait + rng.gauss(0, self.delay_noise_s))
            total = wait + ride + egress
            if total < best["duration"]:
                best = {
                    "duration": total, "walkTime": egress, "waitingTime": wait,
                    "transfers": 0, "walkDistance": coord.haversine_km(cell.center) * 1000,
                }
        return best

    # -- provider interface ------------------------------------------------

    @property
    def max_destinations(self) -> int:
        return 10_000

    def cost_of(self, n_destinations: int) -> int:
        return max(1, n_destinations)

    def plan(self, cells):
        if not cells:
            return []
        return [Batch(cells=tuple(cells), elements=self.cost_of(len(cells)))]

    def _departures(self, when: datetime) -> list[datetime]:
        if self.profile_samples == 1 or self.profile_window_minutes <= 0:
            return [when]
        step = self.profile_window_minutes * 60.0 / self.profile_samples
        return [when + timedelta(seconds=step * i) for i in range(self.profile_samples)]

    def fetch(self, origin, batch, profile, depart_at=None):
        when = depart_at or datetime.now(timezone.utc)
        if self.failure_rate > 0:
            rng = random.Random(f"{self.seed}:fail:{when.timestamp()}")
            if rng.random() < self.failure_rate:
                raise ProviderError("synthetic transit transport failure")
        departures = [when] if profile == "walk" else self._departures(when)
        out: list[Measurement] = []
        unreachable: list[str] = []
        for cell in batch.cells:
            if cell.id in self.dead_cells:
                unreachable.append(cell.id)
                continue
            if profile == "walk":
                d = self.walk_seconds(origin, cell.center)
                out.append(
                    Measurement(
                        cell_id=cell.id, seconds=d, observed_at=when,
                        provider=self.budget_key, profile=profile, elements=1,
                        p50_seconds=d, p10_seconds=d, p90_seconds=d, samples=1,
                        walk_seconds=d, wait_seconds=0.0, transfers=0,
                        distance_m=origin.haversine_km(cell.center) * 1000,
                    )
                )
                continue
            its = [self.journey(origin, cell, d) for d in departures]
            its = [i for i in its if i]
            if not its:
                unreachable.append(cell.id)
                continue
            out.append(self._summarise(cell, its, when, profile))
        self.last_unreachable = unreachable
        return out

    def _summarise(self, cell, its, when, profile) -> Measurement:
        durations = sorted(float(i["duration"]) for i in its)

        def pct(p):
            if len(durations) == 1:
                return durations[0]
            k = (len(durations) - 1) * p
            lo, hi = int(k), min(int(k) + 1, len(durations) - 1)
            return durations[lo] + (durations[hi] - durations[lo]) * (k - lo)

        return Measurement(
            cell_id=cell.id,
            seconds=statistics.median(durations),
            observed_at=when,
            provider=self.budget_key,
            profile=profile,
            elements=1,
            distance_m=statistics.mean(i["walkDistance"] for i in its),
            p10_seconds=pct(0.10),
            p50_seconds=statistics.median(durations),
            p90_seconds=pct(0.90),
            samples=len(durations),
            walk_seconds=statistics.mean(i["walkTime"] for i in its),
            wait_seconds=statistics.mean(i["waitingTime"] for i in its),
            transfers=int(round(statistics.mean(i["transfers"] for i in its))),
        )

    def describe(self):
        d = super().describe()
        d.update({
            "lines": len(self.lines),
            "headways_s": sorted({int(l.headway_s) for l in self.lines}),
            "profile_samples": self.profile_samples,
        })
        return d
