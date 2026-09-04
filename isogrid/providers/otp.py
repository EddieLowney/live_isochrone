"""OpenTripPlanner 2 provider — walking plus public transit, from a local graph.

Two things make this different from every other provider here.

**There is no quota.** OTP runs on your machine against MBTA's GTFS and
GTFS-RT feeds. Queries cost CPU, not money, so `plan()` returns one batch and
the ledger charge is nominal. The sampler and the factor reconstructor become
unnecessary — measure every cell, every cycle.

**One departure time is not an answer.** Transit travel time is a step function
of when you leave: miss a bus on a 15-minute headway and the number jumps by 15
minutes. A single query returns a snapshot of one arbitrary moment and will
flicker between cycles for reasons that have nothing to do with conditions. So
each cell is queried at several departure times across a window and reported as
a distribution — p10, median, p90. The spread is the headway penalty, and it is
arguably more informative than the median.

The walking is not bolted on. OTP routes access, transfer, and egress legs over
the OSM pedestrian network as part of the same search, and returns a plain walk
when transit would be slower — the same behaviour as Google Maps.
"""

from __future__ import annotations

import json
import statistics
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

from ..registry import PROVIDERS
from ..types import Cell, Coord, Measurement, ProviderError
from .base import Batch, TravelTimeProvider

#: OTP 2.2+ serves the GTFS GraphQL API here; older builds use the index path.
DEFAULT_PATHS = ("/otp/gtfs/v1", "/otp/routers/default/index/graphql")

#: Itinerary fields requested. Optional because the GTFS GraphQL schema drifts
#: between OTP versions — 2.5 has no `transfers` on Itinerary, for instance.
#: Anything the server rejects as undefined is dropped and the query retried,
#: so a schema difference degrades a column rather than breaking the run.
ITINERARY_FIELDS = ["duration", "walkTime", "waitingTime", "walkDistance"]
LEG_FIELDS = ["mode", "duration", "distance"]

#: Leg modes that are not transit. Transfers are derived by counting transit
#: legs rather than trusting a schema field to exist.
NON_TRANSIT_MODES = {
    "WALK", "BICYCLE", "BICYCLE_RENT", "CAR", "CAR_PICKUP", "CAR_DROPOFF",
    "SCOOTER", "SCOOTER_RENT", "FLEX", "CAR_HAILING",
}


def build_plan_query(itinerary_fields: Sequence[str]) -> str:
    return (
        "query Plan($from: InputCoordinates!, $to: InputCoordinates!, "
        "$date: String!, $time: String!, $modes: [TransportMode!], "
        "$walkSpeed: Float, $maxTransfers: Int, $n: Int) {\n"
        "  plan(from: $from, to: $to, date: $date, time: $time,\n"
        "       transportModes: $modes, walkSpeed: $walkSpeed,\n"
        "       numItineraries: $n, maxTransfers: $maxTransfers) {\n"
        "    itineraries {\n"
        f"      {' '.join(itinerary_fields)}\n"
        f"      legs {{ {' '.join(LEG_FIELDS)} }}\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def undefined_field(errors: Any) -> Optional[str]:
    """Pull the field name out of a graphql-java FieldUndefined validation error."""
    try:
        for e in errors:
            msg = str(e.get("message", ""))
            if "FieldUndefined" in msg and "@[" in msg:
                path = msg.split("@[", 1)[1].split("]", 1)[0]
                return path.rsplit("/", 1)[-1]
    except Exception:  # noqa: BLE001
        pass
    return None


def count_transfers(itinerary: dict) -> int:
    """Transit legs minus one. Zero for a walk-only itinerary."""
    legs = itinerary.get("legs") or []
    transit = [l for l in legs if str(l.get("mode", "")).upper() not in NON_TRANSIT_MODES]
    return max(0, len(transit) - 1)


@PROVIDERS.register("otp")
class OTPProvider(TravelTimeProvider):
    """Queries a local OTP instance once per cell per departure sample."""

    budget_key = "otp"
    live_profile = "transit"
    #: Walk-only. Used as the free-flow reference: transit is "worth it" only
    #: where it beats walking, and the ratio makes that legible.
    baseline_profile = "walk"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        graphql_path: Optional[str] = None,
        timezone_name: str = "America/New_York",
        walk_speed_ms: float = 1.33,
        max_transfers: int = 3,
        max_walk_minutes: float = 20.0,
        profile_window_minutes: float = 30.0,
        profile_samples: int = 5,
        concurrency: int = 8,
        timeout_s: float = 30.0,
        retries: int = 1,
        **_ignored,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.graphql_path = graphql_path
        self.tz = ZoneInfo(timezone_name)
        self.walk_speed_ms = float(walk_speed_ms)
        self.max_transfers = int(max_transfers)
        self.max_walk_minutes = float(max_walk_minutes)
        #: Width of the departure window sampled per cell.
        self.profile_window_minutes = float(profile_window_minutes)
        #: How many departures inside that window. 1 disables profiling.
        self.profile_samples = max(1, int(profile_samples))
        self.concurrency = max(1, int(concurrency))
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self._resolved_path: Optional[str] = graphql_path
        self.last_unreachable: list[str] = []
        self._itinerary_fields = list(ITINERARY_FIELDS)
        self._query = build_plan_query(self._itinerary_fields)
        self.dropped_fields: list[str] = []
        self._lock = threading.Lock()

    # -- batching ---------------------------------------------------------

    @property
    def max_destinations(self) -> int:
        return 10_000  # local compute; no meaningful per-request ceiling

    def cost_of(self, n_destinations: int) -> int:
        # Charged at one unit per cell so the ledger still records volume and
        # the runaway guard still works. It is not money.
        return max(1, n_destinations)

    def plan(self, cells: Sequence[Cell]) -> list[Batch]:
        if not cells:
            return []
        return [Batch(cells=tuple(cells), elements=self.cost_of(len(cells)))]

    # -- transport --------------------------------------------------------

    def _endpoint(self) -> str:
        return f"{self.base_url}{self._resolved_path or DEFAULT_PATHS[0]}"

    def _post(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"otp http {exc.code}: {exc.read().decode(errors='replace')[:300]}"
            )
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"cannot reach OTP at {self.base_url} ({exc.reason}). "
                "Is the server running? See otp/README.md"
            )

    def resolve_path(self) -> str:
        """Find whichever GraphQL path this OTP build serves."""
        if self._resolved_path:
            return self._resolved_path
        errors = []
        for path in DEFAULT_PATHS:
            try:
                out = self._post(
                    f"{self.base_url}{path}", {"query": "{ feeds { feedId } }"}
                )
                if "data" in out:
                    self._resolved_path = path
                    return path
                errors.append(f"{path}: no data key")
            except ProviderError as exc:
                errors.append(f"{path}: {exc}")
        raise ProviderError("no working GraphQL path found — " + "; ".join(errors))

    # -- querying ---------------------------------------------------------

    def _modes(self, profile: str) -> list[dict]:
        if profile == "walk":
            return [{"mode": "WALK"}]
        return [{"mode": "WALK"}, {"mode": "TRANSIT"}]

    def _departures(self, when: datetime) -> list[datetime]:
        """Evenly spaced departures across the profile window."""
        if self.profile_samples == 1 or self.profile_window_minutes <= 0:
            return [when]
        step = self.profile_window_minutes * 60.0 / self.profile_samples
        return [when + timedelta(seconds=step * i) for i in range(self.profile_samples)]

    def _query_one(
        self, origin: Coord, cell: Cell, depart: datetime, profile: str
    ) -> Optional[dict]:
        local = depart.astimezone(self.tz)
        variables = {
            "from": {"lat": origin.lat, "lon": origin.lon},
            "to": {"lat": cell.center.lat, "lon": cell.center.lon},
            "date": local.strftime("%Y-%m-%d"),
            "time": local.strftime("%H:%M:%S"),
            "modes": self._modes(profile),
            "walkSpeed": self.walk_speed_ms,
            "maxTransfers": self.max_transfers,
            "n": 1,
        }
        last: Optional[Exception] = None
        for _ in range(self.retries + 2):
            try:
                out = self._post(
                    self._endpoint(), {"query": self._query, "variables": variables}
                )
                errors = out.get("errors")
                if errors:
                    field = undefined_field(errors)
                    if field and field in self._itinerary_fields:
                        # This OTP build does not expose the field. Drop it and
                        # retry rather than failing the whole run.
                        with self._lock:
                            if field in self._itinerary_fields:
                                self._itinerary_fields.remove(field)
                                self.dropped_fields.append(field)
                                self._query = build_plan_query(self._itinerary_fields)
                        continue
                    raise ProviderError(f"otp graphql: {errors[:1]}")
                its = ((out.get("data") or {}).get("plan") or {}).get("itineraries") or []
                if not its:
                    return None
                return min(its, key=lambda i: i.get("duration") or 1e9)
            except ProviderError as exc:
                last = exc
        raise ProviderError(str(last))

    def fetch(
        self,
        origin: Coord,
        batch: Batch,
        profile: str,
        depart_at: Optional[datetime] = None,
    ) -> list[Measurement]:
        when = depart_at or datetime.now(timezone.utc)
        self.resolve_path()
        # Walking is deterministic, so profiling it is pure waste.
        departures = [when] if profile == "walk" else self._departures(when)

        def run(cell: Cell) -> Optional[Measurement]:
            got = []
            for d in departures:
                it = self._query_one(origin, cell, d, profile)
                if it is not None:
                    got.append(it)
            if not got:
                return None
            return self._summarise(cell, got, when, profile)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(run, batch.cells))

        out = [m for m in results if m is not None]
        self.last_unreachable = [
            c.id for c, m in zip(batch.cells, results) if m is None
        ]
        return out

    def _summarise(
        self, cell: Cell, itineraries: list[dict], when: datetime, profile: str
    ) -> Measurement:
        durations = sorted(float(i["duration"]) for i in itineraries)
        median = statistics.median(durations)

        def pct(p: float) -> float:
            if len(durations) == 1:
                return durations[0]
            k = (len(durations) - 1) * p
            lo, hi = int(k), min(int(k) + 1, len(durations) - 1)
            return durations[lo] + (durations[hi] - durations[lo]) * (k - lo)

        walk = statistics.mean(float(i.get("walkTime") or 0) for i in itineraries)
        wait = statistics.mean(float(i.get("waitingTime") or 0) for i in itineraries)
        # Derived from the legs, not read from a schema field: `transfers` does
        # not exist on Itinerary in OTP 2.5, and counting is version-proof.
        transfers = int(
            round(statistics.mean(count_transfers(i) for i in itineraries))
        )
        dist = statistics.mean(float(i.get("walkDistance") or 0) for i in itineraries)
        return Measurement(
            cell_id=cell.id,
            # The median, not the first sample: robust to one unlucky departure.
            seconds=median,
            observed_at=when,
            provider=self.budget_key,
            profile=profile,
            elements=1,
            distance_m=dist or None,
            p10_seconds=pct(0.10),
            p50_seconds=median,
            p90_seconds=pct(0.90),
            samples=len(durations),
            walk_seconds=walk,
            wait_seconds=wait,
            transfers=transfers,
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            {
                "base_url": self.base_url,
                "graphql_path": self._resolved_path or "unresolved",
                "profile_window_minutes": self.profile_window_minutes,
                "profile_samples": self.profile_samples,
                "max_transfers": self.max_transfers,
                "walk_speed_ms": self.walk_speed_ms,
                "dropped_fields": self.dropped_fields,
            }
        )
        return d
