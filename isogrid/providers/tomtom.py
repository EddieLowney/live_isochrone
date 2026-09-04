"""TomTom Matrix Routing v2 adapter.

Two things make TomTom a good first provider: the free tier needs no credit
card, and the API returns free-flow alongside every live measurement, so the
baseline arrives for free and from the same request.

Billing is the part to get right. TomTom charges on matrix *dimensions*, not
cells: if either dimension exceeds five, the cost is the larger dimension times
five. A naive 1 x 37 request therefore bills 185 transactions for 37 answers.
Batching in fives keeps both dimensions inside the cheap branch and bills 37 —
five times less for identical data. That is what `plan()` does.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from ..registry import PROVIDERS
from ..types import Cell, Coord, Measurement, ProviderError
from .base import Batch, TravelTimeProvider

ENDPOINT = "https://api.tomtom.com/routing/matrix/2"

#: Cell-level failures TomTom counts against your transaction quota even though
#: they return no route. "Billed" here means "deducted from the daily
#: allowance", not "charged" — the free tier has no payment method and simply
#: returns 403/429 once the quota is gone. Tracked so wasted allowance is
#: visible rather than silently eating the safety margin.
BILLED_FAILURES = {"MAP_MATCHING_FAILURE", "NO_ROUTE_FOUND"}


def matrix_cost(n_origins: int, n_destinations: int) -> int:
    """TomTom's billing rule, verbatim.

    Both dimensions at most five: origins times destinations. Otherwise the
    larger dimension times five.
    """
    if n_origins <= 5 and n_destinations <= 5:
        return n_origins * n_destinations
    return max(n_origins, n_destinations) * 5


@PROVIDERS.register("tomtom")
class TomTomMatrixProvider(TravelTimeProvider):
    budget_key = "tomtom"
    #: Not URL path segments like Mapbox — these map onto request options.
    live_profile = "live"
    baseline_profile = "historical"

    def __init__(
        self,
        api_key: Optional[str] = None,
        key_env: str = "TOMTOM_API_KEY",
        batch_size: int = 5,
        travel_mode: str = "car",
        timeout_s: float = 25.0,
        retries: int = 2,
        backoff_s: float = 2.0,
        min_request_interval_s: float = 0.25,
        avoid: Optional[Sequence[str]] = None,
        **_ignored,
    ) -> None:
        self._key = api_key or os.environ.get(key_env, "")
        # Five is not arbitrary. Six would move both dimensions past the cheap
        # branch and multiply the bill by five.
        self.batch_size = min(int(batch_size), 5)
        self.travel_mode = travel_mode
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        self.min_request_interval_s = float(min_request_interval_s)
        self.avoid = list(avoid) if avoid else []
        self._last_call = 0.0
        #: Cell failures TomTom charged for. Surfaced in describe().
        self.billed_failures = 0

    @property
    def max_destinations(self) -> int:
        return self.batch_size

    def cost_of(self, n_destinations: int) -> int:
        return matrix_cost(1, n_destinations)

    def plan(self, cells: Sequence[Cell]) -> list[Batch]:
        out: list[Batch] = []
        for i in range(0, len(cells), self.batch_size):
            chunk = tuple(cells[i : i + self.batch_size])
            out.append(Batch(cells=chunk, elements=self.cost_of(len(chunk))))
        return out

    # -- transport (overridden in tests) ---------------------------------

    def _post(self, url: str, body: dict) -> dict:
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code in (403, 429):
                # 403 with OVER_TRANSACTION_LIMIT means the daily allowance is
                # gone upstream even though our ledger thinks otherwise.
                raise ProviderError(f"tomtom quota/auth ({exc.code}): {detail}")
            raise ProviderError(f"tomtom http {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            raise ProviderError(f"tomtom transport: {exc.reason}")

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - gap)
        self._last_call = time.monotonic()

    # -- provider interface ----------------------------------------------

    def _options(self, profile: str, depart_at: Optional[datetime]) -> dict[str, Any]:
        opts: dict[str, Any] = {"travelMode": self.travel_mode, "routeType": "fastest"}
        if self.avoid:
            opts["avoid"] = self.avoid
        if profile == "live":
            # traffic=live is incompatible with departAt=any, so "now" is
            # required rather than merely recommended.
            opts["traffic"] = "live"
            opts["departAt"] = "now"
        else:
            opts["traffic"] = "historical"
            if depart_at is not None:
                opts["departAt"] = depart_at.astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
            else:
                # Time-context-free: the closest thing TomTom offers to a
                # stable free-flow reference.
                opts["departAt"] = "any"
        return opts

    def fetch(
        self,
        origin: Coord,
        batch: Batch,
        profile: str,
        depart_at: Optional[datetime] = None,
    ) -> list[Measurement]:
        if not self._key:
            raise ProviderError("no TomTom API key configured (set TOMTOM_API_KEY)")

        body = {
            "origins": [{"point": {"latitude": origin.lat, "longitude": origin.lon}}],
            "destinations": [
                {"point": {"latitude": c.center.lat, "longitude": c.center.lon}}
                for c in batch.cells
            ],
            "options": self._options(profile, depart_at),
        }
        url = f"{ENDPOINT}?key={self._key}"

        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                payload = self._post(url, body)
                return self._parse(payload, batch, profile, depart_at)
            except ProviderError as exc:
                last = exc
                if "quota/auth" in str(exc) or attempt >= self.retries:
                    break
                time.sleep(self.backoff_s * (2**attempt))
        raise ProviderError(f"tomtom request failed: {last}")

    def _parse(
        self,
        payload: dict,
        batch: Batch,
        profile: str,
        depart_at: Optional[datetime],
    ) -> list[Measurement]:
        # A 200 can still carry per-cell failures, so results are matched by
        # destinationIndex rather than assumed to be in request order.
        now = depart_at or datetime.now(timezone.utc)
        out: list[Measurement] = []
        for item in payload.get("data", []):
            idx = item.get("destinationIndex")
            if idx is None or idx >= len(batch.cells):
                continue
            summary = item.get("routeSummary")
            if not summary:
                code = (
                    (item.get("detailedError") or {}).get("innerError", {}).get("code", "")
                )
                if code in BILLED_FAILURES:
                    self.billed_failures += 1
                continue
            secs = summary.get("travelTimeInSeconds")
            if secs is None:
                continue
            delay = summary.get("trafficDelayInSeconds")
            free_flow = float(secs) - float(delay) if delay is not None else None
            dist = summary.get("lengthInMeters")
            out.append(
                Measurement(
                    cell_id=batch.cells[idx].id,
                    seconds=float(secs),
                    observed_at=now,
                    provider=self.budget_key,
                    profile=profile,
                    elements=batch.elements / max(1, len(batch.cells)),
                    free_flow_seconds=free_flow,
                    distance_m=float(dist) if dist is not None else None,
                )
            )
        return out

    def describe(self) -> dict:
        d = super().describe()
        d.update({"batch_size": self.batch_size, "billed_failures": self.billed_failures})
        return d
