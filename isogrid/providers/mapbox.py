"""Mapbox Matrix API adapter.

Request shape is one source against N destinations, which bills N elements —
the cheapest available shape for a single-origin grid. Coordinates are capped
per request and the cap differs by profile, so `max_coords_per_request` is
configurable rather than hardcoded: some accounts see a lower limit on
driving-traffic until Mapbox raises it on request.

Untested against the live API from this sandbox, which has no network. Verify
the driving-traffic coordinate cap and the terms on storing derived results
before running this against a real key.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

from ..registry import PROVIDERS
from ..types import Coord, Measurement, ProviderError
from .base import Batch, TravelTimeProvider

BASE_URL = "https://api.mapbox.com/directions-matrix/v1/mapbox"


@PROVIDERS.register("mapbox")
class MapboxMatrixProvider(TravelTimeProvider):
    budget_key = "mapbox"
    live_profile = "driving-traffic"
    baseline_profile = "driving"

    def __init__(
        self,
        access_token: Optional[str] = None,
        token_env: str = "MAPBOX_TOKEN",
        max_coords_per_request: int = 25,
        timeout_s: float = 20.0,
        retries: int = 2,
        backoff_s: float = 1.5,
        min_request_interval_s: float = 2.1,
        **_ignored,
    ) -> None:
        self._token = access_token or os.environ.get(token_env, "")
        self.max_coords_per_request = int(max_coords_per_request)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        # Client-side pacing. The published ceiling is 30 requests/minute; this
        # keeps us under it without relying on the server to say no.
        self.min_request_interval_s = float(min_request_interval_s)
        self._last_call = 0.0

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - gap)
        self._last_call = time.monotonic()

    def fetch(
        self,
        origin: Coord,
        batch: Batch,
        profile: str,
        depart_at: Optional[datetime] = None,
    ) -> list[Measurement]:
        if not self._token:
            raise ProviderError("no Mapbox access token configured")
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("requests is required for the Mapbox provider") from exc

        # Mapbox takes lon,lat. Origin is index 0 and the only source.
        pts = [origin] + [c.center for c in batch.cells]
        coords = ";".join(f"{p.lon:.6f},{p.lat:.6f}" for p in pts)
        params = {
            "access_token": self._token,
            "sources": "0",
            "destinations": ";".join(str(i) for i in range(1, len(pts))),
            "annotations": "duration",
        }
        if depart_at is not None:
            params["depart_at"] = depart_at.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M"
            )

        url = f"{BASE_URL}/{profile}/{coords}"
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                resp = requests.get(url, params=params, timeout=self.timeout_s)
                if resp.status_code == 429:
                    time.sleep(self.backoff_s * (2**attempt))
                    last = ProviderError("rate limited")
                    continue
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("code") not in (None, "Ok"):
                    raise ProviderError(f"mapbox: {payload.get('code')}")
                row = (payload.get("durations") or [[]])[0]
                if len(row) != len(batch.cells):
                    raise ProviderError("duration row length does not match request")
                now = depart_at or datetime.now(timezone.utc)
                out = []
                for cell, secs in zip(batch.cells, row):
                    if secs is None:
                        # Unroutable destination. Still billed, so it is recorded
                        # as spent but yields no measurement.
                        continue
                    out.append(
                        Measurement(
                            cell_id=cell.id,
                            seconds=float(secs),
                            observed_at=now,
                            provider=self.budget_key,
                            profile=profile,
                            elements=1,
                        )
                    )
                return out
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2**attempt))
        raise ProviderError(f"mapbox request failed: {last}")
