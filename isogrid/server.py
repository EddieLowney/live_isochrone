"""Read-only HTTP surface, a background cycle loop, and a basemap tile proxy.

Deliberately stdlib-only so the prototype has no web framework dependency. The
handler is a thin translation layer over `Engine`, so swapping in FastAPI later
means reimplementing this file and nothing else.

The tile proxy exists for one reason: the browser must not hold the API key.
The same key authorises routing, so shipping it to the page would put your whole
quota one "view source" away. Tiles are fetched server-side, cached to disk, and
metered through the same ledger that guards routing — against a separate
allowance, because TomTom counts tile and non-tile requests apart.
"""

from __future__ import annotations

import json
import mimetypes
import time
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from .engine import Engine
from .types import BudgetExhausted, utcnow

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"

#: /tiles/{layer}/{style}/{z}/{x}/{y}.png
TILE_RE = re.compile(r"^/tiles/(basic|hybrid|labels)/(main|night)/(\d{1,2})/(\d+)/(\d+)\.png$")

TILE_BASE = "https://api.tomtom.com/map/1/tile"

#: 1x1 transparent PNG, served when the tile budget is spent. Better than a
#: broken-image icon: the map degrades to bare colour instead of breaking.
BLANK_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c63000100000500010d0a2db400000"
    "00049454e44ae426082"
)


class TileProxy:
    """Fetches, caches and meters basemap tiles.

    Cache first, budget second, network last. A metro area at one zoom is a few
    dozen tiles, so after the first page load almost every request is a cache
    hit and the tile allowance is barely touched.
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "data/tiles",
        tile_size: int = 512,
        budget_key: str = "tomtom_tiles",
        ledger=None,
        language: str = "en-US",
        timeout_s: float = 15.0,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.tile_size = int(tile_size)
        self.budget_key = budget_key
        self.ledger = ledger
        self.language = language
        self.timeout_s = float(timeout_s)
        self.hits = 0
        self.misses = 0
        self.refused = 0
        self.errors = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _cache_path(self, layer, style, z, x, y) -> Path:
        return self.cache_dir / layer / style / str(z) / str(x) / f"{y}.png"

    def get(self, layer: str, style: str, z: int, x: int, y: int) -> tuple[bytes, str]:
        """Return (png_bytes, source) where source is hit | miss | refused | error."""
        path = self._cache_path(layer, style, z, x, y)
        if path.is_file():
            with self._lock:
                self.hits += 1
            return path.read_bytes(), "hit"

        if self.ledger is not None and not self.ledger.try_spend(self.budget_key, 1):
            with self._lock:
                self.refused += 1
            return BLANK_PNG, "refused"

        url = (
            f"{TILE_BASE}/{layer}/{style}/{z}/{x}/{y}.png"
            f"?key={self.api_key}&tileSize={self.tile_size}&language={self.language}"
        )
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            if self.ledger is not None:
                self.ledger.refund(self.budget_key, 1)
            with self._lock:
                self.errors += 1
            return BLANK_PNG, "error"

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic, so a killed process cannot leave a torn tile
        with self._lock:
            self.misses += 1
        return data, "miss"

    def stats(self) -> dict:
        return {
            "available": self.available,
            "cached_tiles": self.hits + self.misses,
            "hits": self.hits,
            "misses": self.misses,
            "refused": self.refused,
            "errors": self.errors,
            "cache_dir": str(self.cache_dir),
        }


class CycleScheduler:
    """Runs `engine.run_cycle()` on a fixed cadence."""

    def __init__(self, engine: Engine, interval_s: Optional[float] = None,
                 on_cycle: Optional[Callable] = None):
        self.engine = engine
        self.interval_s = float(interval_s or engine.cfg.cycle_seconds)
        #: Called with (surface, report) after each successful cycle — how the
        #: static publisher rides this loop instead of keeping its own clock.
        self.on_cycle = on_cycle
        self.last_report: Optional[dict] = None
        self.last_error: Optional[str] = None
        self.overruns = 0
        #: Wall time of the most recent cycle, and how often it outran the
        #: configured interval. Both are reported by /api/status.
        self.last_duration_s: Optional[float] = None
        self.overruns: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                surface, report = self.engine.run_cycle()
                self.last_report = report.as_dict()
                self.last_error = None
                if self.on_cycle is not None:
                    self.on_cycle(surface, report)
            except BudgetExhausted as exc:
                # Expected terminal state, not a crash. The surface stays live
                # and simply ages.
                self.last_error = f"budget exhausted: {exc}"
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"

            # Sleep to the next period boundary, not for a fixed duration after
            # the work. Sleeping afterwards makes the real interval
            # cycle_seconds PLUS however long the cycle took, so a 21-minute
            # cycle on a 5-minute setting silently becomes 26.
            elapsed = time.monotonic() - started
            self.last_duration_s = elapsed
            if elapsed > self.interval_s:
                self.overruns += 1
                # Surface it rather than drifting quietly.
                self.last_error = (
                    f"cycle took {elapsed:.0f}s but the interval is "
                    f"{self.interval_s:.0f}s — refresh is limited by cycle time, "
                    "not by the schedule"
                )
            self._stop.wait(max(0.0, self.interval_s - elapsed))

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()


class TransitOverlays:
    """Route geometry and live vehicles, served alongside the surface.

    Neither changes a number the map computes — they explain the numbers. Route
    lines say *where* the transit is, so a fast region reads as "the Red Line
    runs through here" rather than as an unexplained patch. Vehicles say why a
    cell just jumped: you can see the train pull away.

    Both come from the GTFS zip and the keyless GTFS-RT feed, so no API key is
    needed for either.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        prov = engine.c.provider
        render = engine.cfg.section("render")
        self.cfg = render.get("overlays", {}) or {}
        self.gtfs_path = getattr(prov, "gtfs_path", None)
        self.cache_dir = getattr(prov, "cache_dir", "data/raptor")
        #: Feed identity for the caches below. Keyed by origin alone, a refreshed
        #: feed would keep serving the old routes and the old trip-to-shape map.
        self.feed_tag = (int(Path(self.gtfs_path).stat().st_mtime)
                         if self.gtfs_path and Path(self.gtfs_path).is_file() else 0)
        self.vehicles_url = self.cfg.get(
            "vehicles_url", "https://cdn.mbta.com/realtime/VehiclePositions.pb")
        self.radius_km = float(self.cfg.get("routes_radius_km", 8.0))
        self.min_refresh_s = float(self.cfg.get("vehicles_min_refresh_s", 10.0))
        self._routes: Optional[dict] = None
        self._stations: Optional[dict] = None
        self.stations_radius_km = float(self.cfg.get("stations_radius_km", 6.0))
        self._fetcher = None
        self._vehicles: list = []
        self._vehicles_at = 0.0
        self.errors: list = []
        self._tracker = None
        self.max_age_s = float(self.cfg.get("vehicle_max_age_s", 180.0))
        self.shapes_radius_km = float(self.cfg.get("shapes_radius_km", 12.0))
        kinds = self.cfg.get("vehicle_modes")
        self.keep_kinds = set(kinds) if kinds else None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True)) and bool(self.gtfs_path)

    def routes_geojson(self) -> dict:
        """Static; built once from shapes.txt and cached to disk."""
        if self._routes is None:
            from .transit.shapes import build_route_geojson

            o = self.engine.cfg.origin
            self._routes = build_route_geojson(
                self.gtfs_path, o.lat, o.lon, radius_km=self.radius_km,
                cache=f"{self.cache_dir}/routes-{o.lat:.4f}_{o.lon:.4f}"
                      f"-{self.radius_km:.0f}km-f{self.feed_tag}.json",
            )
        return self._routes

    def stations_geojson(self) -> dict:
        """Static; rapid-transit stations only, built once and cached."""
        if self._stations is None:
            from .transit.shapes import build_station_geojson

            o = self.engine.cfg.origin
            kinds = {k.replace(" ", "_") for k in (self.cfg.get("station_modes")
                                                   or ["subway", "light rail"])}
            self._stations = build_station_geojson(
                self.gtfs_path, o.lat, o.lon, self.stations_radius_km,
                kinds=kinds,
                cache=f"{self.cache_dir}/stations-{o.lat:.4f}_{o.lon:.4f}"
                      f"-{self.stations_radius_km:.0f}km-f{self.feed_tag}.json")
        return self._stations

    def _vehicle_tracker(self):
        """Built lazily: indexing shapes takes a moment and is only needed once
        a vehicle actually shows up."""
        if self._tracker is None:
            from .transit.vehicles import ShapeIndex, VehicleTracker

            o = self.engine.cfg.origin
            index = None
            try:
                index = ShapeIndex.build(
                    self.gtfs_path, o.lat, o.lon, self.shapes_radius_km,
                    cache=f"{self.cache_dir}/shapeindex-{o.lat:.4f}_{o.lon:.4f}"
                          f"-{self.shapes_radius_km:.0f}km-f{self.feed_tag}.npz")
            except Exception as exc:  # noqa: BLE001
                # Without shapes, tracks fall back to straight lines between
                # fixes. Degraded, not broken.
                self.errors.append(f"shape index unavailable: {exc}")
            route_kinds = {}
            try:
                import csv, io, zipfile
                from .transit.shapes import ROUTE_KIND

                with zipfile.ZipFile(self.gtfs_path) as zf:
                    for r in csv.DictReader(io.StringIO(
                            zf.read("routes.txt").decode("utf-8-sig"))):
                        route_kinds[r["route_id"]] = ROUTE_KIND.get(
                            (r.get("route_type") or "").strip(), "other")
            except Exception:  # noqa: BLE001
                pass
            self._tracker = VehicleTracker(
                index, max_age_s=self.max_age_s,
                route_kinds=route_kinds, keep_kinds=self.keep_kinds)
        return self._tracker

    def vehicles_geojson(self) -> dict:
        """Live, with a floor on refresh so many browser tabs cannot turn into
        many upstream fetches."""
        import time

        from .transit.realtime import VehicleFetcher

        now = time.time()
        if now - self._vehicles_at >= self.min_refresh_s:
            if self._fetcher is None:
                self._fetcher = VehicleFetcher(self.vehicles_url)
            try:
                self._vehicles = self._fetcher.vehicles()
                self._vehicles_at = now
                self.errors = []
            except RuntimeError as exc:
                self.errors = [str(exc)]
        tracker = self._vehicle_tracker()
        tracker.update(self._vehicles, now)
        feats = tracker.segments(self._vehicles, now)
        ages = [f["properties"]["age_s"] for f in feats
                if f["properties"]["age_s"] is not None]
        return {
            "type": "FeatureCollection", "features": feats,
            "metadata": {
                "count": len(feats),
                "server_time": now,
                "median_age_s": sorted(ages)[len(ages) // 2] if ages else None,
                "fetched_s_ago": round(now - self._vehicles_at, 1),
                "errors": self.errors,
                **tracker.describe(),
                "vehicle_buffer_note": "one fix behind; no playback buffer",
                **(self._fetcher.describe() if self._fetcher else {}),
            },
        }


def build_tile_proxy(engine: Engine) -> TileProxy:
    """Wire a proxy from config, taking the key from the provider or the env.

    The key is read server-side only and never appears in any response.
    """
    render = engine.cfg.section("render")
    base = render.get("basemap", {}) or {}
    key = ""
    provider = engine.c.provider
    if getattr(provider, "budget_key", "") == "tomtom":
        key = getattr(provider, "_key", "") or ""
    if not key:
        key = os.environ.get(base.get("key_env", "TOMTOM_API_KEY"), "")
    return TileProxy(
        api_key=key if base.get("enabled", True) else "",
        cache_dir=base.get("cache_dir", "data/tiles"),
        tile_size=int(base.get("tile_size", 512)),
        budget_key=base.get("budget_key", "tomtom_tiles"),
        ledger=engine.c.ledger,
        language=base.get("language", "en-US"),
    )


def make_handler(engine: Engine, scheduler: CycleScheduler, tiles: TileProxy,
                 overlays: "Optional[TransitOverlays]" = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "isogrid/0.2"

        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                m = TILE_RE.match(path)
                if m:
                    return self._tile(m)
                if path == "/api/geojson":
                    return self._json(engine.geojson())
                if path == "/api/routes.geojson":
                    if not (overlays and overlays.enabled):
                        return self._json({"type": "FeatureCollection",
                                           "features": []})
                    # Static for a given origin, so let the browser keep it.
                    body = json.dumps(overlays.routes_geojson()).encode()
                    return self._send(200, body, "application/json",
                                      "public, max-age=86400")
                if path == "/api/stations.geojson":
                    if not (overlays and overlays.enabled):
                        return self._json({"type": "FeatureCollection",
                                           "features": []})
                    body = json.dumps(overlays.stations_geojson()).encode()
                    return self._send(200, body, "application/json",
                                      "public, max-age=86400")
                if path == "/api/vehicles.geojson":
                    if not (overlays and overlays.enabled):
                        return self._json({"type": "FeatureCollection",
                                           "features": []})
                    return self._json(overlays.vehicles_geojson())
                if path == "/api/surface":
                    return self._json(engine.c.store.get_surface() or {})
                if path == "/api/config":
                    cfg = engine.c.describe()
                    cfg["basemap"] = tiles.stats()
                    cfg["overlays"] = {"enabled": bool(overlays and overlays.enabled)}
                    return self._json(cfg)
                if path == "/api/budget":
                    return self._json(engine.c.ledger.snapshot(utcnow()))
                if path == "/api/status":
                    return self._json(
                        {
                            "last_report": scheduler.last_report,
                            "last_error": scheduler.last_error,
                            "interval_s": scheduler.interval_s,
                            "overruns": scheduler.overruns,
                            "last_duration_s": (
                                round(scheduler.last_duration_s, 1)
                                if scheduler.last_duration_s else None
                            ),
                            "effective_interval_s": (
                                round(max(scheduler.interval_s, scheduler.last_duration_s), 1)
                                if scheduler.last_duration_s else scheduler.interval_s
                            ),
                            "overruns": scheduler.overruns,
                            "basemap": tiles.stats(),
                            "server_time": utcnow().isoformat(),
                        }
                    )
                return self._static(path)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def _tile(self, m) -> None:
            layer, style = m.group(1), m.group(2)
            z, x, y = int(m.group(3)), int(m.group(4)), int(m.group(5))
            if z > 22 or x >= 2**z or y >= 2**z:
                return self._json({"error": "tile out of range"}, 400)
            if not tiles.available:
                return self._send(200, BLANK_PNG, "image/png")
            data, source = tiles.get(layer, style, z, x, y)
            # Cached tiles are immutable for our purposes; let the browser keep
            # them so a page reload costs nothing.
            cache = "no-store" if source in ("refused", "error") else "public, max-age=86400"
            self._send(200, data, "image/png", cache)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/cycle":
                return self._json({"error": "not found"}, 404)
            try:
                _surface, report = engine.run_cycle()
                self._json(report.as_dict())
            except BudgetExhausted as exc:
                self._json({"error": str(exc)}, 429)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def _static(self, path: str) -> None:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (WEB_ROOT / rel).resolve()
            if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), ctype)

    return Handler


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8000, autorun: bool = True):
    scheduler = CycleScheduler(engine)
    tiles = build_tile_proxy(engine)
    overlays = TransitOverlays(engine)
    if autorun:
        scheduler.start()
    httpd = ThreadingHTTPServer(
        (host, port), make_handler(engine, scheduler, tiles, overlays))
    print(f"isogrid serving on http://{host}:{port}  (cycle every {scheduler.interval_s:.0f}s)")
    if tiles.available:
        print(f"basemap tiles proxied, cached in {tiles.cache_dir}")
    else:
        print("basemap disabled: no API key found, rendering without a basemap")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.stop()
        httpd.server_close()
