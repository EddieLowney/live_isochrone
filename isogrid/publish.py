"""Publish the map as static files, so it can be hosted without a live server.

`serve` computes on request, and GitHub Pages cannot run code. This turns the
same pipeline into a writer: every engine cycle the surface is rendered to JSON
and pushed to one or more sinks — a local directory for testing, Cloudflare R2
for the deployed site — and the front end fetches those files instead of /api.

Three cadences:

* surface, status    every engine cycle (30 s), riding `CycleScheduler`
* vehicles           every `vehicles_s` (10 s) on their own thread, so a slow
                     cycle cannot stall the trains
* config, routes,    once, retried each cycle until every sink accepts them,
  stations           and again whenever the static feed changes

Nothing here computes anything new. Every payload is exactly what the matching
/api endpoint returns, so the front end cannot tell the two apart except by URL.
"""

from __future__ import annotations

import copy
import gzip
import io
import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional

from .types import utcnow

EMPTY_FC = {"type": "FeatureCollection", "features": []}
MBTA_GTFS_URL = "https://cdn.mbta.com/MBTA_GTFS.zip"


# -- sinks -------------------------------------------------------------------

class DirSink:
    """Writes files under a directory. For running the whole loop locally."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, name: str, body: bytes, content_type: str, cache_control: str) -> None:
        # Write, then rename. A reader mid-fetch must see the old file or the
        # new one, never a truncated hybrid — which a 2.9 MB surface rewritten
        # in place every 30 s would eventually guarantee.
        dest = self.root / name
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, dest)

    def __str__(self) -> str:
        return f"dir:{self.root}"


class R2Sink:
    """Cloudflare R2, through its S3-compatible API.

    Credentials come only from the environment, never from config or argv:
    config is committed to the repo, and argv shows up in `ps`.
    """

    ENV = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

    def __init__(self, bucket: str, endpoint: str = "", key_id: str = "",
                 secret: str = "", prefix: str = "", gzip_bodies: bool = True,
                 client=None):
        if client is None:
            import boto3

            client = boto3.client(
                "s3", endpoint_url=endpoint, aws_access_key_id=key_id,
                aws_secret_access_key=secret, region_name="auto")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self.gzip_bodies = gzip_bodies

    @classmethod
    def from_env(cls, env=None, **kw) -> "R2Sink":
        env = os.environ if env is None else env
        missing = [k for k in cls.ENV if not env.get(k)]
        if missing:
            # Names only. Echoing values would put a secret in a log.
            raise SystemExit("R2 needs these environment variables: " + ", ".join(missing))
        return cls(
            bucket=env["R2_BUCKET"],
            endpoint=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            key_id=env["R2_ACCESS_KEY_ID"], secret=env["R2_SECRET_ACCESS_KEY"],
            prefix=env.get("R2_PREFIX", ""), **kw)

    def put(self, name: str, body: bytes, content_type: str, cache_control: str) -> None:
        kw = {"Bucket": self.bucket, "Key": self.prefix + name, "Body": body,
              "ContentType": content_type, "CacheControl": cache_control}
        if self.gzip_bodies:
            # 2.9 MB of surface becomes 330 KB. R2 stores Content-Encoding and
            # returns it, so browsers decompress transparently. mtime=0 keeps
            # identical payloads byte-identical, so an unchanged file keeps its
            # ETag and a revalidating browser gets a 304.
            kw["Body"] = gzip.compress(body, compresslevel=6, mtime=0)
            kw["ContentEncoding"] = "gzip"
        self.client.put_object(**kw)

    def __str__(self) -> str:
        return f"r2:{self.bucket}/{self.prefix}"


# -- static feed -------------------------------------------------------------

class FeedRefresher:
    """Keeps the static GTFS current.

    A static feed is a timetable with an expiry date. MBTA's covers about a
    month and is republished every week or two. Loaded once and left, a
    long-running service routes against a calendar that no longer contains
    today, and transit quietly drains out of the map: 1,839 cells beat walking
    in August, 63 in November, on a feed that had expired in September.

    Checked with a conditional GET, so an unchanged feed costs a 304. A new one
    is validated before it replaces anything — a truncated download or an HTML
    error page must never overwrite a working feed — and the previous one is
    kept beside it as `.prev`. The provider adopts it at the next 03:00
    service-day rollover.
    """

    REQUIRED = ("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")

    def __init__(self, path, url: str = MBTA_GTFS_URL, every_s: float = 6 * 3600,
                 opener: Optional[Callable] = None, timeout_s: float = 120.0):
        self.path = Path(path)
        self.url = url
        self.every_s = float(every_s)
        self.opener = opener or urllib.request.urlopen
        self.timeout_s = timeout_s
        self.last_check: Optional[float] = None     # monotonic
        self.checked_at: Optional[str] = None       # wall clock, for status
        self.last_error: Optional[str] = None
        self.updates = 0

    def maybe_refresh(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.last_check is not None and now - self.last_check < self.every_s:
            return False
        self.last_check = now
        try:
            changed = self.refresh()
            self.last_error = None
            return changed
        except Exception as exc:  # noqa: BLE001
            # The current feed stays. Stale for a few more hours beats absent.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def refresh(self) -> bool:
        headers = {"User-Agent": "isogrid-feed-refresh"}
        if self.path.is_file():
            headers["If-Modified-Since"] = formatdate(self.path.stat().st_mtime, usegmt=True)
        req = urllib.request.Request(self.url, headers=headers)
        try:
            with self.opener(req, timeout=self.timeout_s) as resp:
                body = resp.read()
                modified = resp.headers.get("Last-Modified")
        except urllib.error.HTTPError as exc:
            self.checked_at = utcnow().isoformat()
            if exc.code == 304:
                return False
            raise
        self.checked_at = utcnow().isoformat()
        self._validate(body)

        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(body)
        if modified:
            # Stamp the file with the server's own Last-Modified, so the next
            # If-Modified-Since asks precisely "anything newer than this one?"
            ts = parsedate_to_datetime(modified).timestamp()
            os.utime(tmp, (ts, ts))
        if self.path.is_file():
            prev = self.path.with_name(self.path.name + ".prev")
            prev.unlink(missing_ok=True)
            # A hard link, then an atomic rename over the original: the feed
            # path never stops existing, not even between the two steps.
            try:
                os.link(self.path, prev)
            except OSError:
                import shutil

                shutil.copy2(self.path, prev)
        os.replace(tmp, self.path)
        self.updates += 1
        return True

    def _validate(self, body: bytes) -> None:
        try:
            zf = zipfile.ZipFile(io.BytesIO(body))
        except zipfile.BadZipFile:
            raise ValueError(
                f"download is not a zip ({len(body)} bytes); keeping the current feed")
        with zf:
            missing = [n for n in self.REQUIRED if n not in set(zf.namelist())]
            if missing:
                raise ValueError(f"download lacks {missing}; keeping the current feed")
            bad = zf.testzip()
            if bad:
                raise ValueError(f"download is corrupt at {bad}; keeping the current feed")

    def describe(self) -> dict:
        mtime = self.path.stat().st_mtime if self.path.is_file() else None
        return {
            "path": str(self.path), "checked_at": self.checked_at,
            "updates": self.updates, "last_error": self.last_error,
            "feed_modified": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime))
                              if mtime else None),
        }


# -- publisher ---------------------------------------------------------------

class Publisher:
    """Renders each payload once and hands the bytes to every sink."""

    #: Live files are revalidated on every fetch rather than cached for a
    #: guessed lifetime: unchanged costs a 304, changed is never hidden.
    LIVE = "no-cache"
    #: Static files change only when the feed does.
    STATIC = "public, max-age=300"

    def __init__(self, engine, sinks, overlays=None, feed=None, vehicles_s: float = 10.0):
        self.engine = engine
        self.vehicles_s = float(vehicles_s)
        self.sinks = list(sinks)
        if overlays is None:
            from .server import TransitOverlays

            overlays = TransitOverlays(engine)
        self.overlays = overlays
        self.feed = feed
        self.static_ok = False
        self.errors: dict = {}
        self.writes: Counter = Counter()

    def _publish(self, name: str, build: Callable[[], dict], cache: str) -> bool:
        try:
            body = json.dumps(build(), separators=(",", ":")).encode()
        except Exception as exc:  # noqa: BLE001
            self.errors[name] = f"build: {type(exc).__name__}: {exc}"
            return False
        ok = True
        for sink in self.sinks:
            try:
                sink.put(name, body, "application/json", cache)
            except Exception as exc:  # noqa: BLE001
                # One sink failing must not starve the others, nor stop the loop.
                ok = False
                self.errors[name] = f"{sink}: {type(exc).__name__}: {exc}"
        if ok:
            self.errors.pop(name, None)
            self.writes[name] += 1
        return ok

    def config_payload(self) -> dict:
        cfg = copy.deepcopy(self.engine.c.describe())
        # A static host has no tile proxy, and a TomTom key must never ship to
        # a browser — so the published page always takes the keyless base.
        cfg.setdefault("render", {}).setdefault("basemap", {})["labels"] = False
        cfg["basemap"] = {"available": False}
        # The page polls vehicles at this rate. Left at the server's 5 s floor,
        # every other poll would re-read a file that has not changed — half of
        # all bucket reads, for nothing.
        render = cfg["render"]
        render["overlays"] = dict(render.get("overlays") or {},
                                  vehicles_min_refresh_s=self.vehicles_s)
        cfg["overlays"] = {"enabled": bool(self.overlays.enabled)}
        return cfg

    def publish_static(self) -> bool:
        ov = self.overlays
        ok = self._publish("config.json", self.config_payload, self.STATIC)
        ok &= self._publish("routes.json",
                            ov.routes_geojson if ov.enabled else lambda: EMPTY_FC,
                            self.STATIC)
        ok &= self._publish("stations.json",
                            ov.stations_geojson if ov.enabled else lambda: EMPTY_FC,
                            self.STATIC)
        self.static_ok = ok
        return ok

    def publish_surface(self, report=None) -> None:
        self._publish("surface.json", self.engine.geojson, self.LIVE)
        self._publish("status.json", lambda: self.status_payload(report), self.LIVE)

    def publish_vehicles(self) -> None:
        ov = self.overlays
        self._publish("vehicles.json",
                      ov.vehicles_geojson if ov.enabled else lambda: EMPTY_FC,
                      self.LIVE)

    def reload_overlays(self) -> None:
        """Rebuild everything derived from the static feed.

        Built fresh rather than patched: route lines, stations and the vehicle
        shape index all come from the feed. The vehicle tracker restarts with
        them, which costs one fix of playback once every few weeks.
        """
        from .server import TransitOverlays

        self.overlays = TransitOverlays(self.engine)
        self.static_ok = False

    def status_payload(self, report=None) -> dict:
        return {
            "published_at": utcnow().isoformat(),
            "cycle_report": report.as_dict() if report is not None else None,
            "errors": dict(self.errors),
            "writes": dict(self.writes),
            "sinks": [str(s) for s in self.sinks],
            "feed": self.feed.describe() if self.feed else None,
        }


def run(engine, sinks, *, vehicles_s: float = 10.0, feed: Optional[FeedRefresher] = None,
        once: bool = False, log: Callable = print) -> Publisher:
    """Publish until interrupted, or for one cycle with `once`."""
    from .server import CycleScheduler

    # Before anything touches the provider. It loads the feed on its first
    # fetch of any kind — the walk-only baseline included — and then keeps it
    # until 03:00, so refreshing even one step later still runs today on the
    # old timetable.
    if feed is not None:
        if feed.maybe_refresh():
            log(f"static feed updated: {feed.describe()['feed_modified']}")
        elif feed.last_error:
            log(f"feed check failed, keeping current feed: {feed.last_error}")
    if not engine.c.store.get_baseline():
        log("no baseline yet — measuring it now")
        engine.bootstrap_baseline()
    pub = Publisher(engine, sinks, feed=feed, vehicles_s=vehicles_s)

    def on_cycle(_surface, report):
        if feed is not None and feed.maybe_refresh():
            log("static feed updated; the engine adopts it at the next 03:00 rollover")
            pub.reload_overlays()
        if not pub.static_ok:
            pub.publish_static()
        pub.publish_surface(report)

    if once:
        surface, report = engine.run_cycle()
        on_cycle(surface, report)
        pub.publish_vehicles()
        return pub

    stop = threading.Event()

    def vehicles_loop():
        while not stop.is_set():
            started = time.monotonic()
            pub.publish_vehicles()
            stop.wait(max(0.0, vehicles_s - (time.monotonic() - started)))

    # SIGTERM matters as much as Ctrl-C: it is how systemd stops a service.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    sched = CycleScheduler(engine, on_cycle=on_cycle)
    sched.start()
    threading.Thread(target=vehicles_loop, name="vehicles", daemon=True).start()
    log(f"publishing to {', '.join(map(str, pub.sinks))}: surface every "
        f"{sched.interval_s:.0f}s, vehicles every {vehicles_s:.0f}s")
    try:
        # One line a minute: enough to read the health of an unattended box
        # from its journal without drowning it.
        while not stop.wait(60):
            rep = sched.last_report or {}
            log(f"cycle {rep.get('cycle', '-')} in {sched.last_duration_s or 0:.1f}s · "
                f"writes {sum(pub.writes.values())} · "
                f"errors {pub.errors or sched.last_error or 'none'}")
    finally:
        sched.stop()
    return pub


def check(url: str, origin: str, log: Callable = print,
          names=("config.json", "surface.json", "vehicles.json")) -> bool:
    """Fetch the published files exactly as a browser on `origin` would.

    Checks the three things that can be wrong without any request failing:
    public access, the CORS header for that origin, and whether a gzipped body
    arrives labelled as gzip — unlabelled, it reaches the page as binary and
    the JSON parse fails with nothing to say why.
    """
    ok = True
    for name in names:
        target = f"{url.rstrip('/')}/{name}"
        req = urllib.request.Request(target, headers={
            "Origin": origin, "Accept-Encoding": "gzip",
            "User-Agent": "isogrid-publish-check"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw, hdr = r.read(), r.headers
        except Exception as exc:  # noqa: BLE001
            log(f"FAIL {name}: {type(exc).__name__}: {exc}")
            ok = False
            continue
        enc = hdr.get("Content-Encoding")
        acao = hdr.get("Access-Control-Allow-Origin")
        problems = []
        if enc == "gzip":
            raw = gzip.decompress(raw)
        elif raw[:2] == b"\x1f\x8b":
            problems.append("gzipped body without Content-Encoding: browsers cannot parse it")
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
            problems.append("body is not JSON")
        if acao not in (origin, "*"):
            problems.append(f"no CORS grant for {origin} (got {acao!r})")
        age = ""
        gen = isinstance(payload, dict) and (payload.get("metadata") or {}).get("generated_at")
        if gen:
            from datetime import datetime

            age = f", {(utcnow() - datetime.fromisoformat(gen)).total_seconds():.0f}s old"
        status = "ok  " if not problems else "FAIL"
        ok &= not problems
        log(f"{status} {name}: {len(raw):,} B, encoding={enc or 'none'}{age}"
            + ("" if not problems else " — " + "; ".join(problems)))
    return ok
