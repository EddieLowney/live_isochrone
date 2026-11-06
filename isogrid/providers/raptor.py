"""RAPTOR provider — transit routing without OTP in the loop.

Where the OTP provider issues one HTTP query per cell per departure, this runs
a single range-RAPTOR search per cycle and composes the result onto cells with
a min-plus product. The consequence is that **per-cycle cost stops depending on
output resolution**: 127 cells and 127,000 cells cost the same search.

Layers, and how often each is paid for:

* GTFS parse -> once per service day, cached
* Footpaths, access, egress matrix -> once per (grid, walk source), cached
* GTFS-RT poll -> once per cycle, one HTTP request
* range-RAPTOR + min-plus -> once per cycle, milliseconds

Set ``walk_source: otp`` to build the walking layers from the running OTP
instance. That costs a one-off pass of walk queries and makes any
RAPTOR-versus-OTP disagreement attributable to the transit algorithm rather
than to two different pedestrian graphs.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from ..registry import PROVIDERS
from ..types import Cell, Coord, Measurement, ProviderError
from .base import Batch, TravelTimeProvider
from ..transit.footpaths import (
    EgressMatrix, GraphWalk, OtpWalk, StraightLineWalk, build_access,
    build_footpaths, haversine_km,
)
from ..transit.gtfs import MAX_TIME, load_gtfs
from ..transit.raptor import range_raptor, raptor
from ..transit.realtime import RealtimeFetcher, TripDelays


@PROVIDERS.register("raptor")
class RaptorProvider(TravelTimeProvider):
    budget_key = "raptor"
    live_profile = "transit"
    baseline_profile = "walk"

    def __init__(
        self,
        gtfs_path: str = "otp/mbta.gtfs.zip",
        cache_dir: str = "data/raptor",
        trip_updates_url: str = "https://cdn.mbta.com/realtime/TripUpdates.pb",
        timezone_name: str = "America/New_York",
        walk_source: str = "straight_line",   # straight_line | graph | otp
        osm_pbf: str = "otp/boston.osm.pbf",
        max_snap_m: float = 250.0,
        crossing_penalty_s: float = 0.0,
        otp_url: str = "http://127.0.0.1:8080",
        walk_speed_ms: float = 1.33,
        walk_detour: float = 1.35,
        access_walk_s: int = 900,
        transfer_walk_s: int = 300,
        egress_walk_s: int = 900,
        # The origin-to-cell walk spans the whole grid, so it needs a far
        # wider bound than a stop-to-cell egress walk.
        direct_walk_s: int = 9000,
        max_rounds: int = 4,
        min_transfer_time: int = 0,
        profile_window_minutes: float = 30.0,
        # Fine sampling on purpose. The quantity of interest is a sawtooth with
        # a period of one headway, so 6 samples over 30 minutes aliases a
        # 10-minute headway into noise. Range-RAPTOR makes extra samples cheap;
        # spend them here.
        profile_samples: int = 31,
        cliff_threshold_s: int = 120,
        realtime: bool = True,
        realtime_max_age_s: float = 180.0,
        **_ignored,
    ) -> None:
        self.gtfs_path = gtfs_path
        self.cache_dir = cache_dir
        self.trip_updates_url = trip_updates_url
        self.tz = ZoneInfo(timezone_name)
        self.walk_source_name = walk_source
        self.osm_pbf = osm_pbf
        self.max_snap_m = float(max_snap_m)
        self.crossing_penalty_s = float(crossing_penalty_s)
        self.otp_url = otp_url
        self.walk_speed_ms, self.walk_detour = walk_speed_ms, walk_detour
        self.access_walk_s = int(access_walk_s)
        self.transfer_walk_s = int(transfer_walk_s)
        self.egress_walk_s = int(egress_walk_s)
        self.direct_walk_s = int(direct_walk_s)
        self.max_rounds = int(max_rounds)
        self.min_transfer_time = int(min_transfer_time)
        self.profile_window_minutes = float(profile_window_minutes)
        # Samples are used to locate the cliff, NOT to average the answer.
        # `seconds` is always the departure-now value; see fetch().
        self.profile_samples = max(1, int(profile_samples))
        self.cliff_threshold_s = int(cliff_threshold_s)
        self.realtime = bool(realtime)
        self.realtime_max_age_s = float(realtime_max_age_s)

        self._fetcher: Optional[RealtimeFetcher] = None
        self._pedgraph = None
        self._direct_cache: Optional[tuple] = None
        self._data = None
        self._service_date = None
        #: mtime of the feed `_data` was parsed from. Part of every derived
        #: cache key, because the stop list those arrays index into belongs to
        #: one specific feed, not to a date.
        self._feed_id: Optional[int] = None
        self._footpaths = None
        self._access: dict[int, int] = {}
        self._egress: Optional[EgressMatrix] = None
        self._egress_key: Optional[str] = None
        self._delays: Optional[TripDelays] = None
        self._delay_array: Optional[np.ndarray] = None
        self._skipped: Optional[np.ndarray] = None
        self.last_unreachable: list[str] = []
        self.stats: dict = {}

    # -- layers -----------------------------------------------------------

    def _walk_source(self, max_seconds: Optional[float] = None):
        """A walk source bounded for one specific layer.

        The cap is not a global setting: access, transfers, egress and the
        direct origin-to-cell walk have genuinely different ranges. Sharing one
        bound silently marks anything beyond it unreachable — with the egress
        cap of 900 s applied to a 4.9 km grid, every cell past a 15 minute
        walk came back as MAX_TIME.
        """
        if self.walk_source_name == "graph":
            return GraphWalk(self._graph(),
                             max_seconds=max_seconds or self.egress_walk_s,
                             max_snap_m=self.max_snap_m)
        if self.walk_source_name == "otp":
            return OtpWalk(base_url=self.otp_url)
        return StraightLineWalk(self.walk_speed_ms, self.walk_detour)

    def _graph(self):
        """Cached pedestrian graph. Built once from the OSM extract."""
        if self._pedgraph is None:
            from ..transit.pedgraph import load_or_build
            t0 = time.time()
            self._pedgraph = load_or_build(
                self.osm_pbf, f"{self.cache_dir}/pedgraph.npz",
                walk_speed_ms=self.walk_speed_ms,
                crossing_penalty_s=self.crossing_penalty_s,
            )
            self.stats["pedgraph_build_s"] = round(time.time() - t0, 1)
            self.stats["pedgraph"] = self._pedgraph.describe()
        return self._pedgraph

    def _ensure_data(self, when: datetime):
        """GTFS for the service day. Reloaded when the day rolls over.

        The service day starts at 03:00 local, not midnight: a trip leaving at
        00:40 belongs to the previous day's schedule, and rolling at midnight
        would drop late-night service every night.

        A feed replaced on disk is adopted at the next rollover, not mid-day.
        Swapping schedules between two cycles would rebuild every walk layer at
        an arbitrary moment; at 03:00 that happens anyway.
        """
        local = when.astimezone(self.tz)
        service_day = (local - timedelta(hours=3)).date()
        if self._data is not None and self._service_date == service_day:
            return
        if not Path(self.gtfs_path).is_file():
            raise ProviderError(
                f"GTFS feed not found at {self.gtfs_path}. "
                "Run otp/setup.sh, or point gtfs_path at a downloaded MBTA_GTFS.zip"
            )
        t0 = time.time()
        self._data = load_gtfs(self.gtfs_path, on=service_day, cache_dir=self.cache_dir)
        self._service_date = service_day
        self._feed_id = int(Path(self.gtfs_path).stat().st_mtime)
        self._footpaths = None
        self._access = {}
        self.stats["gtfs_load_s"] = round(time.time() - t0, 2)
        self.stats["gtfs"] = self._data.describe()

    def _ensure_walk_layers(self, origin: Coord, cells: Sequence[Cell]):
        # The feed belongs in the key as much as the date does. Keyed by date
        # alone, a restart after a new feed landed loaded footpaths, access and
        # egress arrays built against the previous feed's stop list — indexed
        # into the wrong stops, with nothing to say so.
        tag = f"{self.walk_source_name}-{self._service_date}-f{self._feed_id}"
        if self._footpaths is None:
            self._footpaths = build_footpaths(
                self._data, self.transfer_walk_s,
                self._walk_source(self.transfer_walk_s),
                cache=f"{self.cache_dir}/footpaths-{tag}.npz")
            self.stats["footpath_edges"] = int(len(self._footpaths.times))
        if not self._access:
            self._access = build_access(
                self._data, origin.lat, origin.lon, self.access_walk_s,
                self._walk_source(self.access_walk_s),
                cache=f"{self.cache_dir}/access-{tag}-{origin.lat:.5f}_{origin.lon:.5f}.npz")
            self.stats["access_stops"] = len(self._access)
            if not self._access:
                raise ProviderError(
                    f"no stops within {self.access_walk_s}s walk of the origin — "
                    "check the origin coordinates and the GTFS feed")
        pts = np.array([[c.center.lat, c.center.lon] for c in cells])
        # Digest the actual geometry, not `hash()` of a label: string hashing
        # is salted per process, so the old filename changed on every restart
        # and a moved grid could silently reuse a stale matrix.
        digest = hashlib.sha1(
            np.ascontiguousarray(pts, dtype=np.float64).tobytes()
            + f"|{self.egress_walk_s}|{tag}".encode()
        ).hexdigest()[:16]
        if self._egress_key != digest:
            self._egress = EgressMatrix.build(
                self._data, pts, self.egress_walk_s,
                self._walk_source(self.egress_walk_s),
                cache=f"{self.cache_dir}/egress-{digest}.npz")
            self._egress_key = digest
            self.stats["egress"] = self._egress.describe()

    def _ensure_delays(self):
        if not self.realtime:
            self._delay_array = None
            self._skipped = None
            return
        fresh = (
            self._delays is not None
            and self._delays.fetched_at is not None
            and time.time() - self._delays.fetched_at < self.realtime_max_age_s
        )
        if fresh:
            return
        try:
            if self._fetcher is None:
                self._fetcher = RealtimeFetcher(self.trip_updates_url)
            preds = self._fetcher.fetch()
            # MBTA publishes absolute predicted times, so the schedule is
            # required to turn them into delays. Passing `data` is what makes
            # the difference between a live map and a timetable viewer.
            self._delays = preds.resolve(self._data, self._service_epoch())
            self._delay_array = (
                self._delays.per_stoptime
                if self._delays.per_stoptime is not None
                else self._delays.as_array(self._data.trip_ids)
            )
            # Cancellations are not in per_stoptime; overlay them.
            if self._delays.per_stoptime is not None and self._delays.canceled:
                for i, tid in enumerate(self._data.trip_ids):
                    if str(tid) in self._delays.canceled:
                        lo = int(self._data.trip_time_ptr[i])
                        hi = int(self._data.trip_time_ptr[i + 1])
                        self._delay_array[lo:hi] = 10**7
            self._skipped = self._delays.skipped
            self.stats["realtime"] = {**self._delays.describe(),
                                      **self._fetcher.describe()}
        except RuntimeError as exc:
            # Degrade to the published timetable rather than failing the cycle,
            # but say so — a silent fall back to schedule looks identical to
            # working and would be the worst possible failure here.
            self._delay_array = None
            self._skipped = None
            self.stats["realtime"] = {"error": str(exc)}

    # -- provider interface -----------------------------------------------

    @property
    def max_destinations(self) -> int:
        return 1_000_000

    def cost_of(self, n_destinations: int) -> int:
        return 1  # one search, regardless of how many cells it lands on

    def plan(self, cells: Sequence[Cell]) -> list[Batch]:
        return [Batch(cells=tuple(cells), elements=1)] if cells else []

    def _direct_walk(self, origin: Coord, pts: np.ndarray) -> np.ndarray:
        """Origin to every cell on foot.

        Goes through the configured walk source rather than a hardcoded
        straight line: this is the walk baseline *and* the floor of the
        min-plus composition, so a straight-line value here would quietly
        cap every cell's travel time at an optimistic number.
        """
        key = (id(pts), pts.shape, float(pts[0, 0]), float(pts[-1, 0]),
               origin.lat, origin.lon, self.walk_source_name)
        if self._direct_cache and self._direct_cache[0] == key[1:]:
            return self._direct_cache[1]
        src = self._walk_source(self.direct_walk_s)
        if isinstance(src, StraightLineWalk):
            out = (haversine_km(origin.lat, origin.lon, pts[:, 0], pts[:, 1])
                   * 1000.0 * self.walk_detour / self.walk_speed_ms).astype(np.int64)
        else:
            row = src.times(np.array([[origin.lat, origin.lon]]), pts)[0]
            # No walking route at all is still a number the min-plus can use;
            # transit may reach the cell even when walking cannot.
            out = np.where(row < 0, MAX_TIME, row).astype(np.int64)
        self._direct_cache = (key[1:], out)
        return out

    def _service_epoch(self) -> float:
        """Absolute epoch of midnight starting the current service day.

        GTFS times are offsets from this, including the past-24:00 values that
        carry late-night service, so predicted absolute times subtract against
        it directly.
        """
        from datetime import time as _time

        midnight = datetime.combine(self._service_date, _time(0, 0), tzinfo=self.tz)
        return midnight.timestamp()

    def _seconds_of_day(self, when: datetime) -> int:
        local = when.astimezone(self.tz)
        base = local.replace(hour=0, minute=0, second=0, microsecond=0)
        secs = int((local - base).total_seconds())
        # Before 03:00 we are on the previous service day, so express the time
        # as 24:xx rather than 00:xx to match the GTFS convention.
        if local.hour < 3:
            secs += 86400
        return secs

    def fetch(self, origin, batch, profile, depart_at=None):
        when = depart_at or datetime.now(timezone.utc)
        self._ensure_data(when)
        self._ensure_walk_layers(origin, batch.cells)
        cells = batch.cells
        pts = np.array([[c.center.lat, c.center.lon] for c in cells])
        direct = self._direct_walk(origin, pts)

        if profile == "walk":
            out, unwalkable = [], []
            for i, c in enumerate(cells):
                d = float(direct[i])
                if direct[i] >= MAX_TIME:
                    # Genuinely not walkable within the bound — report it as
                    # unreachable rather than emitting a sentinel as a duration.
                    unwalkable.append(c.id)
                    continue
                out.append(Measurement(
                    cell_id=c.id, seconds=d, observed_at=when,
                    provider=self.budget_key, profile=profile, elements=0,
                    p10_seconds=d, p50_seconds=d, p90_seconds=d, samples=1,
                    walk_seconds=d, wait_seconds=0.0, transfers=0,
                ))
            self.last_unreachable = unwalkable
            return out

        self._ensure_delays()
        t0 = time.time()
        prof = range_raptor(
            self._data, self._access, self._seconds_of_day(when),
            window_seconds=int(self.profile_window_minutes * 60),
            samples=self.profile_samples, footpaths=self._footpaths,
            max_rounds=self.max_rounds, delays=self._delay_array,
            min_transfer_time=self.min_transfer_time, skipped=self._skipped,
        )
        self.stats["raptor_s"] = round(time.time() - t0, 3)

        # Compose each departure sample separately, then take percentiles over
        # cells. Composing the median stop-arrival instead would be wrong: the
        # median of a min is not the min of medians.
        n = len(cells)
        per_departure = np.empty((self.profile_samples, n), dtype=np.int64)
        for k in range(self.profile_samples):
            per_departure[k] = self._egress.compose(
                prof.journey[k], direct, MAX_TIME)
        self.stats["compose_s"] = round(time.time() - t0, 3)

        # The cliff is computed on the composed per-cell curves, not on stop
        # arrivals: a cell served by two stops on different headways has a
        # cliff neither stop has on its own.
        cliff = self._cliffs(per_departure)

        out: list[Measurement] = []
        unreachable: list[str] = []
        for i, c in enumerate(cells):
            col = per_departure[:, i]
            good = col[col < MAX_TIME]
            if not good.size:
                unreachable.append(c.id)
                continue
            live = col[0]
            if live >= MAX_TIME:
                # Unreachable right now but reachable later in the window.
                # Report the first departure that works rather than a median.
                live = int(good[0])
            out.append(
                Measurement(
                    cell_id=c.id,
                    # Leaving NOW. Not the median: a median over a departure
                    # window answers a planning question and erases exactly the
                    # temporal edge this map exists to show.
                    seconds=float(live),
                    observed_at=when,
                    provider=self.budget_key, profile=profile, elements=0,
                    # p10 is best-case timing, p90 worst-case, and the pair
                    # still characterises reliability — as a secondary layer.
                    p10_seconds=float(np.percentile(good, 10)),
                    p50_seconds=float(np.percentile(good, 50)),
                    p90_seconds=float(np.percentile(good, 90)),
                    samples=int(good.size),
                    walk_seconds=float(min(direct[i], live)),
                    next_jump_s=(None if cliff[i] >= MAX_TIME else float(cliff[i])),
                )
            )
        self.last_unreachable = unreachable
        return out

    def _cliffs(self, per_departure: np.ndarray) -> np.ndarray:
        """Seconds until each cell's journey time jumps past the threshold."""
        n_dep, n = per_departure.shape
        out = np.full(n, MAX_TIME, dtype=np.int64)
        if n_dep < 2:
            return out
        step = self.profile_window_minutes * 60.0 / self.profile_samples
        base = per_departure[0].copy()
        for k in range(1, n_dep):
            cur = per_departure[k]
            jumped = (
                (cur < MAX_TIME) & (base < MAX_TIME)
                & (cur - base > self.cliff_threshold_s) & (out == MAX_TIME)
            )
            out[jumped] = int(step * k)
            base = np.minimum(base, cur)
        return out

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "gtfs_path": self.gtfs_path,
            "walk_source": self.walk_source_name,
            "osm_pbf": self.osm_pbf,
            "walk_caps_s": {"access": self.access_walk_s,
                            "transfer": self.transfer_walk_s,
                            "egress": self.egress_walk_s,
                            "direct": self.direct_walk_s},
            "departure": "now (exact); "
                         f"{self.profile_samples} samples over "
                         f"{self.profile_window_minutes:.0f} min locate the cliff",
            "profile_samples": self.profile_samples,
            "cliff_threshold_s": self.cliff_threshold_s,
            "max_rounds": self.max_rounds,
            "realtime": self.realtime,
            **{f"stat_{k}": v for k, v in self.stats.items()},
        })
        return d
