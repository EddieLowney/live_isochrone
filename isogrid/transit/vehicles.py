"""Placing vehicles on their route geometry, so they move along the track.

Interpolating between two GPS fixes in latitude and longitude cuts corners: a
Green Line car rounding Kenmore would visibly leave the rails. The fix is to
interpolate in **arc length** along the route polyline instead. Project each
fix to a scalar position `s` on the shape, interpolate `s`, then convert back
to a coordinate by walking the polyline. Geometry never enters the
interpolation, so the path is exact however sharp the curve.

Finding the shape is the other half of the problem, and the harder half in
practice. A trip_id resolves through `trips.txt` when the trip was scheduled,
but MBTA publishes most Green Line runs as ADDED trips carrying an id that
exists nowhere in the static feed. Route plus direction_id always resolves, so
that is the fallback, and it is what keeps two thirds of live rapid transit on
the rails rather than on chords.

The projection itself has one real trap. Nearest-point-on-polyline is
ambiguous wherever a route touches itself — inbound beside outbound, a loop, a
short-turn — and a naive search can snap a vehicle backwards or onto the
parallel track, which reads as a teleport. Two guards here: the search is
windowed around the vehicle's previous `s`, sized by how far it could have
travelled, and a match landing implausibly far from the shape is rejected
rather than used.

Beyond that, correction is deliberately thin. Arc length is monotonic, so one
clamp against the last accepted position absorbs both bad re-matches and GPS
noise, and every leg is built to start exactly where the previous one ended.

The browser is sent only the short stretch of polyline between the two known
fixes. It never needs the shapes, and cannot disagree with the server about
where the track goes.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EARTH_M = 6371008.8


def _local_scale(lat: float) -> tuple[float, float]:
    """Metres per degree at a latitude. Good to a fraction of a percent over a
    metro area, and avoids a projection dependency."""
    return 111_320.0 * math.cos(math.radians(lat)), 110_540.0


@dataclass
class Shape:
    """One route polyline with cumulative arc length in metres."""

    lat: np.ndarray
    lon: np.ndarray
    cum: np.ndarray          # cum[i] = metres from the start to point i

    @property
    def length_m(self) -> float:
        return float(self.cum[-1]) if len(self.cum) else 0.0

    def project(self, lat: float, lon: float,
                hint_s: Optional[float] = None,
                window_m: float = 1500.0) -> tuple[float, float]:
        """Nearest position along the shape. Returns (arc_length, offset_m).

        `hint_s` restricts the search to a window around the previous position,
        which is what stops a vehicle jumping to the opposite track where a
        route runs beside itself.
        """
        lo, hi = 0, len(self.lat) - 1
        if hint_s is not None and self.length_m > 0:
            lo = int(np.searchsorted(self.cum, hint_s - window_m))
            hi = int(np.searchsorted(self.cum, hint_s + window_m))
            lo, hi = max(0, lo - 1), min(len(self.lat) - 1, hi + 1)
            if hi - lo < 2:
                lo, hi = 0, len(self.lat) - 1

        mx, my = _local_scale(lat)
        px, py = lon * mx, lat * my
        ax = self.lon[lo:hi] * mx
        ay = self.lat[lo:hi] * my
        bx = self.lon[lo + 1:hi + 1] * mx
        by = self.lat[lo + 1:hi + 1] * my
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        denom[denom == 0] = 1e-9
        # Parameter of the closest point on each segment, clamped to it.
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / denom, 0.0, 1.0)
        cx, cy = ax + t * dx, ay + t * dy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        k = int(np.argmin(d2))
        seg = lo + k
        s = float(self.cum[seg] + t[k] * (self.cum[seg + 1] - self.cum[seg]))
        return s, float(math.sqrt(d2[k]))

    def slice(self, s0: float, s1: float, max_points: int = 64) -> list:
        """Polyline points from s0 to s1, endpoints included.

        A shape may be drawn against the direction of travel, so s1 < s0 is
        legitimate and returns the stretch in travel order. The caller is
        responsible for having already rejected regressions caused by noise —
        reversing those would animate the vehicle backwards along the track.
        """
        rev = s1 < s0
        a, b = (s1, s0) if rev else (s0, s1)
        # Interior vertices must lie strictly inside (a, b). `searchsorted`
        # returns the first index at or past a value, so including index j
        # overshoots b by up to one vertex spacing — the path then reverses to
        # reach the true endpoint. Played back, that is a visible bounce at the
        # end of every leg.
        i = int(np.searchsorted(self.cum, a, side="right"))
        j = int(np.searchsorted(self.cum, b, side="left"))
        pts = [self._at(a)]
        mid = list(range(max(i, 0), max(min(j, len(self.cum)), max(i, 0))))
        if len(mid) > max_points:
            step = max(1, len(mid) // max_points)
            mid = mid[::step]
        pts += [[float(self.lon[m]), float(self.lat[m])] for m in mid]
        pts.append(self._at(b))
        if rev:
            pts.reverse()
        # Drop consecutive duplicates; they add payload and confuse arc-length
        # interpolation in the browser.
        out = [pts[0]]
        for p in pts[1:]:
            if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
                out.append(p)
        return out

    def _at(self, s: float) -> list:
        s = max(0.0, min(s, self.length_m))
        i = int(np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.cum) - 2))
        span = self.cum[i + 1] - self.cum[i]
        f = 0.0 if span <= 0 else (s - self.cum[i]) / span
        return [float(self.lon[i] + f * (self.lon[i + 1] - self.lon[i])),
                float(self.lat[i] + f * (self.lat[i + 1] - self.lat[i]))]


class ShapeIndex:
    """Route geometry, filtered to the area and keyed for vehicle lookup."""

    #: Candidate shapes kept per route and direction. One is the normal case;
    #: the cap only bounds the cost of the initial projection.
    CANDIDATE_LIMIT = 4

    def __init__(self, shapes: dict, trip_shape: dict,
                 route_dir_shapes: Optional[dict] = None):
        self.shapes = shapes
        self.trip_shape = trip_shape
        #: (route_id, direction_id) -> candidate shape_ids, longest first.
        #:
        #: `trip_shape` alone is not enough. MBTA publishes most Green Line
        #: runs — and some Red and Orange — as ADDED trips, whose trip_id does
        #: not exist in the static feed: measured live, 45 of 72 rapid-transit
        #: vehicles, including every Green Line car. Resolved through trip_id
        #: only, those all fall back to straight chords between fixes, which is
        #: precisely the failure this module opens by describing. Route and
        #: direction are always present, so they are the handle that works.
        self.route_dir_shapes = route_dir_shapes or {}

    @classmethod
    def build(cls, gtfs_path: str, center_lat: float, center_lon: float,
              radius_km: float = 12.0, cache: Optional[str] = None,
              rebuild: bool = False) -> "ShapeIndex":
        if cache and Path(cache).is_file() and not rebuild:
            z = np.load(cache, allow_pickle=True)
            # A cache written before the route/direction fallback existed has
            # no such table. Rebuilding is cheaper than silently running
            # without it and animating the Green Line on chords.
            if "route_dir" in z:
                shapes = {}
                for sid, lat, lon, cum in zip(z["ids"], z["lats"],
                                              z["lons"], z["cums"]):
                    shapes[str(sid)] = Shape(lat, lon, cum)
                return cls(shapes, dict(z["trip_shape"].item()),
                           dict(z["route_dir"].item()))

        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * math.cos(math.radians(center_lat)))
        lat_lo, lat_hi = center_lat - dlat, center_lat + dlat
        lon_lo, lon_hi = center_lon - dlon, center_lon + dlon

        raw: dict = {}
        trip_shape: dict = {}
        route_dir: dict = {}
        with zipfile.ZipFile(gtfs_path) as zf:
            for r in csv.DictReader(io.StringIO(
                    zf.read("trips.txt").decode("utf-8-sig"))):
                sid = (r.get("shape_id") or "").strip()
                if not sid:
                    continue
                trip_shape[r["trip_id"]] = sid
                route_dir.setdefault(
                    ((r.get("route_id") or "").strip(),
                     (r.get("direction_id") or "").strip()), set()).add(sid)
            for r in csv.DictReader(io.StringIO(
                    zf.read("shapes.txt").decode("utf-8-sig"))):
                raw.setdefault(r["shape_id"], []).append((
                    int(r["shape_pt_sequence"]),
                    float(r["shape_pt_lat"]), float(r["shape_pt_lon"])))

        shapes: dict = {}
        for sid, pts in raw.items():
            pts.sort()
            lat = np.array([p[1] for p in pts])
            lon = np.array([p[2] for p in pts])
            # A shape matters if any part of it is in range; a Commuter Rail
            # line reaching Worcester still needs its inner stretch.
            if not np.any((lat >= lat_lo) & (lat <= lat_hi)
                          & (lon >= lon_lo) & (lon <= lon_hi)):
                continue
            mx, my = _local_scale(float(lat.mean()))
            seg = np.hypot(np.diff(lon) * mx, np.diff(lat) * my)
            shapes[sid] = Shape(lat, lon,
                                np.concatenate([[0.0], np.cumsum(seg)]))

        used = set(shapes)
        trip_shape = {t: s for t, s in trip_shape.items() if s in used}
        route_dir_shapes: dict = {}
        for key, sids in route_dir.items():
            pool = [s for s in sids if s in used]
            if not pool:
                continue
            # A `canonical-` shape is the agency's own full-length
            # representative for a pattern, so it is preferred over the
            # short-turns and diversions sharing the key. The Red Line keeps
            # two of them because Ashmont and Braintree really are different
            # track; `candidates` separates those by proximity.
            canon = [s for s in pool if s.startswith("canonical-")]
            pool = canon or pool
            route_dir_shapes[key] = sorted(
                pool, key=lambda s: -shapes[s].length_m)[:cls.CANDIDATE_LIMIT]
        idx = cls(shapes, trip_shape, route_dir_shapes)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            ids = list(shapes)
            np.savez_compressed(
                cache, ids=np.array(ids, dtype=object),
                lats=np.array([shapes[i].lat for i in ids], dtype=object),
                lons=np.array([shapes[i].lon for i in ids], dtype=object),
                cums=np.array([shapes[i].cum for i in ids], dtype=object),
                trip_shape=np.array(trip_shape, dtype=object),
                route_dir=np.array(route_dir_shapes, dtype=object))
        return idx

    def for_trip(self, trip_id: Optional[str]) -> Optional[Shape]:
        if not trip_id:
            return None
        sid = self.trip_shape.get(trip_id)
        return self.shapes.get(sid) if sid else None

    def candidates(self, v) -> list:
        """Shapes the vehicle might be running on, as (shape_id, Shape).

        A scheduled trip names its shape outright, so the list has one entry.
        An ADDED trip names none, and the route's canonical geometry for the
        direction it reports is the answer — normally one shape, two where a
        line genuinely branches.
        """
        sid = self.trip_shape.get(v.trip_id or "")
        if sid and sid in self.shapes:
            return [(sid, self.shapes[sid])]
        d = getattr(v, "direction_id", None)
        key = ((v.route_id or "").strip(), "" if d is None else str(d))
        return [(s, self.shapes[s]) for s in self.route_dir_shapes.get(key, ())
                if s in self.shapes]

    def describe(self) -> dict:
        return {"shapes": len(self.shapes), "trips_mapped": len(self.trip_shape),
                "route_dirs_mapped": len(self.route_dir_shapes)}


@dataclass
class Track:
    """Where one vehicle is, in arc length along the shape it was matched to.

    Arc length only ever increases. GTFS draws each shape in the direction of
    travel, and MBTA gives inbound and outbound distinct shape_ids — checked
    against every rapid-transit shape in the feed, with stop order monotonic on
    all of them. So a fix projecting behind the last accepted one is a bad
    match, never a reversal, and clamping is the whole of the correction
    needed. There is no direction sign to infer and no backtrack budget to
    spend.
    """

    #: The trip identity the position is measured against: (trip, route,
    #: direction). Arc length means nothing across a change of it.
    key: Optional[tuple] = None
    #: The shape chosen for this trip, then held. Re-choosing every fix lets a
    #: Red Line train flip between the Ashmont and Braintree branches.
    shape_id: Optional[str] = None
    last_t: Optional[float] = None
    last_s: Optional[float] = None
    #: Where the last published leg ended. Every new leg starts here, which is
    #: what makes playback continuous.
    last_pt: Optional[list] = None
    #: Completed legs, each built exactly once when its closing fix was
    #: accepted and never recomputed. Rebuilding legs every poll is what let
    #: the *past* change underneath a vehicle midway through playing it — the
    #: marker would be re-placed mid-leg, which is seen as a bounce.
    legs: deque = field(default_factory=lambda: deque(maxlen=6))
    off_shape_m: float = 0.0
    off_shape_run: int = 0

    def reset(self) -> None:
        """Forget the position without forgetting the vehicle."""
        self.shape_id = None
        self.last_t = self.last_s = self.last_pt = None
        self.off_shape_run = 0
        self.legs.clear()


class VehicleTracker:
    """Turns a stream of fixes into animatable path segments.

    A leg is the stretch of track between two consecutive fixes plus the real
    time between them. The browser plays legs in order and halts at the newest,
    so it is always interpolating between two known positions rather than
    guessing past the last one.
    """

    #: Beyond this a projection is not believable — the vehicle is off its
    #: shape, usually a detour or the wrong branch. Draw a straight line rather
    #: than teleporting it onto the route. Station platforms sit up to ~85 m
    #: off their shape in this feed, so the threshold has to clear that.
    MAX_OFF_SHAPE_M = 120.0

    #: Forward movement below this is held rather than applied. A stationary
    #: vehicle's fixes wander by tens of metres and projecting that onto the
    #: track walks the marker up and down the rails. Combined with the
    #: monotonic clamp, one comparison covers both noise and bad re-matches.
    MIN_TRAVEL_M = 25.0

    #: Consecutive unbelievable projections before the shape choice itself is
    #: reconsidered. Guards the one case route+direction cannot settle on its
    #: own: a Red Line train first seen north of JFK, where the Ashmont and
    #: Braintree shapes coincide, and which then takes the other branch.
    OFF_SHAPE_BEFORE_RESHAPE = 3

    #: Cap on plausible rapid-transit speed, used only to size the projection
    #: search window. The Red Line's fastest stretch runs about 26 m/s.
    MAX_SPEED_MS = 35.0
    WINDOW_BASE_M = 150.0
    WINDOW_MAX_M = 2500.0

    def __init__(self, index: Optional[ShapeIndex] = None,
                 max_age_s: float = 180.0,
                 route_kinds: Optional[dict] = None,
                 keep_kinds: Optional[set] = None):
        self.index = index
        self.max_age_s = float(max_age_s)
        #: route_id -> mode name, and the set of modes to animate. Buses report
        #: every 6 s and move ~37 m between fixes, below one cell, so
        #: interpolating them costs work for no visible gain.
        self.route_kinds = route_kinds or {}
        # Config spells these with a space ("light rail"); ROUTE_KIND, which
        # produces the values compared against, spells them with an underscore
        # ("light_rail"). Unnormalised, "light rail" matched nothing, and every
        # route_type 0 — the whole Green Line, plus Mattapan — was silently
        # dropped from the map. Normalising here fixes every caller at once.
        self.keep_kinds = ({k.replace(" ", "_") for k in keep_kinds}
                           if keep_kinds else None)
        self.tracks: dict = {}
        self.stats: dict = {}

    def keep(self, v) -> bool:
        if self.keep_kinds is None:
            return True
        return self.route_kinds.get(v.route_id or "", "other") in self.keep_kinds

    def update(self, vehicles: list, now: Optional[float] = None) -> None:
        import time

        now = time.time() if now is None else now
        seen = set()
        counts = {"on_shape": 0, "off_shape": 0, "no_shape": 0, "held": 0,
                  "trip_change": 0, "reshape": 0}
        for v in vehicles:
            if not self.keep(v):
                continue
            if v.timestamp is not None and now - v.timestamp > self.max_age_s:
                continue          # a ghost: still listed, no longer reporting
            seen.add(v.id)
            tr = self.tracks.setdefault(v.id, Track())

            key = (v.trip_id, v.route_id, getattr(v, "direction_id", None))
            if tr.key != key:
                # New trip: new geometry, and nothing about the old arc length
                # carries over.
                if tr.key is not None:
                    counts["trip_change"] += 1
                tr.reset()
                tr.key = key

            # `or now` would treat a timestamp of 0 as missing.
            t = now if v.timestamp is None else v.timestamp
            if tr.last_t is not None and t <= tr.last_t:
                continue          # already seen this fix, or it arrived late

            # `_locate` may clear `last_s` when it gives up on the shape it had
            # chosen, which is why it runs before the clamp reads it.
            s, shape = self._locate(tr, v, t, counts)
            if s is not None and tr.last_s is not None \
                    and s - tr.last_s < self.MIN_TRAVEL_M:
                # Monotonic, with a deadband, in one comparison. Backwards is a
                # bad re-match — a train pulling into a platform projects
                # behind its last in-transit position routinely — and a small
                # step forward is noise on a stationary car. Hold for both.
                counts["held"] += 1
                s = tr.last_s

            end = (shape._at(s) if shape is not None and s is not None
                   else [v.lon, v.lat])
            if tr.last_t is not None and tr.last_pt is not None:
                leg = self._build_leg(tr.last_pt, end, tr.last_t, t,
                                      shape, tr.last_s, s)
                if leg is not None:
                    tr.legs.append(leg)
                    end = leg["path"][-1]
            tr.last_t, tr.last_s, tr.last_pt = t, s, end
        for vid in [k for k in self.tracks if k not in seen]:
            del self.tracks[vid]
        self.stats = counts

    def _locate(self, tr, v, t, counts) -> tuple:
        """Project the vehicle onto its shape, choosing the shape if needed.

        Returns (arc_length, shape). `arc_length` is None when no believable
        match exists, and the caller then draws a straight line.
        """
        if self.index is None:
            counts["no_shape"] += 1
            return None, None

        # Size the search to how far the vehicle could possibly have travelled.
        # One fixed window is either too tight after a long gap — fix intervals
        # here run to 54 s at p90 — or wide enough after a short one to catch
        # the opposite track where a route runs beside itself.
        window = self.WINDOW_BASE_M
        if tr.last_t is not None:
            window = min(self.WINDOW_MAX_M,
                         window + self.MAX_SPEED_MS * (t - tr.last_t))

        if tr.shape_id is not None:
            shape = self.index.shapes.get(tr.shape_id)
            if shape is not None:
                s, off = shape.project(v.lat, v.lon, hint_s=tr.last_s,
                                       window_m=window)
                tr.off_shape_m = off
                if off <= self.MAX_OFF_SHAPE_M:
                    tr.off_shape_run = 0
                    counts["on_shape"] += 1
                    return s, shape
                tr.off_shape_run += 1
                counts["off_shape"] += 1
                if tr.off_shape_run < self.OFF_SHAPE_BEFORE_RESHAPE:
                    return None, shape
                # Persistently off: the branch guess was wrong. Choose again,
                # and drop the arc length that was measured against the shape
                # being abandoned.
                tr.shape_id = None
                tr.last_s = None
                counts["reshape"] += 1

        best = None
        for sid, shape in self.index.candidates(v):
            s, off = shape.project(v.lat, v.lon)
            if best is None or off < best[2]:
                best = (sid, s, off, shape)
        if best is None:
            counts["no_shape"] += 1
            return None, None
        sid, s, off, shape = best
        tr.off_shape_m = off
        if off > self.MAX_OFF_SHAPE_M:
            counts["off_shape"] += 1
            return None, None
        tr.shape_id = sid
        tr.off_shape_run = 0
        counts["on_shape"] += 1
        return s, shape

    def _build_leg(self, start, end, ta, tb, shape, sa, sb,
                   max_points: int = 24) -> Optional[dict]:
        """One immutable leg, from where the last leg ended to the new fix.

        Built once, when its closing fix is accepted, and never touched again.
        `dur` is the real elapsed time between the fixes, so playing the leg
        over that duration moves the vehicle at the speed it actually
        travelled.

        The path *starts* at `start` — the previous leg's final point — rather
        than at a position projected afresh. That is what makes playback
        continuous. With the two derived independently, any leg that changed
        between on-track and straight-line geometry began somewhere the marker
        was not, and the browser could only jump to it. Measured against the
        live feed, 7% of leg handovers had such a gap, the worst 1.1 km.
        """
        dur = tb - ta
        if dur <= 0:
            return None
        on_track = False
        if shape is not None and sa is not None and sb is not None:
            path = shape.slice(sa, sb, max_points=max_points)
            on_track = True
            if len(path) < 2:
                # Held position: the vehicle is dwelling. Spend the duration
                # standing still rather than dropping the leg, which would let
                # playback run ahead of the feed.
                path = [start, start]
        else:
            path = [start, end]
        return {
            # Stable per vehicle, so the browser can tell a leg it has already
            # queued from a newly published one.
            "id": round(tb, 1),
            "dur": round(dur, 2),
            "path": [[round(x, 6), round(y, 6)] for x, y in path],
            "on_track": on_track,
        }

    def segments(self, vehicles: list, now: Optional[float] = None) -> list:
        """Publish each vehicle's completed legs, oldest first.

        The server does not decide where a vehicle is drawn. It publishes an
        ordered list of legs; the browser plays them one after another and
        halts when it runs out. Nothing here depends on wall clock, so nothing
        the server sends can reposition a vehicle mid-leg.
        """
        import time

        now = time.time() if now is None else now
        out = []
        for v in vehicles:
            tr = self.tracks.get(v.id)
            if tr is None:
                continue
            # Publish even with no legs yet. Dropping the vehicle from the
            # response makes the browser discard its playback state, and on
            # the next poll it re-queues every leg from scratch — replaying
            # ground it has already covered. A trip change empties `legs`, and
            # trip changes happen at terminals, which is exactly where the
            # replay was being seen.
            feat = v.as_feature(now)
            legs = list(tr.legs)
            feat["properties"].update({
                "on_track": bool(legs) and all(l["on_track"] for l in legs),
                "off_shape_m": round(tr.off_shape_m, 1),
                "legs": len(legs),
            })
            # With no legs yet, fall back to the raw fix: the browser holds
            # there until something is available to play.
            feat["geometry"] = {
                "type": "Point",
                "coordinates": tr.last_pt or [v.lon, v.lat],
            }
            feat["legs"] = legs
            out.append(feat)
        return out

    def describe(self) -> dict:
        return {"tracked": len(self.tracks), **self.stats,
                **(self.index.describe() if self.index else {})}
