"""GTFS-Realtime, read without a protobuf dependency.

The wire format is self-describing enough that a 40-line reader handles it:
every field is a varint key encoding (field_number << 3 | wire_type) followed
by a payload whose length is implied by the wire type. Field *names* are not on
the wire, so the numbers below come from the GTFS-Realtime spec.

Avoiding `gtfs-realtime-bindings` is deliberate. It pulls in a protobuf runtime
whose major versions break each other, and this is the one component that must
keep working unattended for weeks.

Only TripUpdates are consumed. VehiclePositions do not affect routing, and
Alerts would need service-level modelling that RAPTOR has no representation
for — a cancelled trip arrives here as a SKIPPED relationship anyway.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Field numbers, from the GTFS-Realtime protobuf definition.
F_ENTITY = 2            # FeedMessage.entity
F_TRIP_UPDATE = 3       # FeedEntity.trip_update
F_VEHICLE = 4           # FeedEntity.vehicle -> VehiclePosition
# VehiclePosition
F_VP_TRIP = 1
F_VP_POSITION = 2
F_VP_STOP_SEQ = 3
F_VP_STATUS = 4         # 0 INCOMING_AT, 1 STOPPED_AT, 2 IN_TRANSIT_TO
F_VP_TIMESTAMP = 5
F_VP_STOP_ID = 7
F_VP_VEHICLE = 8        # VehicleDescriptor
F_VP_OCCUPANCY = 9
# Position — note these are 32-bit floats on the wire, not varints
F_POS_LAT = 1
F_POS_LON = 2
F_POS_BEARING = 3
F_POS_SPEED = 5
F_VD_ID = 1
F_VD_LABEL = 2
F_TRIP_ROUTE_ID = 5     # TripDescriptor.route_id
F_TRIP_DIRECTION = 6    # TripDescriptor.direction_id
F_TRIP = 1              # TripUpdate.trip
F_STOP_TIME_UPDATE = 2  # TripUpdate.stop_time_update
F_TU_DELAY = 5          # TripUpdate.delay
F_TRIP_ID = 1           # TripDescriptor.trip_id
F_SCHEDULE_REL = 4      # TripDescriptor.schedule_relationship
F_STU_STOP_SEQ = 1      # StopTimeUpdate.stop_sequence
F_STU_ARRIVAL = 2       # StopTimeUpdate.arrival
F_STU_DEPARTURE = 3     # StopTimeUpdate.departure
F_STU_STOP_ID = 4       # StopTimeUpdate.stop_id
F_STU_SCHEDULE_REL = 5  # StopTimeUpdate.schedule_relationship
F_EVENT_DELAY = 1       # StopTimeEvent.delay
F_EVENT_TIME = 2        # StopTimeEvent.time — absolute epoch seconds

SCHEDULE_CANCELED = 3   # TripDescriptor.ScheduleRelationship.CANCELED
STOP_SKIPPED = 1        # StopTimeUpdate.ScheduleRelationship.SKIPPED


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def parse_message(buf: bytes) -> dict[int, list[Any]]:
    """One protobuf message to {field_number: [values]}.

    Length-delimited payloads stay as raw bytes; the caller decides whether a
    given field is a nested message or a string.
    """
    out: dict[int, list[Any]] = {}
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fnum, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i:i + ln], i + ln
        elif wire == 5:
            val, i = buf[i:i + 4], i + 4
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        else:
            break  # groups: deprecated, and absent from GTFS-RT
        out.setdefault(fnum, []).append(val)
    return out


VEHICLE_STATUS = {0: "incoming", 1: "stopped", 2: "in_transit"}
OCCUPANCY = {
    0: "empty", 1: "many_seats", 2: "few_seats", 3: "standing_only",
    4: "crushed", 5: "full", 6: "not_accepting",
}


def _f32(raw) -> Optional[float]:
    """Protobuf `float` fields arrive as four little-endian bytes."""
    import struct

    if not isinstance(raw, (bytes, bytearray)) or len(raw) != 4:
        return None
    return float(struct.unpack("<f", raw)[0])


def _first(d: dict, k: int, default=None):
    v = d.get(k)
    return v[0] if v else default


@dataclass
class StopPrediction:
    """One StopTimeUpdate. Either `delay` or `time` is populated, rarely both."""

    stop_id: Optional[str] = None
    stop_sequence: Optional[int] = None
    time: Optional[int] = None     # absolute epoch seconds
    delay: Optional[int] = None    # seconds late, signed
    #: The vehicle will not serve this stop. Carries no time or delay, which is
    #: why a reader keyed on those alone drops it and silently lets passengers
    #: board a train that is not stopping.
    skipped: bool = False


@dataclass
class TripPrediction:
    trip_id: str
    canceled: bool = False
    trip_delay: Optional[int] = None
    stops: list = field(default_factory=list)


@dataclass
class FeedPredictions:
    """What the feed literally said, before any schedule join.

    Kept separate from `TripDelays` because producers differ in what they
    publish. Some emit a `delay` offset; MBTA emits absolute predicted `time`
    values, which mean nothing until subtracted from the scheduled time. Doing
    the join here rather than in the parser keeps the wire format and the
    timetable lookup independently testable.
    """

    trips: list = field(default_factory=list)
    entities: int = 0
    fetched_at: Optional[float] = None

    def resolve(self, data=None, service_epoch: Optional[float] = None) -> "TripDelays":
        """Collapse predictions into per-trip and per-stop-time delays.

        `data` and `service_epoch` are needed only for absolute-time feeds.
        Without them, only trips carrying an explicit delay resolve — which is
        exactly the silent-timetable failure this signature exists to prevent.
        """
        out = TripDelays(entities=self.entities, fetched_at=self.fetched_at)
        stop_index = None
        if data is not None:
            stop_index = {str(s): i for i, s in enumerate(data.stop_ids)}
            trip_index = {str(t): i for i, t in enumerate(data.trip_ids)}
            out.per_stoptime = np.zeros(len(data.trip_arr), dtype=np.int32)
            out.skipped = np.zeros(len(data.trip_arr), dtype=bool)

        for tp in self.trips:
            if tp.canceled:
                out.canceled.add(tp.trip_id)
                continue
            if tp.trip_delay is not None:
                out.delays[tp.trip_id] = tp.trip_delay
                out.from_delay_field += 1
                continue
            if data is None or service_epoch is None:
                out.unresolved += 1
                continue
            ti = trip_index.get(tp.trip_id)
            if ti is None:
                out.unknown_trip += 1
                continue
            if not any(s.time is not None or s.delay is not None for s in tp.stops):
                # No timing information, but skipped stops still matter.
                _resolve_trip(tp, ti, data, stop_index, service_epoch,
                              out.per_stoptime, out.skipped, timing=False)
                out.unresolved += 1
                continue
            resolved = _resolve_trip(tp, ti, data, stop_index, service_epoch,
                                     out.per_stoptime, out.skipped)
            if resolved is None:
                out.unresolved += 1
            else:
                out.delays[tp.trip_id] = resolved
                out.from_time_field += 1
        out.skipped_stops = int(out.skipped.sum()) if out.skipped is not None else 0
        return out


def _resolve_trip(tp, ti, data, stop_index, service_epoch, per_stoptime,
                  skipped=None, timing=True):
    """Predicted absolute times minus scheduled times, per stop.

    GTFS-RT semantics say a delay applies to every subsequent stop until
    superseded, so the per-position values are forward-filled. The scalar
    returned is the median of the resolved stops, which is robust to a single
    wild prediction while still tracking a trip that is genuinely running late.
    """
    route = int(data.trip_route[ti])
    lo, hi = int(data.route_stop_ptr[route]), int(data.route_stop_ptr[route + 1])
    pattern = list(data.route_stops[lo:hi])
    base = int(data.trip_time_ptr[ti])

    found: list[tuple[int, int]] = []
    for sp in tp.stops:
        pos = None
        if sp.stop_id is not None:
            si = stop_index.get(sp.stop_id)
            if si is not None and si in pattern:
                pos = pattern.index(si)
        if pos is None and sp.stop_sequence is not None:
            # Fall back to sequence, 1-based in every feed seen so far.
            cand = sp.stop_sequence - 1
            if 0 <= cand < len(pattern):
                pos = cand
        if pos is None:
            continue
        if sp.skipped and skipped is not None:
            skipped[base + pos] = True
            continue
        if not timing:
            continue
        if sp.delay is not None:
            found.append((pos, sp.delay))
        elif sp.time is not None:
            sched = service_epoch + int(data.trip_dep[base + pos])
            found.append((pos, int(sp.time - sched)))
    if not found or not timing:
        return None

    found.sort()
    cur = found[0][1]
    k = 0
    for pos in range(len(pattern)):
        while k < len(found) and found[k][0] <= pos:
            cur = found[k][1]
            k += 1
        per_stoptime[base + pos] = cur
    return int(np.median([d for _, d in found]))


@dataclass
class TripDelays:
    """Per-trip delay in seconds, keyed by GTFS trip_id."""

    delays: dict[str, int] = field(default_factory=dict)
    canceled: set[str] = field(default_factory=set)
    entities: int = 0
    fetched_at: Optional[float] = None
    #: Delay per (trip, stop position), flat and parallel to GtfsData.trip_arr.
    #: More faithful than one scalar per trip, since delay grows along a run.
    per_stoptime: Optional[np.ndarray] = None
    #: Stop-times the vehicle will not serve, parallel to GtfsData.trip_arr.
    skipped: Optional[np.ndarray] = None
    skipped_stops: int = 0
    from_delay_field: int = 0
    from_time_field: int = 0
    unresolved: int = 0
    unknown_trip: int = 0

    def as_array(self, trip_ids: np.ndarray) -> np.ndarray:
        """Align to a GtfsData trip order. Cancelled trips get a delay large
        enough that no search will ever board them — simpler and safer than
        deleting rows from the timetable."""
        out = np.zeros(len(trip_ids), dtype=np.int32)
        for i, tid in enumerate(trip_ids):
            t = str(tid)
            if t in self.canceled:
                out[i] = 10**7
            else:
                out[i] = self.delays.get(t, 0)
        return out

    def describe(self) -> dict:
        vals = list(self.delays.values())
        return {
            "entities": self.entities,
            "trips_with_delay": len(vals),
            "canceled": len(self.canceled),
            "median_delay_s": int(np.median(vals)) if vals else 0,
            "p90_delay_s": int(np.percentile(vals, 90)) if vals else 0,
            "from_delay_field": self.from_delay_field,
            "from_time_field": self.from_time_field,
            "unresolved": self.unresolved,
            "unknown_trip": self.unknown_trip,
            "per_stoptime": self.per_stoptime is not None,
            "skipped_stops": self.skipped_stops,
        }


def parse_trip_updates(buf: bytes) -> FeedPredictions:
    """Read a TripUpdates feed into raw predictions.

    No interpretation happens here beyond the wire format: absolute times stay
    absolute, and turning them into delays is `FeedPredictions.resolve`'s job.
    """
    out = FeedPredictions()
    feed = parse_message(buf)
    for ent_bytes in feed.get(F_ENTITY, []):
        out.entities += 1
        ent = parse_message(ent_bytes)
        tu_bytes = _first(ent, F_TRIP_UPDATE)
        if tu_bytes is None:
            continue
        tu = parse_message(tu_bytes)
        trip_bytes = _first(tu, F_TRIP)
        if trip_bytes is None:
            continue
        trip = parse_message(trip_bytes)
        tid_raw = _first(trip, F_TRIP_ID)
        if tid_raw is None:
            continue
        tid = tid_raw.decode("utf-8", "replace")

        tp = TripPrediction(trip_id=tid)
        if _first(trip, F_SCHEDULE_REL, 0) == SCHEDULE_CANCELED:
            tp.canceled = True
            out.trips.append(tp)
            continue

        delay = _first(tu, F_TU_DELAY)
        if delay is not None:
            tp.trip_delay = _signed(delay)

        for stu_bytes in tu.get(F_STOP_TIME_UPDATE, []):
            stu = parse_message(stu_bytes)
            sid = _first(stu, F_STU_STOP_ID)
            sp = StopPrediction(
                stop_id=sid.decode("utf-8", "replace") if sid else None,
                stop_sequence=_first(stu, F_STU_STOP_SEQ),
                skipped=_first(stu, F_STU_SCHEDULE_REL, 0) == STOP_SKIPPED,
            )
            # Departure is what a boarding passenger cares about; fall back to
            # arrival for the final stop, which has no departure.
            ev = _first(stu, F_STU_DEPARTURE) or _first(stu, F_STU_ARRIVAL)
            if ev is not None:
                e = parse_message(ev)
                d = _first(e, F_EVENT_DELAY)
                tm = _first(e, F_EVENT_TIME)
                if d is not None:
                    sp.delay = _signed(d)
                if tm is not None:
                    sp.time = _signed(tm)
            if sp.delay is not None or sp.time is not None or sp.skipped:
                tp.stops.append(sp)
        out.trips.append(tp)
    return out


def _signed(v: int) -> int:
    """Protobuf int32/int64 arrive as unsigned varints; negatives wrap."""
    if v >= 2**63:
        return v - 2**64
    if 2**31 <= v < 2**63:
        return v - 2**32 if v < 2**32 else v
    return v


class RealtimeFetcher:
    """Polls a GTFS-RT feed, using conditional requests.

    At a 30 second cadence an unconditional fetch pulls a couple of megabytes
    2,880 times a day — several gigabytes from the agency for data that only
    changes when it changes. MBTA honours `If-None-Match` and
    `If-Modified-Since`, so an unchanged feed costs a round trip and a 304.
    """

    def __init__(self, url: str, timeout_s: float = 15.0):
        self.url = url
        self.timeout_s = float(timeout_s)
        self._etag: Optional[str] = None
        self._modified: Optional[str] = None
        self._last: Optional["FeedPredictions"] = None
        self.fetches = 0
        self.not_modified = 0
        self.bytes_downloaded = 0

    def fetch(self) -> "FeedPredictions":
        import time

        headers = {"Accept-Encoding": "identity"}
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._modified:
            headers["If-Modified-Since"] = self._modified
        req = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                buf = resp.read()
                self._etag = resp.headers.get("ETag") or self._etag
                self._modified = resp.headers.get("Last-Modified") or self._modified
        except urllib.error.HTTPError as exc:
            if exc.code == 304 and self._last is not None:
                self.not_modified += 1
                return self._last
            raise RuntimeError(f"GTFS-RT fetch failed for {self.url}: {exc}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"GTFS-RT fetch failed for {self.url}: {exc}") from exc

        self.fetches += 1
        self.bytes_downloaded += len(buf)
        out = parse_trip_updates(buf)
        out.fetched_at = time.time()
        self._last = out
        return out

    def describe(self) -> dict:
        total = self.fetches + self.not_modified
        return {
            "fetches": self.fetches,
            "not_modified": self.not_modified,
            "saved_pct": round(self.not_modified / total * 100, 1) if total else 0.0,
            "mb_downloaded": round(self.bytes_downloaded / 1e6, 1),
        }


def fetch_trip_updates(url: str, timeout_s: float = 15.0) -> FeedPredictions:
    """One-shot fetch. Prefer RealtimeFetcher for repeated polling."""
    return RealtimeFetcher(url, timeout_s).fetch()


@dataclass
class Vehicle:
    """One vehicle from a VehiclePositions feed."""

    id: str
    lat: float
    lon: float
    route_id: Optional[str] = None
    trip_id: Optional[str] = None
    #: Which way along the route. Needed because MBTA publishes most Green
    #: Line runs as ADDED trips whose trip_id is absent from the static feed,
    #: so route and direction are the only handle on their geometry.
    direction_id: Optional[int] = None
    label: Optional[str] = None
    bearing: Optional[float] = None
    speed_ms: Optional[float] = None
    status: Optional[str] = None
    stop_id: Optional[str] = None
    stop_sequence: Optional[int] = None
    occupancy: Optional[str] = None
    timestamp: Optional[int] = None

    def as_feature(self, now: Optional[float] = None) -> dict:
        import time

        age = None
        if self.timestamp is not None:
            age = max(0, int((time.time() if now is None else now) - self.timestamp))
        return {
            "type": "Feature",
            "id": self.id,
            "properties": {
                "id": self.id, "route_id": self.route_id, "label": self.label,
                "bearing": self.bearing, "speed_ms": self.speed_ms,
                "status": self.status, "occupancy": self.occupancy,
                "stop_id": self.stop_id, "age_s": age,
            },
            "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
        }


def parse_vehicle_positions(buf: bytes) -> list:
    """Vehicles from a GTFS-RT VehiclePositions feed.

    Same keyless endpoint as TripUpdates, so this needs no API key. Entities
    without a position are skipped rather than emitted at (0, 0), which would
    put a train in the Atlantic.
    """
    out: list = []
    for ent_bytes in parse_message(buf).get(F_ENTITY, []):
        ent = parse_message(ent_bytes)
        vp_bytes = _first(ent, F_VEHICLE)
        if vp_bytes is None:
            continue
        vp = parse_message(vp_bytes)
        pos_bytes = _first(vp, F_VP_POSITION)
        if pos_bytes is None:
            continue
        pos = parse_message(pos_bytes)
        lat, lon = _f32(_first(pos, F_POS_LAT)), _f32(_first(pos, F_POS_LON))
        if lat is None or lon is None:
            continue

        vd = parse_message(_first(vp, F_VP_VEHICLE) or b"")
        trip = parse_message(_first(vp, F_VP_TRIP) or b"")
        vid = _first(vd, F_VD_ID) or _first(ent, 1) or b""
        label = _first(vd, F_VD_LABEL)
        route = _first(trip, F_TRIP_ROUTE_ID)
        tid = _first(trip, F_TRIP_ID)
        sid = _first(vp, F_VP_STOP_ID)
        out.append(Vehicle(
            id=vid.decode("utf-8", "replace") if vid else f"v{len(out)}",
            lat=lat, lon=lon,
            route_id=route.decode("utf-8", "replace") if route else None,
            trip_id=tid.decode("utf-8", "replace") if tid else None,
            direction_id=_first(trip, F_TRIP_DIRECTION),
            label=label.decode("utf-8", "replace") if label else None,
            bearing=_f32(_first(pos, F_POS_BEARING)),
            speed_ms=_f32(_first(pos, F_POS_SPEED)),
            status=VEHICLE_STATUS.get(_first(vp, F_VP_STATUS)),
            stop_id=sid.decode("utf-8", "replace") if sid else None,
            stop_sequence=_first(vp, F_VP_STOP_SEQ),
            occupancy=OCCUPANCY.get(_first(vp, F_VP_OCCUPANCY)),
            timestamp=_first(vp, F_VP_TIMESTAMP),
        ))
    return out


class VehicleFetcher(RealtimeFetcher):
    """Conditional polling for VehiclePositions."""

    def vehicles(self) -> list:
        import time

        headers = {"Accept-Encoding": "identity"}
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._modified:
            headers["If-Modified-Since"] = self._modified
        req = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                buf = resp.read()
                self._etag = resp.headers.get("ETag") or self._etag
                self._modified = resp.headers.get("Last-Modified") or self._modified
        except urllib.error.HTTPError as exc:
            if exc.code == 304 and self._last is not None:
                self.not_modified += 1
                return self._last
            raise RuntimeError(f"VehiclePositions fetch failed: {exc}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"VehiclePositions fetch failed: {exc}") from exc
        self.fetches += 1
        self.bytes_downloaded += len(buf)
        self._last = parse_vehicle_positions(buf)
        self.fetched_at = time.time()
        return self._last


def encode_trip_updates(entries: list[tuple[str, Optional[int], bool]]) -> bytes:
    """Build a minimal TripUpdates feed. Used by tests to avoid needing the
    network or a protobuf compiler."""
    def tag(f: int, wire: int) -> bytes:
        return _uvarint((f << 3) | wire)

    def _uvarint(v: int) -> bytes:
        out = bytearray()
        while True:
            b = v & 0x7F
            v >>= 7
            out.append(b | (0x80 if v else 0))
            if not v:
                return bytes(out)

    def ld(f: int, payload: bytes) -> bytes:
        return tag(f, 2) + _uvarint(len(payload)) + payload

    body = b""
    for tid, delay, canceled in entries:
        trip = ld(F_TRIP_ID, tid.encode())
        if canceled:
            trip += tag(F_SCHEDULE_REL, 0) + _uvarint(SCHEDULE_CANCELED)
        tu = ld(F_TRIP, trip)
        if delay is not None and not canceled:
            tu += tag(F_TU_DELAY, 0) + _uvarint(delay)
        body += ld(F_ENTITY, ld(F_TRIP_UPDATE, tu))
    return body
