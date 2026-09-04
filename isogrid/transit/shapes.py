"""Route geometry and colours, straight from the GTFS zip.

No API key and no network: `shapes.txt` and `routes.txt` are already in the
feed downloaded for RAPTOR. That also keeps the drawn lines consistent with the
timetable being routed on, which a separate API fetch would not guarantee.

The work here is reduction. MBTA's `shapes.txt` carries every variant of every
bus route across eastern Massachusetts — millions of points, nearly all of them
irrelevant to a 5 km grid. Three passes cut it down:

1. keep only shapes whose trips belong to a route touching the bounding box,
2. keep the longest few shapes per route and direction, since a route has many
   near-identical variants differing only at the ends,
3. simplify each polyline, because at map zoom the difference between 400 and
   40 points is invisible.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

#: GTFS route_type -> a rendering class. Bus is by far the most numerous and
#: should recede; rapid transit should dominate.
ROUTE_KIND = {
    "0": "light_rail", "1": "subway", "2": "commuter_rail",
    "3": "bus", "4": "ferry",
}

#: Fallbacks for feeds that omit route_color. MBTA populates it, but a missing
#: colour should not render as an invisible black line.
DEFAULT_COLOR = {
    "light_rail": "00843D", "subway": "494F5C", "commuter_rail": "80276C",
    "bus": "FFC72C", "ferry": "008EAA",
}


def _rows(zf: zipfile.ZipFile, name: str):
    try:
        raw = zf.read(name)
    except KeyError:
        return
    yield from csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))


def _simplify(pts: list[tuple[float, float]], tol_deg: float) -> list:
    """Douglas-Peucker. Perpendicular distance in degrees is fine here — the
    only consumer is a renderer, and the error is far below a pixel."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        ax, ay = pts[lo]
        bx, by = pts[hi]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy) or 1e-12
        worst, wi = 0.0, -1
        for i in range(lo + 1, hi):
            px, py = pts[i]
            d = abs(dy * px - dx * py + bx * ay - by * ax) / norm
            if d > worst:
                worst, wi = d, i
        if wi >= 0 and worst > tol_deg:
            keep[wi] = True
            stack.append((lo, wi))
            stack.append((wi, hi))
    return [p for p, k in zip(pts, keep) if k]


def build_station_geojson(
    gtfs_path: str,
    center_lat: float,
    center_lon: float,
    radius_km: float = 6.0,
    kinds: Optional[set] = None,
    cache: Optional[str] = None,
    rebuild: bool = False,
) -> dict:
    """Rapid-transit stations within the radius, as GeoJSON points.

    Two reductions matter. First, only stops served by routes of the requested
    kinds — the feed has 7,769 stops and the overwhelming majority are bus
    stops 200 m apart, which would bury the map. Second, platforms are
    collapsed onto their `parent_station`, so Park Street is one marker rather
    than six stacked on top of each other.

    The route join goes through one representative trip per route rather than
    every trip, which keeps the `stop_times` scan cheap.
    """
    if cache and Path(cache).is_file() and not rebuild:
        return json.loads(Path(cache).read_text())

    kinds = kinds or {"subway", "light_rail"}
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(center_lat)))

    with zipfile.ZipFile(gtfs_path) as zf:
        want_routes = {}
        for r in _rows(zf, "routes.txt"):
            kind = ROUTE_KIND.get((r.get("route_type") or "").strip(), "bus")
            if kind in kinds:
                want_routes[r["route_id"]] = {
                    "kind": kind,
                    "name": (r.get("route_short_name")
                             or r.get("route_long_name") or r["route_id"]),
                    "color": "#" + ((r.get("route_color") or "").strip()
                                    or DEFAULT_COLOR[kind]),
                }
        # One representative trip per route and direction is enough to
        # enumerate the stops a route serves.
        rep: dict[str, str] = {}
        for r in _rows(zf, "trips.txt"):
            rid = r["route_id"]
            if rid not in want_routes:
                continue
            key = f"{rid}:{r.get('direction_id', '0')}"
            rep.setdefault(key, r["trip_id"])
        trip_route = {tid: k.split(":")[0] for k, tid in rep.items()}

        stop_routes: dict[str, set] = defaultdict(set)
        for r in _rows(zf, "stop_times.txt"):
            rid = trip_route.get(r["trip_id"])
            if rid:
                stop_routes[r["stop_id"]].add(rid)

        stops = {}
        for r in _rows(zf, "stops.txt"):
            stops[r["stop_id"]] = r

    # Collapse platforms onto their parent station.
    grouped: dict[str, dict] = {}
    for sid, routes in stop_routes.items():
        s = stops.get(sid)
        if not s:
            continue
        parent = (s.get("parent_station") or "").strip()
        key = parent or sid
        rec = stops.get(key, s)
        try:
            lat, lon = float(rec["stop_lat"]), float(rec["stop_lon"])
        except (KeyError, ValueError):
            continue
        if abs(lat - center_lat) > dlat or abs(lon - center_lon) > dlon:
            continue
        g = grouped.setdefault(key, {
            "name": rec.get("stop_name", key), "lat": lat, "lon": lon,
            "routes": set(),
        })
        g["routes"] |= routes

    feats = []
    for key, g in sorted(grouped.items()):
        names = sorted(want_routes[r]["name"] for r in g["routes"]
                       if r in want_routes)
        kind_set = {want_routes[r]["kind"] for r in g["routes"]
                    if r in want_routes}
        feats.append({
            "type": "Feature", "id": key,
            "properties": {
                "name": g["name"], "routes": ", ".join(names),
                "n_routes": len(names),
                # A station on several lines is an interchange and worth
                # showing slightly larger.
                "kind": "subway" if "subway" in kind_set else "light_rail",
            },
            "geometry": {"type": "Point",
                         "coordinates": [round(g["lon"], 6), round(g["lat"], 6)]},
        })
    out = {"type": "FeatureCollection", "features": feats,
           "metadata": {"stations": len(feats)}}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(out))
    return out


def build_route_geojson(
    gtfs_path: str,
    center_lat: float,
    center_lon: float,
    radius_km: float = 8.0,
    per_route_per_direction: int = 1,
    simplify_m: float = 15.0,
    cache: Optional[str] = None,
    rebuild: bool = False,
) -> dict:
    """Route polylines within `radius_km`, as a GeoJSON FeatureCollection."""
    if cache and Path(cache).is_file() and not rebuild:
        return json.loads(Path(cache).read_text())

    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(center_lat)))
    lat_lo, lat_hi = center_lat - dlat, center_lat + dlat
    lon_lo, lon_hi = center_lon - dlon, center_lon + dlon

    with zipfile.ZipFile(gtfs_path) as zf:
        routes = {}
        for r in _rows(zf, "routes.txt"):
            kind = ROUTE_KIND.get((r.get("route_type") or "").strip(), "bus")
            routes[r["route_id"]] = {
                "name": (r.get("route_short_name") or r.get("route_long_name")
                         or r["route_id"]),
                "long_name": r.get("route_long_name", ""),
                "kind": kind,
                "color": "#" + ((r.get("route_color") or "").strip()
                                or DEFAULT_COLOR[kind]),
                "text_color": "#" + ((r.get("route_text_color") or "").strip() or "FFFFFF"),
                "sort": int(r.get("route_sort_order") or 0),
            }

        # shape_id -> (route_id, direction). A shape is shared by many trips.
        shape_route: dict[str, tuple[str, str]] = {}
        for r in _rows(zf, "trips.txt"):
            sid = (r.get("shape_id") or "").strip()
            if sid and sid not in shape_route:
                shape_route[sid] = (r["route_id"], r.get("direction_id", "0"))

        pts: dict[str, list] = defaultdict(list)
        for r in _rows(zf, "shapes.txt"):
            sid = r["shape_id"]
            if sid not in shape_route:
                continue
            pts[sid].append((
                int(r["shape_pt_sequence"]),
                float(r["shape_pt_lon"]), float(r["shape_pt_lat"]),
            ))

    # Keep the longest few shapes per (route, direction): a route has many
    # variants that differ only in where they short-turn.
    by_route: dict[tuple[str, str], list] = defaultdict(list)
    for sid, raw in pts.items():
        raw.sort()
        coords = [(lon, lat) for _, lon, lat in raw]
        if not any(lat_lo <= la <= lat_hi and lon_lo <= lo <= lon_hi
                   for lo, la in coords):
            continue
        by_route[shape_route[sid]].append((len(coords), sid, coords))

    tol = simplify_m / 111_320.0
    features = []
    for (route_id, direction), variants in by_route.items():
        meta = routes.get(route_id)
        if meta is None:
            continue
        variants.sort(reverse=True)
        for _n, sid, coords in variants[:per_route_per_direction]:
            simple = _simplify(coords, tol)
            features.append({
                "type": "Feature",
                "id": sid,
                "properties": {
                    "route_id": route_id, "direction": direction,
                    "name": meta["name"], "long_name": meta["long_name"],
                    "kind": meta["kind"], "color": meta["color"],
                    "sort": meta["sort"],
                },
                "geometry": {"type": "LineString",
                             "coordinates": [[round(x, 5), round(y, 5)]
                                             for x, y in simple]},
            })

    # Draw order: bus underneath, rapid transit on top.
    order = {"bus": 0, "ferry": 1, "commuter_rail": 2, "light_rail": 3, "subway": 4}
    features.sort(key=lambda f: order.get(f["properties"]["kind"], 0))
    out = {"type": "FeatureCollection", "features": features,
           "metadata": {"routes": len(by_route),
                        "points": sum(len(f["geometry"]["coordinates"])
                                      for f in features)}}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(out))
    return out
