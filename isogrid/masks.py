"""Deciding which cells are worth spending quota on.

Two complementary approaches, because they fail in opposite directions:

* :class:`PolygonMask` uses geography you supply — a water layer, a study-area
  boundary — and knows about every excluded cell before the first request. It
  needs a data file and it can only exclude what the file describes.
* :class:`QuarantineMask` learns from the provider itself. A cell that is
  requested but never comes back is unroutable for *some* reason: water, a
  military base, a parking structure, bad geometry. It costs a few transactions
  to discover each one, needs no GIS data, and works with any provider.

Use both. The polygon mask handles the obvious cases for free; the quarantine
catches everything the polygon layer did not anticipate.

Shapely and geopandas are used when installed and are not required — the
fallback is a pure-Python ray-casting test with a bounding-box pre-filter,
which is entirely fast enough for the few hundred cells a grid contains.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .registry import Registry
from .types import Cell, Coord

MASKS: Registry = Registry("mask")

try:  # optional acceleration
    from shapely.geometry import Point as _ShapelyPoint  # type: ignore
    from shapely.geometry import shape as _shapely_shape  # type: ignore
    from shapely.strtree import STRtree as _STRtree  # type: ignore

    HAVE_SHAPELY = True
except Exception:  # pragma: no cover
    HAVE_SHAPELY = False


# -- geometry helpers -----------------------------------------------------

def point_in_ring(lon: float, lat: float, ring: Sequence[Sequence[float]]) -> bool:
    """Ray casting. `ring` is GeoJSON order: [lon, lat] pairs."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            denom = yj - yi
            if denom != 0 and lon < (xj - xi) * (lat - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def point_in_polygon(lon: float, lat: float, rings: Sequence[Sequence[Sequence[float]]]) -> bool:
    """First ring is the exterior; the rest are holes."""
    if not rings or not point_in_ring(lon, lat, rings[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in rings[1:])


def _bbox(rings) -> tuple[float, float, float, float]:
    xs = [p[0] for p in rings[0]]
    ys = [p[1] for p in rings[0]]
    return min(xs), min(ys), max(xs), max(ys)


def load_polygons(path: str) -> list[list[list[list[float]]]]:
    """Read polygons from GeoJSON, or from anything geopandas can open.

    Returns a flat list of polygons, each a list of rings, each a list of
    [lon, lat] pairs — the GeoJSON coordinate convention throughout.
    """
    if path.lower().endswith((".json", ".geojson")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        geoms = []
        if data.get("type") == "FeatureCollection":
            geoms = [f.get("geometry") for f in data.get("features", [])]
        elif data.get("type") == "GeometryCollection":
            geoms = data.get("geometries", [])
        else:
            geoms = [data]
        out = []
        for g in geoms:
            if not g:
                continue
            if g["type"] == "Polygon":
                out.append(g["coordinates"])
            elif g["type"] == "MultiPolygon":
                out.extend(g["coordinates"])
        return out

    # Shapefile, GeoPackage, FlatGeobuf and friends.
    try:
        import geopandas as gpd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"reading '{path}' needs geopandas; convert to GeoJSON, or "
            "pip install geopandas"
        ) from exc
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        # Everything downstream assumes lon/lat degrees.
        gdf = gdf.to_crs(4326)
    return load_polygons_from_geojson_dict(json.loads(gdf.to_json()))


def load_polygons_from_geojson_dict(data: dict) -> list:
    out = []
    for f in data.get("features", []):
        g = f.get("geometry")
        if not g:
            continue
        if g["type"] == "Polygon":
            out.append(g["coordinates"])
        elif g["type"] == "MultiPolygon":
            out.extend(g["coordinates"])
    return out


# -- mask interface -------------------------------------------------------

@dataclass
class MaskDecision:
    """Why a cell was excluded. Reasons are surfaced in the UI so an excluded
    cell reads as 'no road here' rather than 'the map is broken'."""

    excluded: dict[str, str] = field(default_factory=dict)

    def is_excluded(self, cell_id: str) -> bool:
        return cell_id in self.excluded

    def reason(self, cell_id: str) -> Optional[str]:
        return self.excluded.get(cell_id)


class CellMask(ABC):
    """Filters cells before any quota is spent on them."""

    def prepare(self, cells: Sequence[Cell], state: Optional[dict] = None) -> None:
        """Called once at startup with any persisted state."""

    @abstractmethod
    def evaluate(self, cells: Sequence[Cell]) -> MaskDecision: ...

    def observe(self, requested: Iterable[str], returned: Iterable[str]) -> None:
        """Feed back what a cycle asked for and what came back. Cells requested
        but not returned failed upstream."""

    def state(self) -> dict:
        """Serialisable state to persist between runs."""
        return {}

    def describe(self) -> dict:
        return {"type": getattr(self, "registry_name", type(self).__name__)}


@MASKS.register("none")
class NullMask(CellMask):
    def evaluate(self, cells):
        return MaskDecision()


@MASKS.register("polygon")
class PolygonMask(CellMask):
    """Exclude (or restrict to) cells falling inside supplied polygons.

    Rather than testing the centre alone, the cell polygon's vertices are
    sampled too and the cell is kept only if enough of those points are on
    land. A hex straddling a shoreline with its centre just offshore still has
    routable roads in it, and testing one point would throw it away.

    Sources worth knowing about, all free:

    * US Census TIGER/Line **AREAWATER** shapefiles — per county, authoritative,
      and the right choice for a US metro.
    * OpenStreetMap water polygons from ``osmdata.openstreetmap.de`` — global
      coastline plus inland water.
    * Natural Earth — global but far too coarse for a city; it will happily
      call the Charles River dry land.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        mode: str = "exclude",
        min_land_fraction: float = 0.5,
        sample_vertices: bool = True,
        reason: str = "water",
        **_ignored,
    ):
        self.path = path
        if mode not in ("exclude", "include"):
            raise ValueError("mode must be 'exclude' or 'include'")
        self.mode = mode
        self.min_land_fraction = float(min_land_fraction)
        self.sample_vertices = bool(sample_vertices)
        self.reason = reason
        self._polys: list = []
        self._bboxes: list = []
        self._tree = None
        self._shapely_polys: list = []
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        self._polys = load_polygons(path)
        self._bboxes = [_bbox(p) for p in self._polys]
        if HAVE_SHAPELY and self._polys:
            self._shapely_polys = [
                _shapely_shape({"type": "Polygon", "coordinates": p}) for p in self._polys
            ]
            self._tree = _STRtree(self._shapely_polys)

    def contains(self, coord: Coord) -> bool:
        lon, lat = coord.lon, coord.lat
        if self._tree is not None:
            pt = _ShapelyPoint(lon, lat)
            for idx in self._tree.query(pt):
                if self._shapely_polys[int(idx)].contains(pt):
                    return True
            return False
        for poly, (x0, y0, x1, y1) in zip(self._polys, self._bboxes):
            if x0 <= lon <= x1 and y0 <= lat <= y1 and point_in_polygon(lon, lat, poly):
                return True
        return False

    def _sample_points(self, cell: Cell) -> list[Coord]:
        pts = [cell.center]
        if self.sample_vertices and cell.polygon:
            # Pull vertices toward the centre so a shared edge is not counted
            # for both neighbours.
            for v in cell.polygon:
                pts.append(
                    Coord(
                        cell.center.lat + 0.75 * (v.lat - cell.center.lat),
                        cell.center.lon + 0.75 * (v.lon - cell.center.lon),
                    )
                )
        return pts

    def evaluate(self, cells):
        decision = MaskDecision()
        if not self._polys:
            return decision
        for cell in cells:
            pts = self._sample_points(cell)
            hits = sum(1 for p in pts if self.contains(p))
            inside_frac = hits / len(pts)
            land_frac = (1.0 - inside_frac) if self.mode == "exclude" else inside_frac
            if land_frac < self.min_land_fraction:
                decision.excluded[cell.id] = self.reason
        return decision

    def describe(self):
        d = super().describe()
        d.update(
            {
                "polygons": len(self._polys),
                "mode": self.mode,
                "shapely": HAVE_SHAPELY,
                "source": self.path,
            }
        )
        return d


@MASKS.register("quarantine")
class QuarantineMask(CellMask):
    """Learn unroutable cells from provider behaviour.

    A cell requested but not returned failed upstream. After
    `failures_before_quarantine` consecutive failures it is excluded, then
    re-probed after a backoff that doubles with each further failure. That way
    a transient outage does not permanently delete a cell, but a hex sitting in
    the harbour stops costing quota after a handful of attempts.

    Cheap insurance: with TomTom each discovery costs a couple of transactions
    once, versus one per cycle forever.
    """

    def __init__(
        self,
        failures_before_quarantine: int = 3,
        base_backoff_cycles: int = 24,
        max_backoff_cycles: int = 1024,
        reason: str = "unroutable",
        **_ignored,
    ):
        self.threshold = int(failures_before_quarantine)
        self.base_backoff = int(base_backoff_cycles)
        self.max_backoff = int(max_backoff_cycles)
        self.reason = reason
        self.cycle = 0
        #: cell_id -> {"fails": int, "quarantines": int, "until": int}
        self.records: dict[str, dict] = {}

    def prepare(self, cells, state=None):
        if state:
            self.records = {k: dict(v) for k, v in state.get("records", {}).items()}
            self.cycle = int(state.get("cycle", 0))

    def tick(self, cycle: int) -> None:
        self.cycle = cycle

    def evaluate(self, cells):
        decision = MaskDecision()
        for cell in cells:
            rec = self.records.get(cell.id)
            if rec and rec.get("until", 0) > self.cycle:
                decision.excluded[cell.id] = self.reason
        return decision

    def observe(self, requested, returned):
        returned = set(returned)
        for cell_id in requested:
            if cell_id in returned:
                # One success clears the record entirely. Roads get built and
                # geometry gets fixed; nothing should be excluded forever.
                self.records.pop(cell_id, None)
                continue
            rec = self.records.setdefault(cell_id, {"fails": 0, "quarantines": 0, "until": 0})
            rec["fails"] += 1
            if rec["fails"] >= self.threshold:
                backoff = min(
                    self.base_backoff * (2 ** rec["quarantines"]), self.max_backoff
                )
                rec["quarantines"] += 1
                rec["fails"] = 0
                rec["until"] = self.cycle + backoff

    def state(self):
        return {"records": self.records, "cycle": self.cycle}

    def describe(self):
        d = super().describe()
        active = sum(1 for r in self.records.values() if r.get("until", 0) > self.cycle)
        d.update({"quarantined": active, "tracked": len(self.records)})
        return d


@MASKS.register("composite")
class CompositeMask(CellMask):
    """Chain several masks. A cell excluded by any one is excluded."""

    def __init__(self, masks: Optional[Sequence] = None, **_ignored):
        self.masks: list[CellMask] = []
        for spec in masks or []:
            if isinstance(spec, CellMask):
                self.masks.append(spec)
            else:
                spec = dict(spec)
                kind = spec.pop("kind")
                self.masks.append(MASKS.build(kind, **spec))

    def prepare(self, cells, state=None):
        state = state or {}
        for i, m in enumerate(self.masks):
            m.prepare(cells, state.get(str(i)))

    def tick(self, cycle: int) -> None:
        for m in self.masks:
            if hasattr(m, "tick"):
                m.tick(cycle)

    def evaluate(self, cells):
        out = MaskDecision()
        for m in self.masks:
            for cid, reason in m.evaluate(cells).excluded.items():
                out.excluded.setdefault(cid, reason)
        return out

    def observe(self, requested, returned):
        requested = list(requested)
        returned = list(returned)
        for m in self.masks:
            m.observe(requested, returned)

    def state(self):
        return {str(i): m.state() for i, m in enumerate(self.masks)}

    def describe(self):
        d = super().describe()
        d["masks"] = [m.describe() for m in self.masks]
        return d


def build_mask(kind: str, **kwargs) -> CellMask:
    return MASKS.build(kind, **kwargs)
