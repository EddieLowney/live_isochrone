"""Pedestrian network from OSM, and shortest walking times over it.

This replaces the straight-line walk model. Straight-line multiplies great
circle distance by a constant detour factor, which is roughly right on open
street grid and badly wrong exactly where a Boston map most needs to be right:
across the Charles, over the Mass Pike, through a rail yard. The error is not
noise around a mean, it is structured and concentrated at barriers.

Design notes worth knowing before changing anything here:

* **The graph is the cached artifact, not the answer.** It is origin-,
  grid- and resolution-independent, so it survives every change except an OSM
  refresh. Walk *matrices* derived from it are cheap and disposable.
* **Distance-limited search is what makes this fast.** Capping at a 15-minute
  walk touches roughly 2% of a metro-scale graph, so a source costs ~12 ms
  rather than ~56 ms. All 1,437 stops within reach of a 5 km grid take seconds.
* **Sources must be chunked.** scipy returns a dense (sources x nodes) array
  regardless of how few columns you want. 1,000 sources over 300k nodes is
  2.4 GB; in chunks of 50 it is 120 MB.

Extraction shells out to `osmium`, which is already required to crop the OSM
extract during OTP setup. That avoids pulling in pyrosm and its geopandas
stack for what amounts to reading a list of line strings.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

EARTH_KM = 6371.0088

#: Ways a pedestrian may use. Anything not listed is ignored, which is safer
#: than a denylist: an unfamiliar highway type should not silently become a
#: walking route.
WALKABLE = {
    "footway", "path", "pedestrian", "steps", "living_street", "residential",
    "service", "unclassified", "tertiary", "tertiary_link", "secondary",
    "secondary_link", "primary", "primary_link", "track", "corridor",
    "crossing", "cycleway",
}

#: Speed multipliers. Stairs are genuinely slower, and a trunk road with a
#: sidewalk is unpleasant enough that people avoid it — a mild penalty
#: approximates that without needing a full route-choice model.
SPEED_FACTOR = {"steps": 0.4, "primary": 0.9, "secondary": 0.95, "track": 0.8}

MAX_TIME = np.int32(2**31 - 1)


def _osmium() -> str:
    exe = shutil.which("osmium")
    if not exe:
        raise RuntimeError(
            "osmium not found. brew install osmium-tool — it is the same tool "
            "otp/setup.sh uses to crop the OSM extract."
        )
    return exe


def export_walk_geojson(pbf_path: str, out_path: str) -> str:
    """Filter an OSM extract to walkable ways and export as GeoJSON-seq.

    Two passes on purpose: filtering first shrinks the export by an order of
    magnitude, and `osmium export` is the slow step.
    """
    osmium = _osmium()
    with tempfile.TemporaryDirectory() as tmp:
        filtered = str(Path(tmp) / "walk.osm.pbf")
        subprocess.run(
            [osmium, "tags-filter", pbf_path, "w/highway", "-o", filtered,
             "--overwrite"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [osmium, "export", filtered, "-f", "geojsonseq",
             "--geometry-types=linestring", "-o", out_path, "--overwrite"],
            check=True, capture_output=True,
        )
    return out_path


@dataclass
class PedestrianGraph:
    """Undirected walking network with node coordinates and a spatial index."""

    lat: np.ndarray
    lon: np.ndarray
    graph: object                 # scipy CSR, edge weights in seconds
    walk_speed_ms: float
    #: Locally projected metres, for the KD-tree. Equirectangular is accurate
    #: enough over a metro area and avoids a projection dependency.
    _xy: Optional[np.ndarray] = None
    _tree: Optional[cKDTree] = None

    def __post_init__(self):
        lat0 = float(np.mean(self.lat)) if len(self.lat) else 0.0
        k = math.cos(math.radians(lat0))
        self._xy = np.c_[
            np.radians(self.lon) * EARTH_KM * 1000 * k,
            np.radians(self.lat) * EARTH_KM * 1000,
        ]
        self._tree = cKDTree(self._xy)

    @property
    def n_nodes(self) -> int:
        return len(self.lat)

    def snap(self, lats, lons, max_snap_m: float = 250.0):
        """Nearest network node to each point, plus the off-network walk.

        A cell centre is rarely on a sidewalk. The snap distance is added as a
        straight-line penalty, which is the one place a straight line is
        defensible: it is the last few metres from the footway to the point.
        Returns (node_index, penalty_seconds); index is -1 beyond `max_snap_m`.
        """
        lat = np.atleast_1d(np.asarray(lats, float))
        lon = np.atleast_1d(np.asarray(lons, float))
        lat0 = float(np.mean(self.lat))
        k = math.cos(math.radians(lat0))
        q = np.c_[np.radians(lon) * EARTH_KM * 1000 * k,
                  np.radians(lat) * EARTH_KM * 1000]
        dist, idx = self._tree.query(q, k=1)
        idx = np.asarray(idx, dtype=np.int64)
        too_far = dist > max_snap_m
        idx[too_far] = -1
        penalty = (dist / self.walk_speed_ms).astype(np.int32)
        penalty[too_far] = 0
        return idx, penalty

    def walk_matrix(
        self,
        src_lat, src_lon,
        dst_lat, dst_lon,
        max_seconds: float = 900.0,
        chunk: int = 50,
        max_snap_m: float = 250.0,
    ) -> np.ndarray:
        """Walk seconds from each source to each destination.

        Shape (n_src, n_dst), MAX_TIME where unreachable within `max_seconds`.
        """
        s_node, s_pen = self.snap(src_lat, src_lon, max_snap_m)
        d_node, d_pen = self.snap(dst_lat, dst_lon, max_snap_m)
        n_src, n_dst = len(s_node), len(d_node)
        out = np.full((n_src, n_dst), MAX_TIME, dtype=np.int32)

        valid_dst = d_node >= 0
        if not valid_dst.any():
            return out
        cols = d_node[valid_dst]

        for lo in range(0, n_src, chunk):
            hi = min(lo + chunk, n_src)
            block = s_node[lo:hi]
            live = block >= 0
            if not live.any():
                continue
            # `limit` prunes the search rather than masking afterwards, which
            # is where nearly all of the speed comes from.
            dist = dijkstra(
                self.graph, directed=False, indices=block[live],
                limit=float(max_seconds),
            )
            sub = dist[:, cols]
            rows = np.flatnonzero(live) + lo
            total = sub + s_pen[rows][:, None] + d_pen[valid_dst][None, :]
            total[~np.isfinite(sub)] = MAX_TIME
            total = np.where(total > max_seconds, MAX_TIME, total)
            out[np.ix_(rows, np.flatnonzero(valid_dst))] = total.astype(np.int32)
        return out

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        g = self.graph.tocoo()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, lat=self.lat, lon=self.lon, row=g.row.astype(np.int32),
            col=g.col.astype(np.int32), w=g.data.astype(np.float32),
            speed=np.array([self.walk_speed_ms]),
        )

    @classmethod
    def load(cls, path: str) -> "PedestrianGraph":
        z = np.load(path)
        n = len(z["lat"])
        g = coo_matrix((z["w"], (z["row"], z["col"])), shape=(n, n)).tocsr()
        return cls(lat=z["lat"], lon=z["lon"], graph=g,
                   walk_speed_ms=float(z["speed"][0]))

    def describe(self) -> dict:
        return {
            "nodes": int(self.n_nodes),
            "edges": int(self.graph.nnz // 2),
            "walk_speed_ms": self.walk_speed_ms,
        }


def build_from_geojsonseq(
    path: str,
    walk_speed_ms: float = 1.33,
    precision: int = 7,
    crossing_penalty_s: float = 0.0,
) -> PedestrianGraph:
    """Assemble a graph from exported line strings.

    Node identity comes from rounded coordinates rather than OSM node ids,
    which the GeoJSON export drops. Seven decimal places is ~1 cm, far below
    the precision at which two ways genuinely share a node, so this joins the
    network correctly without needing the original ids.
    """
    coord_id: dict[tuple, int] = {}
    lats: list[float] = []
    lons: list[float] = []
    er, ec, ew = [], [], []

    def node(lon: float, lat: float) -> int:
        key = (round(lon, precision), round(lat, precision))
        i = coord_id.get(key)
        if i is None:
            i = len(lats)
            coord_id[key] = i
            lons.append(lon)
            lats.append(lat)
        return i

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip().lstrip("\x1e")  # geojsonseq record separator
            if not line:
                continue
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                continue
            geom = feat.get("geometry") or {}
            if geom.get("type") != "LineString":
                continue
            tags = feat.get("properties") or {}
            hw = tags.get("highway")
            if hw not in WALKABLE:
                continue
            if tags.get("foot") in ("no", "private"):
                continue
            if tags.get("access") in ("no", "private") and tags.get("foot") not in ("yes", "designated"):
                continue
            speed = walk_speed_ms * SPEED_FACTOR.get(hw, 1.0)
            coords = geom.get("coordinates") or []
            prev = None
            for lon, lat in coords:
                cur = node(float(lon), float(lat))
                if prev is not None and prev != cur:
                    d = _haversine_m(lats[prev], lons[prev], lats[cur], lons[cur])
                    w = d / speed
                    if hw == "crossing":
                        w += crossing_penalty_s
                    er.append(prev); ec.append(cur); ew.append(w)
                    er.append(cur); ec.append(prev); ew.append(w)
                prev = cur

    n = len(lats)
    if n == 0:
        raise RuntimeError(f"no walkable ways found in {path}")
    g = coo_matrix(
        (np.asarray(ew, np.float32), (np.asarray(er), np.asarray(ec))),
        shape=(n, n),
    ).tocsr()
    return PedestrianGraph(
        lat=np.asarray(lats), lon=np.asarray(lons), graph=g,
        walk_speed_ms=walk_speed_ms,
    )


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * 1000 * math.asin(math.sqrt(min(1.0, a)))


def load_or_build(
    pbf_path: str,
    cache_path: str,
    walk_speed_ms: float = 1.33,
    rebuild: bool = False,
    crossing_penalty_s: float = 0.0,
) -> PedestrianGraph:
    """Cached graph. Extraction is the slow part and happens once."""
    if Path(cache_path).is_file() and not rebuild:
        g = PedestrianGraph.load(cache_path)
        if abs(g.walk_speed_ms - walk_speed_ms) < 1e-9:
            return g
    with tempfile.TemporaryDirectory() as tmp:
        gj = str(Path(tmp) / "walk.geojsonseq")
        export_walk_geojson(pbf_path, gj)
        g = build_from_geojsonseq(gj, walk_speed_ms,
                                  crossing_penalty_s=crossing_penalty_s)
    g.save(cache_path)
    return g
