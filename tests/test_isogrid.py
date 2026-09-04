"""Tests. Run with `python -m pytest tests` or `python tests/test_isogrid.py`.

The budget tests matter most: a quota that is merely usually respected is not a
quota. They cover concurrent spend, refunds, and period rollover.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isogrid import Config, Engine  # noqa: E402
from isogrid.budget import MemoryLedger, ProportionalPacer, SqliteLedger, period_key  # noqa: E402
from isogrid.grid import HexGrid  # noqa: E402
from isogrid.reconstruct import FactorReconstructor, IDWReconstructor  # noqa: E402
from isogrid.types import Coord, ProviderError  # noqa: E402

ORIGIN = Coord(42.3519, -71.0552)
T0 = datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)


def _cfg(**over):
    base = {
        "store": {"kind": "memory"},
        "budget": {"ledger": {"kind": "memory"}},
    }
    base.update(over)
    return Config.load("config/prototype.yaml", overrides=base)


# -- grid -----------------------------------------------------------------

def test_hex_cell_count():
    for rings, expected in ((0, 1), (1, 7), (2, 19), (3, 37), (4, 61)):
        assert len(HexGrid(ORIGIN, cell_km=1.5, rings=rings)) == expected


def test_hex_neighbours_are_symmetric():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=2)
    for cell in g:
        for n in g.neighbours(cell.id):
            assert cell.id in g.neighbours(n)


def test_hex_spacing_matches_cell_km():
    g = HexGrid(ORIGIN, cell_km=2.0, rings=1)
    centre = g.get("h:0:0")
    dists = [centre.center.haversine_km(g.get(n).center) for n in g.neighbours("h:0:0")]
    for d in dists:
        assert abs(d - 2.0 * 1.7320508 / 2 * 2) < 0.05  # sqrt(3) * cell_km


# -- budget ---------------------------------------------------------------

def test_ledger_refuses_past_cap():
    led = MemoryLedger({"p": 100}, period="monthly", safety=1.0)
    assert led.try_spend("p", 60, T0)
    assert led.try_spend("p", 40, T0)
    assert not led.try_spend("p", 1, T0)
    assert led.remaining("p", T0) == 0


def test_safety_margin_reserves_headroom():
    led = MemoryLedger({"p": 1000}, safety=0.9)
    assert led.effective_quota("p") == 900
    assert not led.try_spend("p", 901, T0)


def test_refund_restores_capacity():
    led = MemoryLedger({"p": 10}, safety=1.0)
    assert led.try_spend("p", 10, T0)
    led.refund("p", 4, T0)
    assert led.try_spend("p", 4, T0)
    assert not led.try_spend("p", 1, T0)


def test_period_rollover_resets():
    led = MemoryLedger({"p": 10}, period="monthly", safety=1.0)
    assert led.try_spend("p", 10, T0)
    assert not led.try_spend("p", 1, T0)
    assert led.try_spend("p", 10, T0 + timedelta(days=40))


def test_sqlite_ledger_is_atomic_under_threads():
    """The point of the conditional UPDATE: concurrent workers must not
    collectively exceed the cap."""
    with tempfile.TemporaryDirectory() as d:
        led = SqliteLedger({"p": 100}, safety=1.0, path=os.path.join(d, "b.db"))
        wins = []
        lock = threading.Lock()

        def worker():
            for _ in range(30):
                if led.try_spend("p", 1, T0):
                    with lock:
                        wins.append(1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(wins) == 100
        assert led.spent("p", T0) == 100


def test_proportional_pacer_tapers_as_budget_drains():
    led = MemoryLedger({"p": 10000}, safety=1.0)
    pacer = ProportionalPacer(burst=1.0)
    early = pacer.allowance(led, "p", T0, 300)
    led.try_spend("p", 9000, T0)
    late = pacer.allowance(led, "p", T0, 300)
    assert late < early


def test_period_key_shapes():
    assert period_key("monthly", T0) == "2026-03"
    assert period_key("daily", T0) == "2026-03-02"


# -- engine ---------------------------------------------------------------

def test_engine_never_exceeds_quota():
    """The headline guarantee, exercised against a quota small enough to bite."""
    cfg = _cfg()
    cfg.raw["budget"]["quotas"] = {"synthetic": 500}
    cfg.raw["budget"]["safety"] = 1.0
    cfg.raw["budget"]["pacer"] = {"kind": "flat", "units_per_cycle": 25}
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    eng.simulate(200, start=T0)
    assert c.ledger.spent("synthetic", T0) <= 500


def test_engine_degrades_instead_of_raising_when_exhausted():
    cfg = _cfg()
    cfg.raw["budget"]["quotas"] = {"synthetic": 60}
    cfg.raw["budget"]["safety"] = 1.0
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    reports = eng.simulate(30, start=T0)
    assert any(r.batches_refused > 0 for r in reports)
    assert reports[-1].cells_measured == 0  # still returns a surface


def test_provider_failure_is_refunded():
    cfg = _cfg()
    cfg.raw["provider"]["failure_rate"] = 1.0
    cfg.raw["budget"]["quotas"] = {"synthetic": 1000}
    c = cfg.build()
    eng = Engine(c)
    c.store.set_baseline({cell.id: 600.0 for cell in c.grid}, "driving")
    before = c.ledger.spent("synthetic", T0)
    _s, rep = eng.run_cycle(T0)
    assert rep.errors
    assert c.ledger.spent("synthetic", T0) == before


def test_surface_covers_every_cell():
    c = _cfg().build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    surface, _ = eng.run_cycle(T0)
    assert len(surface.cells) == len(c.grid)
    assert all(cell.baseline_s is not None for cell in surface.cells)


def test_geojson_is_wellformed():
    c = _cfg().build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    eng.run_cycle(T0)
    gj = eng.geojson()
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == len(c.grid)
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]  # closed


# -- reconstruction -------------------------------------------------------

def test_factor_model_beats_spatial_interpolation():
    """Congestion is corridor-structured, so a factor model should beat pure
    distance weighting. If this regresses, the sampler is probably starving the
    rotation and the model has no history for unmeasured cells."""
    scores = {}
    for kind in ("idw", "factor"):
        cfg = _cfg()
        cfg.raw["reconstruct"]["kind"] = kind
        cfg.raw["budget"]["pacer"] = {"kind": "flat", "units_per_cycle": 10}
        c = cfg.build()
        eng = Engine(c)
        eng.bootstrap_baseline(when=T0)
        reps = eng.simulate(280, start=T0)
        warm = [r.holdout_mae_s for r in reps[120:] if r.holdout_mae_s is not None]
        scores[kind] = sum(warm) / len(warm)
    assert scores["factor"] < scores["idw"], scores


def test_factor_falls_back_before_it_is_fitted():
    r = FactorReconstructor(min_cycles=100)
    r.fit([], {}, [])
    assert not r.fitted
    g = HexGrid(ORIGIN, cell_km=1.5, rings=1)
    base = {c.id: 600.0 for c in g}
    out = r.estimate(g.cells, {"h:1:0": 1.4}, base)
    assert len(out) == len(g)
    assert all(v > 0 for v in out.values())


def test_idw_reproduces_observed_cells_exactly():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=2)
    base = {c.id: 600.0 for c in g}
    obs = {"h:2:0": 1.8}
    out = IDWReconstructor().estimate(g.cells, obs, base)
    assert abs(out["h:2:0"] - 1.8) < 1e-9


def test_config_swaps_components_by_name():
    cfg = _cfg()
    cfg.raw["grid"] = {"kind": "square", "cell_km": 1.0, "rings": 2}
    cfg.raw["sampling"] = {"kind": "staleness"}
    cfg.raw["reconstruct"] = {"kind": "passthrough"}
    c = cfg.build()
    assert len(c.grid) == 25
    Engine(c).bootstrap_baseline(when=T0)


# -- tomtom ---------------------------------------------------------------

from isogrid.providers.tomtom import TomTomMatrixProvider, matrix_cost  # noqa: E402


class StubTomTom(TomTomMatrixProvider):
    """TomTom with the HTTP call replaced by a canned response matching the
    documented v2 schema, including an out-of-order cell and a billed failure."""

    def __init__(self, **kw):
        super().__init__(api_key="stub", **kw)
        self.bodies = []

    def _post(self, url, body):
        self.bodies.append(body)
        n = len(body["destinations"])
        data = []
        for i in reversed(range(n)):          # deliberately out of order
            if i == 1 and n > 2:
                data.append({"originIndex": 0, "destinationIndex": i,
                             "detailedError": {"code": "CELL_PROCESSING_ERROR",
                                               "innerError": {"code": "NO_ROUTE_FOUND"}}})
                continue
            data.append({"originIndex": 0, "destinationIndex": i,
                         "routeSummary": {"lengthInMeters": 5000 + i,
                                          "travelTimeInSeconds": 600 + 60 * i,
                                          "trafficDelayInSeconds": 90 + i}})
        return {"data": data, "statistics": {"totalCount": n}}


def test_tomtom_billing_rule():
    assert matrix_cost(1, 5) == 5          # both dims <= 5: cheap branch
    assert matrix_cost(5, 5) == 25
    assert matrix_cost(1, 37) == 185       # penalty branch: max(dim) * 5
    assert matrix_cost(37, 37) == 185      # square costs the same as 1xN


def test_tomtom_batching_beats_naive_request():
    """The whole reason batch_size is 5. A single 1x37 request bills 185; eight
    batches of five bill 37 for identical data."""
    g = HexGrid(ORIGIN, cell_km=1.5, rings=3)
    p = StubTomTom()
    batched = sum(b.elements for b in p.plan(g.cells))
    assert batched == len(g) == 37
    assert matrix_cost(1, len(g)) == 185
    assert all(len(b) <= 5 for b in p.plan(g.cells))


def test_tomtom_batch_size_cannot_exceed_five():
    assert StubTomTom(batch_size=25).batch_size == 5


def test_tomtom_matches_results_by_destination_index():
    """Responses are not ordered, so cells must be matched by index."""
    g = HexGrid(ORIGIN, cell_km=1.5, rings=1)
    p = StubTomTom()
    batch = p.plan(g.cells[:4])[0]
    got = {m.cell_id: m.seconds for m in p.fetch(ORIGIN, batch, "live")}
    for i, cell in enumerate(batch.cells):
        if i == 1:
            assert cell.id not in got      # the failed cell
        else:
            assert got[cell.id] == 600 + 60 * i


def test_tomtom_derives_free_flow_from_traffic_delay():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=1)
    p = StubTomTom()
    batch = p.plan(g.cells[:2])[0]
    for m in p.fetch(ORIGIN, batch, "live"):
        assert m.free_flow_seconds is not None
        assert m.seconds - m.free_flow_seconds > 0     # delay is positive


def test_tomtom_counts_billed_cell_failures():
    """NO_ROUTE_FOUND and MAP_MATCHING_FAILURE are charged, so they must be
    visible rather than silently eating the safety margin."""
    g = HexGrid(ORIGIN, cell_km=1.5, rings=1)
    p = StubTomTom()
    p.fetch(ORIGIN, p.plan(g.cells[:4])[0], "live")
    assert p.billed_failures == 1


def test_tomtom_live_and_historical_options_differ():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=1)
    p = StubTomTom()
    batch = p.plan(g.cells[:2])[0]
    p.fetch(ORIGIN, batch, "live")
    p.fetch(ORIGIN, batch, "historical")
    live, hist = p.bodies[0]["options"], p.bodies[1]["options"]
    assert live["traffic"] == "live" and live["departAt"] == "now"
    # traffic=live is incompatible with departAt=any, so they must not mix
    assert hist["traffic"] == "historical" and hist["departAt"] == "any"


def test_daily_period_resets_overnight():
    """TomTom's quota is daily, unlike Mapbox's monthly."""
    led = MemoryLedger({"tomtom": 100}, period="daily", safety=1.0)
    assert led.try_spend("tomtom", 100, T0)
    assert not led.try_spend("tomtom", 1, T0)
    assert led.try_spend("tomtom", 100, T0 + timedelta(days=1))


def test_seed_baseline_from_live_measurements():
    """Providers returning free-flow skip the separate baseline sweep."""
    c = _cfg().build()
    eng = Engine(c)
    n = eng.seed_baseline_from_live(when=T0)
    assert n == len(c.grid)
    assert all(v >= 0 for v in c.store.get_baseline().values())


def test_tomtom_config_fits_its_daily_quota():
    """The shipped config must not plan to overspend."""
    cfg = Config.load("config/tomtom.yaml",
                      overrides={"store": {"kind": "memory"},
                                 "budget": {"ledger": {"kind": "memory"}}})
    c = cfg.build()
    cycles = 86400 / cfg.cycle_seconds
    allowance = c.pacer.allowance(c.ledger, "tomtom", T0, cfg.cycle_seconds)
    assert allowance * cycles <= c.ledger.effective_quota("tomtom")


# -- masks ----------------------------------------------------------------

from isogrid.masks import (  # noqa: E402
    CompositeMask, PolygonMask, QuarantineMask, point_in_polygon,
)


def _water_geojson(path, min_lat, min_lon, max_lat, max_lon):
    """A rectangular 'lake' in GeoJSON lon/lat order."""
    import json
    ring = [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat],
            [min_lon, max_lat], [min_lon, min_lat]]
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Polygon", "coordinates": [ring]}}]}, fh)
    return path


def test_point_in_polygon_handles_holes():
    outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]
    assert point_in_polygon(1, 1, [outer])
    assert point_in_polygon(1, 1, [outer, hole])
    assert not point_in_polygon(5, 5, [outer, hole])   # inside the hole
    assert not point_in_polygon(20, 20, [outer])


def test_polygon_mask_excludes_cells_in_water():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=3)
    with tempfile.TemporaryDirectory() as d:
        # A lake covering everything north of the origin.
        path = _water_geojson(os.path.join(d, "w.geojson"),
                              ORIGIN.lat + 0.01, ORIGIN.lon - 1,
                              ORIGIN.lat + 1, ORIGIN.lon + 1)
        m = PolygonMask(path=path, min_land_fraction=0.5)
        decision = m.evaluate(g.cells)
    assert decision.excluded
    for cid in decision.excluded:
        assert g.get(cid).center.lat > ORIGIN.lat      # only northern cells
        assert decision.reason(cid) == "water"
    assert not decision.is_excluded("h:0:0")           # origin stays


def test_polygon_mask_include_mode_is_a_study_boundary():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=3)
    with tempfile.TemporaryDirectory() as d:
        path = _water_geojson(os.path.join(d, "b.geojson"),
                              ORIGIN.lat - 0.02, ORIGIN.lon - 0.02,
                              ORIGIN.lat + 0.02, ORIGIN.lon + 0.02)
        keep = PolygonMask(path=path, mode="include", reason="out_of_area")
        decision = keep.evaluate(g.cells)
    assert not decision.is_excluded("h:0:0")           # centre is inside
    assert len(decision.excluded) > 0                  # outer rings are not


def test_polygon_mask_with_no_data_excludes_nothing():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=2)
    assert PolygonMask().evaluate(g.cells).excluded == {}


def test_quarantine_needs_repeated_failures():
    m = QuarantineMask(failures_before_quarantine=3)
    g = HexGrid(ORIGIN, cell_km=1.5, rings=1)
    ids = [c.id for c in g]
    for _ in range(2):
        m.observe(ids, [i for i in ids if i != "h:1:0"])
        assert not m.evaluate(g.cells).is_excluded("h:1:0")   # not yet
    m.observe(ids, [i for i in ids if i != "h:1:0"])
    assert m.evaluate(g.cells).is_excluded("h:1:0")


def test_quarantine_clears_on_a_single_success():
    """Roads get built and geometry gets fixed; nothing is excluded forever."""
    m = QuarantineMask(failures_before_quarantine=2, base_backoff_cycles=5)
    g = HexGrid(ORIGIN, cell_km=1.5, rings=1)
    ids = [c.id for c in g]
    for _ in range(2):
        m.observe(ids, [i for i in ids if i != "h:1:0"])
    assert m.evaluate(g.cells).is_excluded("h:1:0")
    m.tick(10)                                   # backoff expires
    assert not m.evaluate(g.cells).is_excluded("h:1:0")
    m.observe(ids, ids)                          # succeeds on re-probe
    assert "h:1:0" not in m.records


def test_quarantine_backoff_doubles():
    m = QuarantineMask(failures_before_quarantine=1, base_backoff_cycles=10)
    ids = ["a"]
    m.observe(ids, [])
    first = m.records["a"]["until"]
    m.tick(first)
    m.observe(ids, [])
    assert m.records["a"]["until"] - first >= 20   # doubled


def test_quarantine_state_survives_a_restart():
    m = QuarantineMask(failures_before_quarantine=1)
    m.observe(["a"], [])
    state = m.state()
    fresh = QuarantineMask(failures_before_quarantine=1)
    fresh.prepare([], state)
    assert fresh.records == m.records


def test_composite_mask_unions_exclusions():
    g = HexGrid(ORIGIN, cell_km=1.5, rings=2)
    q = QuarantineMask(failures_before_quarantine=1)
    q.observe(["h:1:0"], [])
    with tempfile.TemporaryDirectory() as d:
        path = _water_geojson(os.path.join(d, "w.geojson"),
                              ORIGIN.lat + 0.01, ORIGIN.lon - 1,
                              ORIGIN.lat + 1, ORIGIN.lon + 1)
        comp = CompositeMask(masks=[{"kind": "polygon", "path": path}, q])
        decision = comp.evaluate(g.cells)
    assert decision.is_excluded("h:1:0")
    assert any(r == "water" for r in decision.excluded.values())


def test_engine_quarantines_unroutable_cells_and_stops_paying():
    """The end-to-end payoff: dead cells cost quota briefly, then stop."""
    dead = ["h:2:-1", "h:3:-2", "h:2:0", "h:3:-1"]
    cfg = _cfg()
    cfg.raw["provider"]["dead_cells"] = dead
    cfg.raw["budget"]["pacer"] = {"kind": "flat", "units_per_cycle": 37}
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    reps = eng.simulate(10, start=T0)
    assert reps[0].cells_failed == len(dead)
    assert reps[-1].cells_masked == len(dead)
    assert reps[-1].cells_failed == 0
    assert reps[-1].elements_spent < reps[0].elements_spent


def test_masked_cells_render_as_holes_not_estimates():
    cfg = _cfg()
    cfg.raw["provider"]["dead_cells"] = ["h:2:-1"]
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    eng.simulate(6, start=T0)
    surface, _ = eng.run_cycle(T0)
    masked = [x for x in surface.cells if x.masked]
    assert masked and all(x.estimated_s is None for x in masked)
    assert all(x.delta_s is None for x in masked)


def test_sampler_never_selects_masked_cells():
    from isogrid.sampling import SamplingContext, VolatilitySampler
    g = HexGrid(ORIGIN, cell_km=1.5, rings=2)
    excluded = {"h:1:0", "h:0:1"}
    ctx = SamplingContext(g, {c.id: 600.0 for c in g}, {}, [], 1, T0, excluded)
    picked = {c.id for c in VolatilitySampler().select(ctx, 19)}
    assert not (picked & excluded)
    assert len(picked) == len(g) - len(excluded)


def test_shipped_tomtom_config_never_walls_on_a_clean_day():
    """144 cycles x 15 units plus the seed must fit the daily cap."""
    cfg = Config.load("config/tomtom.yaml", overrides={
        "provider": {"kind": "synthetic"}, "store": {"kind": "memory"},
        "budget": {"ledger": {"kind": "memory"}, "quotas": {"synthetic": 2500}}})
    c = cfg.build()
    eng = Engine(c)
    t = datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)
    eng.seed_baseline_from_live(when=t)
    stalled = 0
    for _ in range(144):
        _s, r = eng.run_cycle(t)
        stalled += int(r.budget_stalled)
        t += timedelta(seconds=600)
    assert stalled == 0
    assert c.ledger.spent("synthetic", t - timedelta(seconds=600)) <= 2250


def test_zero_allowance_is_reported_not_silent():
    """A map that stops updating must say why."""
    cfg = _cfg()
    cfg.raw["budget"]["quotas"] = {"synthetic": 40}
    cfg.raw["budget"]["safety"] = 1.0
    cfg.raw["budget"]["pacer"] = {"kind": "flat", "units_per_cycle": 15}
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    reports = eng.simulate(8, start=T0)
    stalled = [r for r in reports if r.budget_stalled]
    assert stalled
    assert stalled[0].cells_measured == 0
    assert "quota exhausted" in stalled[0].errors[0]


def test_proportional_pacer_tapers_rather_than_stopping():
    """The contrast with flat: it slows down instead of going silent."""
    cfg = _cfg()
    cfg.raw["budget"]["quotas"] = {"synthetic": 400}
    cfg.raw["budget"]["safety"] = 1.0
    cfg.raw["budget"]["period"] = "daily"
    cfg.raw["budget"]["pacer"] = {"kind": "proportional", "burst": 1.0, "floor": 1}
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    reports = eng.simulate(120, start=T0)
    assert not any(r.budget_stalled for r in reports)
    assert reports[-1].cells_measured > 0


# -- tile proxy -----------------------------------------------------------

from isogrid.server import BLANK_PNG, TILE_RE, TileProxy  # noqa: E402


def test_tile_route_rejects_traversal_and_bad_layers():
    assert TILE_RE.match("/tiles/basic/night/11/618/755.png")
    assert TILE_RE.match("/tiles/labels/main/14/4944/6041.png")
    assert not TILE_RE.match("/tiles/satellite/main/11/618/755.png")
    assert not TILE_RE.match("/tiles/basic/dusk/11/618/755.png")
    assert not TILE_RE.match("/tiles/basic/main/../../etc/passwd.png")


def test_blank_png_is_a_valid_png():
    assert BLANK_PNG.startswith(b"\x89PNG\r\n\x1a\n")


def test_cache_hit_costs_no_quota():
    with tempfile.TemporaryDirectory() as d:
        led = MemoryLedger({"tomtom_tiles": 5}, period="daily", safety=1.0)
        tp = TileProxy(api_key="k", cache_dir=d, ledger=led)
        p = os.path.join(d, "basic", "night", "11", "618")
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "755.png"), "wb") as fh:
            fh.write(b"PNGDATA")
        data, src = tp.get("basic", "night", 11, 618, 755)
        assert src == "hit" and data == b"PNGDATA"
        assert led.spent("tomtom_tiles") == 0


def test_failed_tile_fetch_is_refunded():
    """No network in tests, so every miss fails — the quota must survive it."""
    with tempfile.TemporaryDirectory() as d:
        led = MemoryLedger({"tomtom_tiles": 5}, period="daily", safety=1.0)
        tp = TileProxy(api_key="k", cache_dir=d, ledger=led)
        data, src = tp.get("basic", "night", 11, 618, 755)
        assert src == "error" and data == BLANK_PNG
        assert led.spent("tomtom_tiles") == 0


def test_tile_quota_exhaustion_returns_blank_not_error():
    with tempfile.TemporaryDirectory() as d:
        led = MemoryLedger({"tomtom_tiles": 0}, period="daily", safety=1.0)
        tp = TileProxy(api_key="k", cache_dir=d, ledger=led)
        data, src = tp.get("basic", "night", 11, 1, 1)
        assert src == "refused" and data == BLANK_PNG


def test_tile_budget_is_separate_from_routing_budget():
    """Tiles must never eat the routing allowance."""
    cfg = Config.load("config/tomtom.yaml")
    quotas = cfg.raw["budget"]["quotas"]
    assert quotas["tomtom"] == 2500
    assert quotas["tomtom_tiles"] == 50000
    assert cfg.raw["render"]["basemap"]["budget_key"] == "tomtom_tiles"


def test_tile_proxy_disabled_without_a_key():
    assert not TileProxy(api_key="").available


def test_hexes_tessellate_without_gaps_or_overlap():
    """Every neighbour pair must share exactly one edge, i.e. two vertices.

    Catches the orientation bug: a flat-top hexagon drawn on a pointy-top
    lattice looks plausible cell by cell but overlaps along one axis and gaps
    along the other.
    """
    for cell_km in (0.099, 0.5, 2.0):
        g = HexGrid(ORIGIN, cell_km=cell_km, rings=2)
        for cell in g:
            for nid in g.neighbours(cell.id):
                shared = sum(
                    1
                    for va in cell.polygon
                    for vb in g.get(nid).polygon
                    if va.haversine_km(vb) * 1000 < 8.0
                )
                assert shared == 2, f"{cell.id}/{nid} shares {shared} vertices"


def test_hex_has_a_vertex_due_north():
    """Pointy-top orientation, which is what the axial spacing assumes."""
    g = HexGrid(ORIGIN, cell_km=1.0, rings=0)
    c = g.get("h:0:0")
    north = max(c.polygon, key=lambda v: v.lat)
    assert abs(north.lon - c.center.lon) < 1e-9
    assert abs(c.center.haversine_km(north) - 1.0) < 0.01


def test_grid_extent_matches_configured_scale():
    g = HexGrid(ORIGIN, cell_km=0.099, rings=3)
    extent = max(ORIGIN.haversine_km(v) for c in g for v in c.polygon) * 1000
    assert 550 < extent < 650, extent
    assert len(g) == 37


def test_changed_grid_is_refused_not_silently_reused():
    """Cell ids are axial coordinates, so they survive a geometry change. The
    stored baseline must not."""
    over = {"store": {"kind": "memory"}, "budget": {"ledger": {"kind": "memory"}},
            "provider": {"kind": "synthetic"}}
    cfg = Config.load("config/tomtom.yaml", overrides=over)
    c = cfg.build()
    eng = Engine(c)
    eng.seed_baseline_from_live(when=T0)

    moved = Config.load("config/tomtom.yaml", overrides=over)
    moved.raw["grid"]["cell_km"] = 0.25
    c2 = moved.build()
    c2.store = c.store
    try:
        Engine(c2)
        raise AssertionError("stale baseline was accepted")
    except RuntimeError as exc:
        assert "grid geometry changed" in str(exc)


def test_reset_clears_baseline_and_allows_the_new_grid():
    over = {"store": {"kind": "memory"}, "budget": {"ledger": {"kind": "memory"}},
            "provider": {"kind": "synthetic"}}
    cfg = Config.load("config/tomtom.yaml", overrides=over)
    c = cfg.build()
    Engine(c).seed_baseline_from_live(when=T0)
    c.store.reset()
    assert c.store.get_baseline() == {}
    moved = Config.load("config/tomtom.yaml", overrides=over)
    moved.raw["grid"]["cell_km"] = 0.25
    c2 = moved.build()
    c2.store = c.store
    Engine(c2)   # must not raise


# -- archive --------------------------------------------------------------

from isogrid.store import SCHEMA_VERSION, SqliteStore  # noqa: E402


def _archive_engine(tmp, cycles=4):
    cfg = Config.load("config/tomtom.yaml", overrides={
        "provider": {"kind": "synthetic"},
        "store": {"path": os.path.join(tmp, "s.db")},
        "budget": {"ledger": {"path": os.path.join(tmp, "b.db")},
                   "quotas": {"synthetic": 100000}}})
    c = cfg.build()
    eng = Engine(c)
    eng.seed_baseline_from_live(when=T0)
    t = T0
    for _ in range(cycles):
        eng.run_cycle(t)
        t += timedelta(seconds=600)
    return c, eng


def test_archive_captures_free_flow_and_distance():
    """Both come free with every TomTom response; dropping them would be
    unrecoverable since you cannot re-request past traffic."""
    with tempfile.TemporaryDirectory() as d:
        c, _ = _archive_engine(d)
        st = c.store.archive_stats()
        assert st["observations"] > 0
        assert st["with_free_flow"] == st["observations"]
        assert st["with_distance"] == st["observations"]


def test_archive_logs_cycles_estimates_and_geometry():
    with tempfile.TemporaryDirectory() as d:
        c, _ = _archive_engine(d, cycles=4)
        st = c.store.archive_stats()
        assert st["cycles_logged"] == 4
        assert st["estimates"] > st["observations"]      # every cell, not just measured
        assert st["cells_known"] == len(c.grid)
        assert st["baseline_snapshots"] > 0


def test_archive_records_failed_cells_individually():
    with tempfile.TemporaryDirectory() as d:
        cfg = Config.load("config/tomtom.yaml", overrides={
            "provider": {"kind": "synthetic", "dead_cells": ["h:2:-1", "h:3:-2"]},
            "store": {"path": os.path.join(d, "s.db")},
            "budget": {"ledger": {"path": os.path.join(d, "b.db")},
                       "quotas": {"synthetic": 100000}}})
        cfg.raw["budget"]["pacer"] = {"kind": "flat", "units_per_cycle": 37}
        c = cfg.build()
        eng = Engine(c)
        eng.seed_baseline_from_live(when=T0)
        eng.run_cycle(T0)
        assert c.store.archive_stats()["failures"] >= 2


def test_export_view_has_derived_training_columns():
    with tempfile.TemporaryDirectory() as d:
        c, _ = _archive_engine(d)
        cols, rows = c.store.export_rows()
        for needed in ("cell_id", "epoch", "seconds", "free_flow_seconds",
                       "distance_m", "delta_s", "ratio", "speed_kmh",
                       "hour_utc", "dow", "lat", "lon", "ring"):
            assert needed in cols, needed
        assert rows
        row = dict(zip(cols, rows[0]))
        assert row["lat"] is not None and row["speed_kmh"] > 0


def test_reset_keeps_the_archive_by_default():
    """A grid change invalidates the baseline, not the record of what the roads
    were actually doing."""
    with tempfile.TemporaryDirectory() as d:
        c, _ = _archive_engine(d)
        before = c.store.archive_stats()["observations"]
        c.store.reset(keep_archive=True)
        assert c.store.get_baseline() == {}
        assert c.store.archive_stats()["observations"] == before
        c.store.reset(keep_archive=False)
        assert c.store.archive_stats()["observations"] == 0


def test_legacy_database_migrates_without_data_loss():
    """An existing v1 file must upgrade in place, not be silently mismatched."""
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "old.db")
        conn = sqlite3.connect(p)
        conn.executescript(
            "CREATE TABLE baseline (cell_id TEXT PRIMARY KEY, seconds REAL NOT NULL,"
            " profile TEXT NOT NULL, updated_at TEXT NOT NULL);"
            "CREATE TABLE observation (cell_id TEXT NOT NULL, cycle INTEGER NOT NULL,"
            " seconds REAL NOT NULL, observed_at TEXT NOT NULL, provider TEXT NOT NULL,"
            " profile TEXT NOT NULL, PRIMARY KEY (cell_id, cycle));"
            "CREATE TABLE kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);"
        )
        conn.execute("INSERT INTO baseline VALUES ('h:0:0',300.0,'live','2026-08-15T12:00:00+00:00')")
        for i in (1, 2, 3):
            conn.execute("INSERT INTO observation VALUES ('h:0:0',?,?, "
                         "'2026-08-15T12:00:00+00:00','tomtom','live')", (i, 400.0 + i))
        conn.commit()
        conn.close()

        s = SqliteStore(path=p)
        assert s.schema_version == SCHEMA_VERSION
        assert s.get_baseline() == {"h:0:0": 300.0}
        assert len(s.history()) == 3
        assert s.next_cycle() == 4
        cols, rows = s.export_rows()
        assert len(rows) == 3


# -- transit --------------------------------------------------------------

from isogrid.providers.otp import OTPProvider  # noqa: E402
from isogrid.providers.synthetic_transit import SyntheticTransitProvider  # noqa: E402


class _StubOTP(OTPProvider):
    """OTP with canned responses in the documented GraphQL shape, including a
    headway-driven wait that varies with departure minute."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def _post(self, url, payload):
        if "feeds" in payload.get("query", ""):
            return {"data": {"feeds": [{"feedId": "MBTA"}]}}
        v = payload["variables"]
        self.calls.append(v)
        if v["modes"] == [{"mode": "WALK"}]:
            return {"data": {"plan": {"itineraries": [
                {"duration": 1500, "walkTime": 1500, "waitingTime": 0,
                 "walkDistance": 1900, "legs": [{"mode": "WALK"}]}]}}}
        wait = (10 - (int(v["time"].split(":")[1]) % 10)) * 60
        return {"data": {"plan": {"itineraries": [
            {"duration": 600 + wait, "walkTime": 420, "waitingTime": wait,
             "walkDistance": 560,
             "legs": [{"mode": "WALK"}, {"mode": "SUBWAY"},
                      {"mode": "WALK"}, {"mode": "BUS"}, {"mode": "WALK"}]}]}}}


def _transit_cfg(**over):
    base = {"provider": {"kind": "synthetic_transit"}, "store": {"kind": "memory"},
            "budget": {"ledger": {"kind": "memory"}}}
    base.update(over)
    return Config.load("config/mbta-transit.yaml", overrides=base)


def test_otp_batches_every_cell_into_one_request():
    """No quota, so there is nothing to split."""
    g = HexGrid(ORIGIN, cell_km=0.42, rings=3)
    batches = _StubOTP().plan(g.cells)
    assert len(batches) == 1
    assert len(batches[0].cells) == len(g)


def test_otp_profiles_the_departure_window():
    """One departure is a snapshot of an arbitrary moment; the window exposes
    the headway."""
    g = HexGrid(ORIGIN, cell_km=0.42, rings=1)
    p = _StubOTP(profile_samples=5, profile_window_minutes=30)
    m = p.fetch(ORIGIN, p.plan(g.cells)[0], "transit", T0)[0]
    assert len(p.calls) == len(g) * 5
    assert m.samples == 5
    assert m.spread_seconds > 0
    assert m.p10_seconds <= m.seconds <= m.p90_seconds


def test_otp_does_not_profile_walking():
    """Walking is deterministic — sampling it is pure waste."""
    g = HexGrid(ORIGIN, cell_km=0.42, rings=1)
    p = _StubOTP(profile_samples=5)
    p.fetch(ORIGIN, p.plan(g.cells)[0], "walk", T0)
    assert len(p.calls) == len(g)


def test_otp_derives_transfers_from_legs():
    """OTP 2.5 has no `transfers` field on Itinerary. Counting transit legs is
    version-proof; reading a schema field is not."""
    from isogrid.providers.otp import ITINERARY_FIELDS, count_transfers
    assert "transfers" not in ITINERARY_FIELDS
    assert count_transfers({"legs": [{"mode": "WALK"}]}) == 0
    assert count_transfers({"legs": [
        {"mode": "WALK"}, {"mode": "SUBWAY"}, {"mode": "WALK"}]}) == 0
    assert count_transfers({"legs": [
        {"mode": "WALK"}, {"mode": "SUBWAY"}, {"mode": "WALK"},
        {"mode": "BUS"}, {"mode": "WALK"}]}) == 1


def test_otp_drops_unknown_fields_instead_of_failing():
    """The GTFS GraphQL schema drifts between OTP versions. An unrecognised
    field should cost a column, not the run."""
    class Rejects(_StubOTP):
        def _post(self, url, payload):
            q = payload.get("query", "")
            if "feeds" in q:
                return {"data": {"feeds": [{"feedId": "MBTA"}]}}
            if "waitingTime" in q:
                return {"errors": [{"message":
                    "Validation error (FieldUndefined@[plan/itineraries/waitingTime]) :"
                    " Field 'waitingTime' in type 'Itinerary' is undefined"}]}
            return {"data": {"plan": {"itineraries": [{
                "duration": 1080, "walkTime": 420, "walkDistance": 560,
                "legs": [{"mode": "WALK"}, {"mode": "SUBWAY"}, {"mode": "WALK"}]}]}}}

    g = HexGrid(ORIGIN, cell_km=0.42, rings=1)
    p = Rejects(profile_samples=2, concurrency=1)
    ms = p.fetch(ORIGIN, p.plan(g.cells)[0], "transit", T0)
    assert p.dropped_fields == ["waitingTime"]
    assert ms and ms[0].seconds == 1080


def test_otp_parses_journey_composition():
    g = HexGrid(ORIGIN, cell_km=0.42, rings=1)
    p = _StubOTP(profile_samples=3)
    m = p.fetch(ORIGIN, p.plan(g.cells)[0], "transit", T0)[0]
    assert m.walk_seconds == 420 and m.transfers == 1
    assert m.wait_seconds > 0


def test_otp_unreachable_cells_are_named_not_inferred():
    class NoRoute(_StubOTP):
        def _post(self, url, payload):
            if "feeds" in payload.get("query", ""):
                return {"data": {"feeds": []}}
            return {"data": {"plan": {"itineraries": []}}}
    g = HexGrid(ORIGIN, cell_km=0.42, rings=1)
    p = NoRoute()
    out = p.fetch(ORIGIN, p.plan(g.cells)[0], "transit", T0)
    assert out == []
    assert set(p.last_unreachable) == {c.id for c in g}


def test_synthetic_transit_travel_time_is_a_step_function_of_departure():
    """The property that makes transit different from driving. If this ever
    goes smooth, the simulator is hiding the bugs that matter.

    Cells off a corridor are walk-only and correctly show no discontinuity, so
    the test looks for one that transit actually serves.
    """
    g = HexGrid(ORIGIN, cell_km=0.42, rings=4)
    p = SyntheticTransitProvider(seed=11, delay_noise_s=0.0)
    served = [
        c for c in g.cells
        if p.journey(ORIGIN, c, T0)["waitingTime"] > 0
    ]
    assert served, "simulator produced no transit-served cells at all"
    biggest = 0.0
    for cell in served:
        times = [
            p.journey(ORIGIN, cell, T0 + timedelta(seconds=30 * i))["duration"]
            for i in range(40)
        ]
        biggest = max(biggest, *[abs(b - a) for a, b in zip(times, times[1:])])
    assert biggest > 120, f"no headway discontinuity present (max jump {biggest:.0f}s)"


def test_cells_beyond_walking_range_of_any_stop_stay_walk_only():
    """Not a bug: a hex with no stop within the egress limit is genuinely
    walk-only, and the map should say so rather than invent a journey."""
    g = HexGrid(ORIGIN, cell_km=0.42, rings=4)
    p = SyntheticTransitProvider(seed=11)
    j = p.journey(ORIGIN, g.cells[-1], T0)
    assert j["waitingTime"] == 0 and j["transfers"] == 0
    assert abs(j["duration"] - p.walk_seconds(ORIGIN, g.cells[-1].center)) < 1


def test_transit_profile_captures_the_headway_spread():
    g = HexGrid(ORIGIN, cell_km=0.42, rings=3)
    p = SyntheticTransitProvider(seed=11, profile_samples=6, profile_window_minutes=30)
    ms = p.fetch(ORIGIN, p.plan(g.cells)[0], "transit", T0)
    spreads = [m.spread_seconds for m in ms if m.spread_seconds]
    assert spreads and max(spreads) > 60


def test_transit_config_measures_every_cell_each_cycle():
    cfg = _transit_cfg()
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    _s, r = eng.run_cycle(T0)
    assert r.cells_measured == len(c.grid)
    assert r.cells_failed == 0


def test_transit_surface_carries_spread_and_composition():
    cfg = _transit_cfg()
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    surface, _ = eng.run_cycle(T0)
    d = surface.as_dict()["cells"]
    withspread = [x for x in d if x["spread_s"] is not None]
    assert len(withspread) == len(c.grid)
    assert all(x["wait_s"] is not None for x in withspread)
    assert any(x["transfers"] is not None for x in withspread)


def test_walk_baseline_makes_transit_advantage_legible():
    """delta = transit - walk. Negative means transit is worth taking."""
    cfg = _transit_cfg()
    c = cfg.build()
    eng = Engine(c)
    eng.bootstrap_baseline(when=T0)
    surface, _ = eng.run_cycle(T0)
    rows = [x for x in surface.cells if x.observed_s and x.baseline_s]
    wins = [x for x in rows if x.delta_s < 0]
    near = [x for x in rows if x.center.haversine_km(c.config.origin) < 0.5]
    assert wins, "transit should beat walking somewhere"
    # Close to the origin, walking should win — no stop is worth the wait.
    assert all(x.delta_s >= 0 for x in near), "transit should not beat walking next door"


def test_transit_archive_persists_profile_columns():
    with tempfile.TemporaryDirectory() as d:
        cfg = _transit_cfg(store={"path": os.path.join(d, "s.db")},
                           budget={"ledger": {"path": os.path.join(d, "b.db")},
                                   "quotas": {"synthetic_transit": 100000}})
        c = cfg.build()
        eng = Engine(c)
        eng.bootstrap_baseline(when=T0)
        eng.run_cycle(T0)
        cols, rows = c.store.export_rows()
        for needed in ("p10_seconds", "p90_seconds", "spread_s", "wait_seconds",
                       "transfers", "samples"):
            assert needed in cols, needed
        row = dict(zip(cols, rows[-1]))
        assert row["samples"] and row["spread_s"] is not None


def test_shipped_configs_can_measure_their_whole_grid():
    """A pacer allowance below the cell count makes the sampler rotate through
    a subset, which looks like a working map while two thirds of it goes
    stale — and silently rebuilds the egress matrix every cycle."""
    for name in ("config/mbta-raptor.yaml", "config/mbta-transit.yaml"):
        cfg = Config.load(name)
        c = cfg.build()
        allow = c.pacer.allowance(c.ledger, c.provider.budget_key, T0,
                                  cfg.cycle_seconds)
        assert allow >= len(c.grid), (
            f"{name}: allowance {allow} < {len(c.grid)} cells")


def test_transit_config_uses_passthrough_not_smooth_reconstruction():
    """A transit surface is piecewise. Smooth reconstruction erases the creases
    at stop watersheds, which are the informative part."""
    cfg = Config.load("config/mbta-transit.yaml")
    assert cfg.raw["reconstruct"]["kind"] == "passthrough"
    assert cfg.raw["engine"]["validation_holdout"] == 0


# -- RAPTOR ---------------------------------------------------------------

from datetime import date as _date  # noqa: E402
from isogrid.transit.gtfs import MAX_TIME, load_gtfs  # noqa: E402
from isogrid.transit.raptor import range_raptor, raptor  # noqa: E402
from isogrid.transit.realtime import encode_trip_updates, parse_trip_updates  # noqa: E402
from isogrid.transit.testfeed import footpaths_for, write_feed  # noqa: E402
from isogrid.providers.raptor import RaptorProvider  # noqa: E402

FEED_DATE = _date(2026, 8, 18)
EIGHT_AM = 8 * 3600


def _feed(tmp):
    path = write_feed(os.path.join(tmp, "test.zip"))
    return load_gtfs(path, on=FEED_DATE), path


def test_gtfs_groups_trips_into_raptor_routes_by_pattern():
    """A RAPTOR route is a stop pattern, not a GTFS route."""
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        assert data.n_stops == 5 and data.n_routes == 2 and data.n_trips == 4
        for r in range(data.n_routes):
            stops = data.route_stops[data.route_stop_ptr[r]:data.route_stop_ptr[r + 1]]
            assert len(stops) == 3


def test_gtfs_time_parsing_survives_past_midnight():
    from isogrid.transit.gtfs import parse_gtfs_time
    assert parse_gtfs_time("08:10:00") == 29400
    # 24:40 is Friday's service day, not Saturday. Dropping it would delete
    # late-night service every night.
    assert parse_gtfs_time("24:40:00") == 88800
    assert parse_gtfs_time("25:13:00") == 90780


def test_raptor_earliest_arrival_matches_hand_computation():
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        res = raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                     footpaths=footpaths_for(data))
        want = {"A": 28800, "B": 29400, "C": 29730, "D": 29580, "E": 30060}
        got = {str(s): int(res.arrival[i]) for i, s in enumerate(data.stop_ids)}
        assert got == want, got


def test_raptor_prefers_transfer_plus_walk_over_a_direct_route():
    """C is 08:20 direct on RED, but 08:15:30 via BLU to D plus a 150 s walk.
    An implementation that stops improving a stop once transit reaches it
    returns the slower answer and looks entirely plausible."""
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        c = data.index_of("C")
        with_fp = raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                         footpaths=footpaths_for(data))
        without = raptor(data, {data.index_of("A"): 0}, EIGHT_AM, footpaths=None)
        assert int(with_fp.arrival[c]) == 29730
        assert int(without.arrival[c]) == 29400 + 600      # direct RED, 08:20
        assert with_fp.arrival[c] < without.arrival[c]


def test_raptor_cannot_board_a_departed_trip():
    """One second late must cost a full headway, not zero."""
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        b = data.index_of("B")
        on_time = raptor(data, {data.index_of("A"): 0}, EIGHT_AM)
        late = raptor(data, {data.index_of("A"): 0}, EIGHT_AM + 1)
        assert int(on_time.arrival[b]) == 29400
        assert int(late.arrival[b]) == 29400 + 1800


def test_range_raptor_exposes_the_headway_as_spread():
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        prof = range_raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                            window_seconds=2400, samples=8,
                            footpaths=footpaths_for(data))
        b = data.index_of("B")
        assert prof.spread()[b] > 600, "30 min headway should show a large spread"
        assert prof.percentile(10)[b] <= prof.median()[b] <= prof.percentile(90)[b]
        assert prof.coverage()[data.index_of("A")] == 1.0


def test_realtime_roundtrip_and_effect_on_routing():
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        buf = encode_trip_updates([("RED_0800", 240, False), ("RED_0830", None, True)])
        td = parse_trip_updates(buf).resolve()
        assert td.delays["RED_0800"] == 240 and "RED_0830" in td.canceled

        b = data.index_of("B")
        base = raptor(data, {data.index_of("A"): 0}, EIGHT_AM)
        late = raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                      delays=td.as_array(data.trip_ids))
        assert int(late.arrival[b]) == int(base.arrival[b]) + 240


def test_cancelling_a_route_removes_it_but_leaves_alternatives():
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        dly = parse_trip_updates(encode_trip_updates(
            [("RED_0800", None, True), ("RED_0830", None, True)])
        ).resolve().as_array(data.trip_ids)
        res = raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                     footpaths=footpaths_for(data), delays=dly)
        assert res.arrival[data.index_of("B")] > 10**6      # RED gone
        assert int(res.arrival[data.index_of("C")]) == 29730  # BLU + walk survives


def test_raptor_provider_costs_one_search_regardless_of_cell_count():
    """The property the whole design exists for."""
    with tempfile.TemporaryDirectory() as d:
        _, feed = _feed(d)
        o = Coord(42.3600, -71.0580)
        p = RaptorProvider(gtfs_path=feed, cache_dir=os.path.join(d, "c"),
                           realtime=False, access_walk_s=600)
        small = HexGrid(o, cell_km=0.35, rings=3)
        large = HexGrid(o, cell_km=0.15, rings=7)
        assert len(p.plan(small.cells)) == 1
        assert p.plan(small.cells)[0].elements == 1
        assert p.plan(large.cells)[0].elements == 1
        when = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
        a = p.fetch(o, p.plan(small.cells)[0], "transit", when)
        b = p.fetch(o, p.plan(large.cells)[0], "transit", when)
        assert len(a) > 0 and len(b) > len(a)


def test_raptor_provider_walk_profile_needs_no_search():
    with tempfile.TemporaryDirectory() as d:
        _, feed = _feed(d)
        o = Coord(42.3600, -71.0580)
        p = RaptorProvider(gtfs_path=feed, cache_dir=os.path.join(d, "c"),
                           realtime=False, access_walk_s=600)
        g = HexGrid(o, cell_km=0.35, rings=2)
        ms = p.fetch(o, p.plan(g.cells)[0], "walk", T0)
        assert len(ms) == len(g)
        assert all(m.wait_seconds == 0 and m.transfers == 0 for m in ms)


def test_min_plus_composition_is_a_lower_envelope():
    """T(p) can never exceed the direct walk, and never beats the best stop."""
    with tempfile.TemporaryDirectory() as d:
        _, feed = _feed(d)
        o = Coord(42.3600, -71.0580)
        p = RaptorProvider(gtfs_path=feed, cache_dir=os.path.join(d, "c"),
                           realtime=False, access_walk_s=600)
        g = HexGrid(o, cell_km=0.35, rings=3)
        when = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
        walk = {m.cell_id: m.seconds for m in p.fetch(o, p.plan(g.cells)[0], "walk", when)}
        transit = p.fetch(o, p.plan(g.cells)[0], "transit", when)
        for m in transit:
            assert m.seconds <= walk[m.cell_id] + 1, f"{m.cell_id} worse than walking"


def test_missing_gtfs_feed_reports_clearly():
    p = RaptorProvider(gtfs_path="/nonexistent/feed.zip", realtime=False)
    g = HexGrid(ORIGIN, cell_km=0.35, rings=1)
    try:
        p.fetch(ORIGIN, p.plan(g.cells)[0], "transit", T0)
        raise AssertionError("should have raised")
    except ProviderError as exc:
        assert "GTFS feed not found" in str(exc)


def test_profile_now_is_not_the_median():
    """The correction that matters. A median over a departure window answers a
    planning question; a live map must answer "if I leave right now"."""
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        prof = range_raptor(data, {data.index_of("A"): 0}, EIGHT_AM - 300,
                            window_seconds=2400, samples=40,
                            footpaths=footpaths_for(data))
        b = data.index_of("B")
        # Departing 07:55 catches the 08:00 RED: 15 minutes.
        assert prof.now()[b] == 900
        # The median over the window is much worse and is experienced at no
        # single moment.
        assert prof.median()[b] > prof.now()[b] + 300


def test_journey_time_is_a_sawtooth_in_departure_time():
    """Falls with slope -1 while the trip is still catchable, then leaps by a
    headway. Flattening that is what the median was doing."""
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        prof = range_raptor(data, {data.index_of("A"): 0}, EIGHT_AM - 300,
                            window_seconds=2400, samples=41,
                            footpaths=footpaths_for(data))
        b = data.index_of("B")
        col = [int(v) for v in prof.journey[:, b] if v < MAX_TIME]
        drops = [a - c for a, c in zip(col, col[1:]) if a > c]
        jumps = [c - a for a, c in zip(col, col[1:]) if c > a]
        assert drops, "should decline while waiting"
        assert jumps and max(jumps) > 1200, "should leap by roughly a headway"


def test_time_to_cliff_reports_the_deadline():
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        prof = range_raptor(data, {data.index_of("A"): 0}, EIGHT_AM - 300,
                            window_seconds=2400, samples=41,
                            footpaths=footpaths_for(data))
        b = data.index_of("B")
        cliff = prof.time_to_cliff()[b]
        # Departing 07:55, the 08:00 RED is catchable for another 5 minutes.
        assert 240 <= cliff <= 420, cliff


def test_raptor_provider_reports_live_value_and_cliff():
    with tempfile.TemporaryDirectory() as d:
        _, feed = _feed(d)
        o = Coord(42.3600, -71.0580)
        p = RaptorProvider(gtfs_path=feed, cache_dir=os.path.join(d, "c"),
                           realtime=False, access_walk_s=600,
                           profile_samples=31, profile_window_minutes=30)
        g = HexGrid(o, cell_km=0.35, rings=3)
        when = datetime(2026, 8, 18, 11, 55, tzinfo=timezone.utc)   # 07:55 local
        ms = p.fetch(o, p.plan(g.cells)[0], "transit", when)
        assert ms
        # seconds must be the live value, which for at least one cell differs
        # from the window median.
        assert any(abs(m.seconds - m.p50_seconds) > 60 for m in ms)
        assert any(m.next_jump_s is not None for m in ms)


def test_absolute_time_predictions_resolve_against_the_schedule():
    """MBTA publishes predicted epochs, not delay offsets. A reader that only
    looks for a `delay` field silently produces a timetable viewer."""
    from datetime import time as _time
    from zoneinfo import ZoneInfo
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        tz = ZoneInfo("America/New_York")
        epoch = datetime.combine(FEED_DATE, _time(0, 0), tzinfo=tz).timestamp()

        def uv(v):
            o = bytearray()
            while True:
                b = v & 0x7F; v >>= 7; o.append(b | (0x80 if v else 0))
                if not v:
                    return bytes(o)
        tg = lambda f, w: uv((f << 3) | w)          # noqa: E731
        ld = lambda f, p: tg(f, 2) + uv(len(p)) + p  # noqa: E731

        # RED_0800 departs B at 08:10 scheduled; predict 08:15.
        stu = tg(1, 0) + uv(2) + ld(4, b"B") + ld(3, tg(2, 0) + uv(int(epoch + 8 * 3600 + 900)))
        buf = ld(2, ld(3, ld(1, ld(1, b"RED_0800")) + ld(2, stu)))

        preds = parse_trip_updates(buf)
        assert preds.trips[0].stops[0].time is not None
        assert preds.trips[0].stops[0].delay is None

        # Without the schedule it cannot resolve, and says so.
        bare = preds.resolve()
        assert bare.delays == {} and bare.unresolved == 1

        got = preds.resolve(data, epoch)
        assert got.delays["RED_0800"] == 300
        assert got.from_time_field == 1 and got.per_stoptime is not None

        b = data.index_of("B")
        base = raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                      footpaths=footpaths_for(data))
        live = raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                      footpaths=footpaths_for(data), delays=got.per_stoptime)
        assert int(live.arrival[b]) == int(base.arrival[b]) + 300


def test_per_stoptime_delays_forward_fill_along_the_trip():
    """GTFS-RT says a delay applies to later stops until superseded."""
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        from isogrid.transit.realtime import (
            FeedPredictions, StopPrediction, TripPrediction,
        )
        tp = TripPrediction(trip_id="RED_0800",
                            stops=[StopPrediction(stop_id="B", delay=180)])
        got = FeedPredictions(trips=[tp], entities=1).resolve(data, 0.0)
        ti = list(map(str, data.trip_ids)).index("RED_0800")
        lo = int(data.trip_time_ptr[ti])
        # A, B, C: the delay at B must carry through to C.
        assert list(got.per_stoptime[lo:lo + 3]) == [180, 180, 180]


def test_egress_cache_key_is_stable_across_processes():
    """`hash()` is salted per process; the cache filename must not be."""
    src = pathlib.Path("isogrid/providers/raptor.py").read_text()
    assert "hashlib.sha1" in src
    assert "abs(hash(key))" not in src


def test_skipped_stops_are_parsed_without_any_timing_event():
    """MBTA sends SKIPPED stops as stop_sequence + stop_id + schedule_relationship
    and nothing else. A reader keyed on time or delay drops them silently and
    routes passengers onto a train that is not stopping."""
    def uv(v):
        o = bytearray()
        while True:
            b = v & 0x7F; v >>= 7; o.append(b | (0x80 if v else 0))
            if not v:
                return bytes(o)
    tg = lambda f, w: uv((f << 3) | w)           # noqa: E731
    ld = lambda f, p: tg(f, 2) + uv(len(p)) + p  # noqa: E731
    stu = tg(1, 0) + uv(2) + ld(4, b"B") + tg(5, 0) + uv(1)
    buf = ld(2, ld(3, ld(1, ld(1, b"RED_0800")) + ld(2, stu)))
    preds = parse_trip_updates(buf)
    sp = preds.trips[0].stops[0]
    assert sp.skipped and sp.time is None and sp.delay is None


def test_skipped_stop_falls_through_to_the_next_serving_trip():
    """Not unreachable — you wait for the train that does stop."""
    from isogrid.transit.realtime import (
        FeedPredictions, StopPrediction, TripPrediction,
    )
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        fp = footpaths_for(data)
        a, b = data.index_of("A"), data.index_of("B")
        base = raptor(data, {a: 0}, EIGHT_AM, footpaths=fp)
        assert int(base.arrival[b]) == 29400

        td = FeedPredictions(trips=[TripPrediction(
            trip_id="RED_0800",
            stops=[StopPrediction(stop_id="B", skipped=True)])],
            entities=1).resolve(data, 0.0)
        assert td.skipped_stops == 1
        got = raptor(data, {a: 0}, EIGHT_AM, footpaths=fp,
                     delays=td.per_stoptime, skipped=td.skipped)
        assert int(got.arrival[b]) == 29400 + 1800     # the 08:30 RED
        # C goes via BLU and a footpath, so it is untouched.
        assert int(got.arrival[data.index_of("C")]) == 29730


def test_stop_skipped_by_every_trip_becomes_unreachable():
    from isogrid.transit.realtime import (
        FeedPredictions, StopPrediction, TripPrediction,
    )
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed(d)
        td = FeedPredictions(trips=[
            TripPrediction(trip_id=t, stops=[StopPrediction(stop_id="B", skipped=True)])
            for t in ("RED_0800", "RED_0830")], entities=2).resolve(data, 0.0)
        got = raptor(data, {data.index_of("A"): 0}, EIGHT_AM,
                     footpaths=footpaths_for(data),
                     delays=td.per_stoptime, skipped=td.skipped)
        assert got.arrival[data.index_of("B")] >= MAX_TIME


# -- pedestrian graph -----------------------------------------------------

from isogrid.transit.pedgraph import (  # noqa: E402
    MAX_TIME as PG_MAX, PedestrianGraph, build_from_geojsonseq,
)


def _grid_with_river(path, lat0=42.36, lon0=-71.06, n=11, river_row=5, bridge_col=5):
    """Street grid at ~100 m spacing, cut by a river with one bridge."""
    dlat, dlon = 100 / 111320.0, 100 / (111320.0 * 0.7373)
    feats = []

    def line(pts, hw="residential"):
        feats.append({"type": "Feature", "properties": {"highway": hw},
                      "geometry": {"type": "LineString", "coordinates": pts}})

    for r in range(n):
        if r == river_row:
            continue
        line([[lon0 + c * dlon, lat0 + r * dlat] for c in range(n)])
    for c in range(n):
        if c == bridge_col:
            line([[lon0 + c * dlon, lat0 + r * dlat] for r in range(n)], "footway")
        else:
            line([[lon0 + c * dlon, lat0 + r * dlat] for r in range(river_row)])
            line([[lon0 + c * dlon, lat0 + r * dlat] for r in range(river_row + 1, n)])
    with open(path, "w") as fh:
        for f in feats:
            fh.write(json.dumps(f) + "\n")
    return lat0, lon0, dlat, dlon


def test_graph_routes_around_barriers_a_straight_line_ignores():
    """The whole reason for this module. Two points 200 m apart across a river
    are a 15 minute walk, not a 3 minute one."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.geojsonseq")
        lat0, lon0, dlat, dlon = _grid_with_river(path)
        g = build_from_geojsonseq(path)
        assert g.n_nodes > 50

        straight = 200 * 1.35 / 1.33          # the straight-line model
        across = g.walk_matrix([lat0 + 4 * dlat], [lon0], [lat0 + 6 * dlat], [lon0],
                               max_seconds=3000)[0, 0]
        along = g.walk_matrix([lat0], [lon0], [lat0 + 2 * dlat], [lon0],
                              max_seconds=3000)[0, 0]
        assert across > 3 * straight, "river should force a long detour"
        assert along < 1.3 * straight, "open grid should roughly agree"


def test_graph_respects_the_distance_limit():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.geojsonseq")
        lat0, lon0, dlat, dlon = _grid_with_river(path)
        g = build_from_geojsonseq(path)
        near = g.walk_matrix([lat0], [lon0], [lat0 + 2 * dlat], [lon0],
                             max_seconds=3000)[0, 0]
        capped = g.walk_matrix([lat0], [lon0], [lat0 + 2 * dlat], [lon0],
                               max_seconds=60)[0, 0]
        assert near < 3000 and capped == PG_MAX


def test_graph_excludes_ways_pedestrians_may_not_use():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.geojsonseq")
        with open(path, "w") as fh:
            for hw, extra_tags in (("motorway", {}), ("residential", {"foot": "no"}),
                                   ("footway", {})):
                fh.write(json.dumps({
                    "type": "Feature",
                    "properties": {"highway": hw, **extra_tags},
                    "geometry": {"type": "LineString",
                                 "coordinates": [[-71.06, 42.36], [-71.059, 42.36]]},
                }) + "\n")
        g = build_from_geojsonseq(path)
        # Only the footway survives: 2 nodes, 1 undirected edge.
        assert g.n_nodes == 2 and g.graph.nnz == 2


def test_graph_survives_a_save_load_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.geojsonseq")
        lat0, lon0, dlat, _ = _grid_with_river(path)
        g = build_from_geojsonseq(path)
        cache = os.path.join(d, "g.npz")
        g.save(cache)
        back = PedestrianGraph.load(cache)
        assert back.n_nodes == g.n_nodes and back.graph.nnz == g.graph.nnz
        a = g.walk_matrix([lat0], [lon0], [lat0 + 4 * dlat], [lon0], max_seconds=3000)
        b = back.walk_matrix([lat0], [lon0], [lat0 + 4 * dlat], [lon0], max_seconds=3000)
        assert a[0, 0] == b[0, 0]


def test_snapping_reports_points_too_far_from_any_way():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.geojsonseq")
        lat0, lon0, _, _ = _grid_with_river(path)
        g = build_from_geojsonseq(path)
        idx, pen = g.snap([lat0, lat0 + 5.0], [lon0, lon0], max_snap_m=250)
        assert idx[0] >= 0 and idx[1] == -1


def test_direct_walk_cap_spans_the_whole_grid():
    """The origin-to-cell walk crosses the entire grid, so it cannot share the
    egress cap. With 900 s applied to a 4.9 km grid every outer cell came back
    as MAX_TIME and rendered as 35,791,394 minutes."""
    from isogrid.providers.raptor import RaptorProvider
    cfg = Config.load("config/mbta-raptor.yaml")
    c = cfg.build()
    origin = cfg.origin
    furthest = max(origin.haversine_km(x.center) for x in c.grid) * 1000
    # Network walking is longer than straight line; 1.6x is a conservative floor.
    needed = furthest * 1.6 / c.provider.walk_speed_ms
    assert c.provider.direct_walk_s > needed, (
        f"direct_walk_s={c.provider.direct_walk_s} cannot span "
        f"{furthest:.0f} m (needs > {needed:.0f} s)")
    assert c.provider.direct_walk_s > c.provider.egress_walk_s


def test_walk_profile_never_emits_the_unreachable_sentinel():
    """MAX_TIME is not a duration. Cells beyond the cap must be reported as
    unreachable, not rendered as 35 million minutes."""
    from isogrid.providers.raptor import RaptorProvider
    from isogrid.transit.gtfs import MAX_TIME as GMAX
    with tempfile.TemporaryDirectory() as d:
        _, feed = _feed(d)
        o = Coord(42.3600, -71.0580)
        p = RaptorProvider(gtfs_path=feed, cache_dir=os.path.join(d, "c"),
                           realtime=False, access_walk_s=600,
                           walk_source="straight_line")
        g = HexGrid(o, cell_km=0.35, rings=3)
        for m in p.fetch(o, p.plan(g.cells)[0], "walk", T0):
            assert m.seconds < GMAX
            assert m.seconds < 86400, "a walk of over a day is a sentinel leak"


# -- official transfers ---------------------------------------------------

def _feed_with_transfers(tmp, rows):
    import zipfile
    path = write_feed(os.path.join(tmp, "t.zip"))
    header = ["from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time"]
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("transfers.txt",
                    "\n".join(",".join(r) for r in [header] + rows) + "\n")
    return load_gtfs(path, on=FEED_DATE), path


def test_official_transfer_time_overrides_geometry():
    """An agency knows how long it takes to cross a station underground; a
    street network does not."""
    from isogrid.transit.footpaths import StraightLineWalk, build_footpaths
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed_with_transfers(d, [["C", "D", "2", "45"]])
        assert data.describe()["official_transfers"] == 1
        fp = build_footpaths(data, max_walk_s=600, source=StraightLineWalk())
        c, dd = data.index_of("C"), data.index_of("D")
        got = [int(fp.times[j]) for j in range(fp.ptr[c], fp.ptr[c + 1])
               if fp.targets[j] == dd]
        assert got == [45], got


def test_transfer_type_3_is_excluded_entirely():
    from isogrid.transit.footpaths import StraightLineWalk, build_footpaths
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed_with_transfers(d, [["C", "D", "3", ""]])
        assert data.describe()["forbidden_transfers"] == 1
        fp = build_footpaths(data, max_walk_s=600, source=StraightLineWalk())
        c, dd = data.index_of("C"), data.index_of("D")
        assert not [j for j in range(fp.ptr[c], fp.ptr[c + 1])
                    if fp.targets[j] == dd]


def test_timed_transfer_costs_nothing():
    """transfer_type 1 means the connecting vehicle waits."""
    from isogrid.transit.footpaths import StraightLineWalk, build_footpaths
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed_with_transfers(d, [["C", "D", "1", ""]])
        fp = build_footpaths(data, max_walk_s=600, source=StraightLineWalk())
        c, dd = data.index_of("C"), data.index_of("D")
        got = [int(fp.times[j]) for j in range(fp.ptr[c], fp.ptr[c + 1])
               if fp.targets[j] == dd]
        assert got == [0]


def test_most_permissive_row_wins_for_a_repeated_pair():
    """Route-qualified rows collapse to one stop-to-stop footpath, so a ban on
    one route must not delete a transfer that another route allows."""
    from isogrid.transit.footpaths import StraightLineWalk, build_footpaths
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed_with_transfers(
            d, [["C", "D", "3", ""], ["C", "D", "2", "90"]])
        fp = build_footpaths(data, max_walk_s=600, source=StraightLineWalk())
        c, dd = data.index_of("C"), data.index_of("D")
        got = [int(fp.times[j]) for j in range(fp.ptr[c], fp.ptr[c + 1])
               if fp.targets[j] == dd]
        assert got == [90], got


def test_walk_caps_are_ordered_sensibly():
    """Transfers are in the hot path, so they stay tightest. Direct spans the
    whole grid, so it is loosest by a wide margin."""
    c = Config.load("config/mbta-raptor.yaml").build().provider
    assert c.transfer_walk_s < c.access_walk_s
    assert c.transfer_walk_s < c.egress_walk_s
    assert c.direct_walk_s > 4 * c.egress_walk_s


def test_declared_transfers_ignore_the_geometric_cap():
    """MBTA publishes transfers up to 545 s against a 450 s cap. The cap bounds
    *candidate* footpaths; a declared transfer is a fact about the network."""
    from isogrid.transit.footpaths import StraightLineWalk, build_footpaths
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed_with_transfers(d, [["C", "D", "2", "545"]])
        fp = build_footpaths(data, max_walk_s=450, source=StraightLineWalk())
        c, dd = data.index_of("C"), data.index_of("D")
        got = [int(fp.times[j]) for j in range(fp.ptr[c], fp.ptr[c + 1])
               if fp.targets[j] == dd]
        assert got == [545], got


def test_declared_transfer_survives_a_stop_with_no_geometric_neighbours():
    """An early exit on an empty candidate list used to skip the official pass
    entirely, silently dropping transfers at isolated stops."""
    from isogrid.transit.footpaths import StraightLineWalk, build_footpaths
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed_with_transfers(d, [["B", "C", "2", "210"]])
        # 100 s bound leaves every stop without a geometric neighbour.
        fp = build_footpaths(data, max_walk_s=100, source=StraightLineWalk())
        b, c = data.index_of("B"), data.index_of("C")
        got = [int(fp.times[j]) for j in range(fp.ptr[b], fp.ptr[b + 1])
               if fp.targets[j] == c]
        assert got == [210], got


def test_in_seat_transfer_types_are_ignored():
    """Types 4 and 5 link trips, not stops. A stop-to-stop footpath would be
    far more permissive than the feed intends."""
    from isogrid.transit.footpaths import StraightLineWalk, build_footpaths
    with tempfile.TemporaryDirectory() as d:
        data, _ = _feed_with_transfers(d, [["C", "D", "4", ""], ["A", "B", "5", ""]])
        assert data.describe()["official_transfers"] == 0


def test_min_walk_time_is_used_when_min_transfer_time_is_absent():
    with tempfile.TemporaryDirectory() as d:
        import zipfile
        path = write_feed(os.path.join(d, "t.zip"))
        with zipfile.ZipFile(path, "a") as zf:
            zf.writestr("transfers.txt",
                        "from_stop_id,to_stop_id,transfer_type,min_transfer_time,"
                        "min_walk_time\nB,C,2,,210\n")
        data = load_gtfs(path, on=FEED_DATE)
        assert list(data.transfer_secs) == [210]


def test_archive_every_throttles_writes_without_throttling_the_map():
    """Display cadence and archive cadence are different questions. At 30 s
    cycles a row per cell per cycle is millions of rows a day."""
    with tempfile.TemporaryDirectory() as d:
        cfg = _transit_cfg(store={"path": os.path.join(d, "s.db")},
                           budget={"ledger": {"path": os.path.join(d, "b.db")},
                                   "quotas": {"synthetic_transit": 999999}})
        cfg.raw["engine"]["archive_every"] = 5
        c = cfg.build()
        eng = Engine(c)
        eng.bootstrap_baseline(when=T0)
        reps = eng.simulate(10, start=T0)
        # every cycle still produces a full surface
        assert all(r.cells_measured == len(c.grid) for r in reps)
        # but only every 5th writes observations
        assert sum(r.archived for r in reps) == 2
        cycles_stored = len(c.store.history(100))
        assert cycles_stored == 2, cycles_stored


def test_conditional_realtime_fetch_reuses_an_unchanged_feed():
    """A 304 must return the previous parse, not raise or return nothing."""
    import urllib.error
    from isogrid.transit.realtime import RealtimeFetcher, encode_trip_updates

    calls = {"n": 0}
    buf = encode_trip_updates([("RED_0800", 120, False)])

    class Stub(RealtimeFetcher):
        def fetch(self):
            calls["n"] += 1
            if calls["n"] == 1:
                out = parse_trip_updates(buf)
                out.fetched_at = 1.0
                self._last = out
                self.fetches += 1
                return out
            self.not_modified += 1
            return self._last

    f = Stub("http://example.invalid/x.pb")
    a, b = f.fetch(), f.fetch()
    assert a is b
    assert f.describe()["not_modified"] == 1
    assert f.describe()["saved_pct"] == 50.0


def test_raptor_config_cycle_fits_inside_its_period():
    """A 30 s cadence only works if a cycle finishes in well under 30 s."""
    cfg = Config.load("config/mbta-raptor.yaml")
    assert cfg.cycle_seconds == 30
    c = cfg.build()
    n = len(c.grid)

    # Constants below are measured, not guessed. Compose, all 31 departure
    # samples, on the reference machine:
    #     1,519 cells 0.04 s | 4,000 0.12 s | 8,000 0.24 s
    #    15,000 cells 0.50 s | 30,000 1.01 s
    # which is ~30 us per cell per cycle, linear. RAPTOR is a flat 2.3 s and
    # does not vary with cell count at all.
    COMPOSE_S_PER_CELL = 3.0e-5
    RAPTOR_S = 2.3
    cycle_s = RAPTOR_S + n * COMPOSE_S_PER_CELL
    assert cycle_s < cfg.cycle_seconds * 0.5, (
        f"{n} cells: {cycle_s:.1f}s of a {cfg.cycle_seconds:.0f}s period")

    # The real ceiling is the browser, not compute. Every refresh ships the
    # whole GeoJSON and MapLibre re-tessellates it on the main thread, which
    # starts to jank past roughly 8 MB. Measured at ~0.56 kB per cell.
    payload_mb = n * 5.6e-4
    assert payload_mb < 8.0, f"{n} cells is {payload_mb:.1f} MB per refresh"
    assert cfg.raw["engine"]["archive_every"] >= 5, (
        "30 s cycles need archive throttling or the store grows by millions "
        "of rows a day")
    assert cfg.raw["store"]["record_estimates"] is False


# -- overlays: routes and vehicles ----------------------------------------

from isogrid.transit.realtime import parse_vehicle_positions  # noqa: E402
from isogrid.transit.shapes import build_route_geojson  # noqa: E402


def _vp_bytes(lat, lon, bearing=315.0, speed=11.2, route=b"Red", status=2,
              occupancy=2, ts=1787000000):
    """A VehiclePositions feed in the documented wire format."""
    import struct

    def uv(v):
        o = bytearray()
        while True:
            b = v & 0x7F; v >>= 7; o.append(b | (0x80 if v else 0))
            if not v:
                return bytes(o)
    tg = lambda f, w: uv((f << 3) | w)             # noqa: E731
    ld = lambda f, p: tg(f, 2) + uv(len(p)) + p    # noqa: E731
    f32 = lambda f, x: tg(f, 5) + struct.pack("<f", x)   # noqa: E731
    pos = f32(1, lat) + f32(2, lon) + f32(3, bearing) + f32(5, speed)
    vd = ld(1, b"y1234") + ld(2, b"3673-3838")
    trip = ld(1, b"12345") + ld(5, route)
    vp = (ld(1, trip) + ld(2, pos) + tg(3, 0) + uv(310) + tg(4, 0) + uv(status)
          + tg(5, 0) + uv(ts) + ld(7, b"place-portr") + ld(8, vd)
          + tg(9, 0) + uv(occupancy))
    return ld(2, ld(4, vp))


def test_vehicle_positions_decode_float_wire_fields():
    """lat/lon/bearing/speed are 32-bit floats, not varints — reading them as
    varints yields a vehicle in the Atlantic."""
    v = parse_vehicle_positions(_vp_bytes(42.3601, -71.0589))[0]
    assert abs(v.lat - 42.3601) < 1e-4 and abs(v.lon + 71.0589) < 1e-4
    assert abs(v.bearing - 315.0) < 0.1
    assert abs(v.speed_ms - 11.2) < 0.1
    assert v.route_id == "Red" and v.label == "3673-3838"
    assert v.status == "in_transit" and v.occupancy == "few_seats"


def test_vehicle_without_a_position_is_skipped():
    """Emitting it at (0,0) would put a train off the coast of Africa."""
    def uv(v):
        o = bytearray()
        while True:
            b = v & 0x7F; v >>= 7; o.append(b | (0x80 if v else 0))
            if not v:
                return bytes(o)
    tg = lambda f, w: uv((f << 3) | w)             # noqa: E731
    ld = lambda f, p: tg(f, 2) + uv(len(p)) + p    # noqa: E731
    vp = ld(1, ld(1, b"12345"))          # trip only, no position
    assert parse_vehicle_positions(ld(2, ld(4, vp))) == []


def test_vehicle_feature_reports_position_age():
    v = parse_vehicle_positions(_vp_bytes(42.36, -71.06, ts=1787000000))[0]
    f = v.as_feature(now=1787000045)
    assert f["properties"]["age_s"] == 45
    assert f["geometry"]["type"] == "Point"


def test_route_shapes_are_filtered_simplified_and_coloured():
    import zipfile
    with tempfile.TemporaryDirectory() as d:
        path = write_feed(os.path.join(d, "t.zip"))
        # A shape near the origin and one 3 degrees away that must be dropped.
        near = [["s1", str(-71.06 + i * 0.001), str(42.36 + i * 0.0005), str(i)]
                for i in range(60)]
        far = [["s2", str(-74.0 + i * 0.001), str(45.0), str(i)] for i in range(20)]
        rows = [["shape_id", "shape_pt_lon", "shape_pt_lat", "shape_pt_sequence"]]
        with zipfile.ZipFile(path, "a") as zf:
            zf.writestr("shapes.txt",
                        "\n".join(",".join(r) for r in rows + near + far) + "\n")
            zf.writestr("trips.txt",
                        "route_id,service_id,trip_id,direction_id,shape_id\n"
                        "RED,WEEKDAY,RED_0800,0,s1\n"
                        "BLU,WEEKDAY,BLU_0805,0,s2\n")
            zf.writestr("routes.txt",
                        "route_id,route_short_name,route_long_name,route_type,"
                        "route_color\nRED,Red,Red Line,1,DA291C\n"
                        "BLU,Blue,Blue Line,1,003DA5\n")
        gj = build_route_geojson(path, 42.36, -71.06, radius_km=8)
        ids = {f["properties"]["route_id"] for f in gj["features"]}
        assert ids == {"RED"}, ids
        f = gj["features"][0]
        assert f["properties"]["color"] == "#DA291C"
        assert f["properties"]["kind"] == "subway"
        # simplification: a near-straight run of 60 points should collapse
        assert len(f["geometry"]["coordinates"]) < 60


def test_route_shapes_fall_back_to_a_colour_when_the_feed_omits_one():
    import zipfile
    with tempfile.TemporaryDirectory() as d:
        path = write_feed(os.path.join(d, "t.zip"))
        pts = [["s1", str(-71.06 + i * 0.001), str(42.36), str(i)] for i in range(5)]
        with zipfile.ZipFile(path, "a") as zf:
            zf.writestr("shapes.txt", "shape_id,shape_pt_lon,shape_pt_lat,"
                        "shape_pt_sequence\n"
                        + "\n".join(",".join(r) for r in pts) + "\n")
            zf.writestr("trips.txt", "route_id,service_id,trip_id,direction_id,"
                        "shape_id\nB1,WEEKDAY,RED_0800,0,s1\n")
            zf.writestr("routes.txt", "route_id,route_short_name,route_long_name,"
                        "route_type,route_color\nB1,1,Bus One,3,\n")
        gj = build_route_geojson(path, 42.36, -71.06, radius_km=8)
        p = gj["features"][0]["properties"]
        assert p["kind"] == "bus" and p["color"] == "#FFC72C"


# -- vehicle track interpolation ------------------------------------------

from isogrid.transit.vehicles import (  # noqa: E402
    Shape, ShapeIndex, VehicleTracker, _local_scale,
)


def _curve_shape():
    """East along a street, then a 90-degree turn north."""
    pts = [(42.3600, -71.0600 + i * 0.0005) for i in range(11)]
    pts += [(42.3600 + i * 0.0004, -71.0550) for i in range(1, 11)]
    lat = np.array([p[0] for p in pts])
    lon = np.array([p[1] for p in pts])
    mx, my = _local_scale(42.36)
    seg = np.hypot(np.diff(lon) * mx, np.diff(lat) * my)
    return Shape(lat, lon, np.concatenate([[0.0], np.cumsum(seg)])), mx, my


def test_path_follows_the_track_around_a_corner():
    """The whole point: a chord between two fixes cuts the corner."""
    sh, mx, my = _curve_shape()
    s0, off0 = sh.project(42.3600, -71.0580)
    s1, off1 = sh.project(42.3620, -71.0550)
    assert off0 < 5 and off1 < 5
    path = sh.slice(s0, s1)
    along = sum(math.hypot((b[0] - a[0]) * mx, (b[1] - a[1]) * my)
                for a, b in zip(path, path[1:]))
    chord = math.hypot((path[-1][0] - path[0][0]) * mx,
                       (path[-1][1] - path[0][1]) * my)
    assert along > chord * 1.15, f"along {along:.0f} vs chord {chord:.0f}"


def test_projection_window_resists_the_parallel_track():
    """Where a route runs beside itself, an unwindowed nearest-point search can
    snap a vehicle onto the opposite direction and teleport it."""
    sh, _, _ = _curve_shape()
    lat2 = np.concatenate([sh.lat, sh.lat[::-1] + 0.00036])
    lon2 = np.concatenate([sh.lon, sh.lon[::-1]])
    mx, my = _local_scale(42.36)
    seg = np.hypot(np.diff(lon2) * mx, np.diff(lat2) * my)
    doubled = Shape(lat2, lon2, np.concatenate([[0.0], np.cumsum(seg)]))
    hinted, _ = doubled.project(42.36018, -71.0580, hint_s=200.0, window_m=300.0)
    assert hinted < sh.length_m, "window should keep it on the outbound leg"


def test_slice_endpoints_land_where_asked():
    sh, mx, my = _curve_shape()
    path = sh.slice(150.0, 500.0)
    a, b = sh._at(150.0), sh._at(500.0)
    assert math.hypot((path[0][0] - a[0]) * mx, (path[0][1] - a[1]) * my) < 1
    assert math.hypot((path[-1][0] - b[0]) * mx, (path[-1][1] - b[1]) * my) < 1


def test_reversed_slice_is_returned_in_travel_order():
    sh, _, _ = _curve_shape()
    fwd = sh.slice(150.0, 500.0)
    rev = sh.slice(500.0, 150.0)
    assert rev[0] == fwd[-1] and rev[-1] == fwd[0]


def test_vehicle_is_published_before_it_has_any_legs():
    """It must still appear, with an empty leg list. Omitting it makes the
    browser discard its playback state, and on its return every leg looks new
    — the vehicle rewinds and replays ground it has already covered."""
    from isogrid.transit.realtime import Vehicle
    tr = VehicleTracker(index=None)
    v = Vehicle(id="v1", lat=42.36, lon=-71.06, trip_id="t1", timestamp=1000)
    tr.update([v], now=1000)
    segs = tr.segments([v], now=1000)
    assert len(segs) == 1 and segs[0]["legs"] == []
    v2 = Vehicle(id="v1", lat=42.361, lon=-71.059, trip_id="t1", timestamp=1020)
    tr.update([v2], now=1020)
    segs = tr.segments([v2], now=1020)
    assert len(segs) == 1
    legs = segs[0]["legs"]
    assert len(legs) == 1 and legs[0]["dur"] == 20
    # No shape index, so it falls back to a straight line and says so.
    assert segs[0]["properties"]["on_track"] is False
    assert len(legs[0]["path"]) == 2


def test_tracker_drops_ghosts():
    """Vehicles that stopped reporting stay in the feed at their last position.
    Animating them would be smoothly moving something that is not there."""
    from isogrid.transit.realtime import Vehicle
    tr = VehicleTracker(index=None, max_age_s=180)
    stale = Vehicle(id="ghost", lat=42.36, lon=-71.06, timestamp=1000)
    tr.update([stale], now=1000 + 600)
    assert "ghost" not in tr.tracks


def test_new_trip_resets_arc_length():
    """A vehicle starting a new trip is on different geometry; carrying the old
    position forward would draw a segment across the city."""
    from isogrid.transit.realtime import Vehicle
    tr = VehicleTracker(index=None)
    for t, trip in ((1000, "t1"), (1020, "t1"), (1040, "t2")):
        tr.update([Vehicle(id="v1", lat=42.36, lon=-71.06, trip_id=trip,
                           timestamp=t)], now=t)
    assert tr.tracks["v1"].key[0] == "t2"
    assert not tr.tracks["v1"].legs


def test_far_off_shape_falls_back_to_a_straight_line():
    """A detour or a stale trip assignment must not teleport the vehicle onto
    the route it is no longer following."""
    from isogrid.transit.realtime import Vehicle
    sh, _, _ = _curve_shape()
    idx = ShapeIndex({"s1": sh}, {"t1": "s1"})
    tr = VehicleTracker(index=idx)
    # ~1 km off the shape.
    for t in (1000, 1020):
        tr.update([Vehicle(id="v1", lat=42.3700, lon=-71.0700, trip_id="t1",
                           timestamp=t)], now=t)
    assert tr.stats["off_shape"] == 1


def _straight_track():
    lat = np.full(101, 42.36)
    lon = -71.06 + np.arange(101) * 0.00012
    mx, my = _local_scale(42.36)
    seg = np.hypot(np.diff(lon) * mx, np.diff(lat) * my)
    sh = Shape(lat, lon, np.concatenate([[0.0], np.cumsum(seg)]))
    return sh, ShapeIndex({"s1": sh}, {"t1": "s1"})


def test_stationary_vehicle_does_not_drift_along_the_track():
    """GPS noise on a stopped train projects to arc-length wander, which reads
    as the marker sliding forward and back along the rails."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    seen = []
    for i, jitter in enumerate((0, 18, -20, 15, -12)):
        q = sh._at(600 + jitter)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           timestamp=base + i * 20)], now=base + i * 20)
        seen.append(tr.tracks["v1"].last_s)
    assert len(set(round(s, 3) for s in seen)) == 1, seen


def test_arc_length_never_moves_backwards():
    """A vehicle must not rewind. Clearing the track on a large backward jump
    made it rebuild from the earlier position and re-traverse ground already
    covered — seen as a teleport back and a replay, and it fired at stations,
    where arriving trains are re-matched to the platform coordinate behind
    their last in-transit position."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    drawn = []
    for i, along in enumerate((0, 300, 600, 350, 700, 900)):
        q = sh._at(min(along, sh.length_m))
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="in_transit", timestamp=base + i * 20)],
                  now=base + i * 20)
        drawn.append(tr.tracks["v1"].last_s)
    assert all(b >= a - 1e-6 for a, b in zip(drawn, drawn[1:])), drawn
    # The single backward fix is absorbed, not replayed.
    assert abs(drawn[3] - drawn[2]) < 1e-6


def test_persistently_backward_fixes_hold_rather_than_re_anchoring():
    """The old code re-anchored after three backward fixes: it cleared the
    track and rebuilt from the new position, which the browser could only
    render as a teleport back followed by a replay of ground already covered.
    Backward fixes turn out to be vanishingly rare once the projection is
    windowed — zero in 144 consecutive live fix pairs — so the clamp simply
    holds, and the machinery that made re-anchoring necessary is gone."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    drawn = []
    for i, along in enumerate((0, 300, 600, 350, 300, 250)):
        q = sh._at(min(along, sh.length_m))
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="in_transit", timestamp=base + i * 20)],
                  now=base + i * 20)
        drawn.append(tr.tracks["v1"].last_s)
    assert all(b >= a - 1e-6 for a, b in zip(drawn, drawn[1:])), drawn
    assert abs(drawn[-1] - drawn[2]) < 1e-6, "held at the furthest point"
    assert tr.stats["reshape"] == 0, "no re-anchor, so nothing to replay"


def test_small_backward_move_is_held_at_the_last_good_position():
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000

    def push(i, along):
        q = sh._at(along)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           timestamp=base + i * 20)], now=base + i * 20)
        return tr.tracks["v1"].last_s

    push(0, 100)
    forward = push(1, 400)             # establishes forward direction
    held = push(2, 380)                # 20 m back: noise, held
    # Projection returns a float; compare with tolerance, not equality.
    assert abs(held - forward) < 1e-6
    still_held = push(3, 50)           # one big jump back is still held
    assert abs(still_held - forward) < 1e-6


def test_stopped_vehicle_still_reaches_the_platform():
    """STOPPED_AT means the vehicle is at a stop, not that it never got there.

    Holding arc length at the *previous* fix — the old behaviour — froze the
    marker wherever it last reported in transit. Measured against the live
    feed, STOPPED_AT is 17% of rapid-transit fixes, and the arc length
    discarded on each is a median of 253 m, p90 534 m. The train stopped a
    quarter-kilometre short of the platform, then jumped past the station on
    departure without ever appearing at it: the missed station, in one line.

    A vehicle whose fixes genuinely are not moving is already covered, by the
    deadband."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    for i, along in enumerate((300, 700, 715)):
        q = sh._at(along)
        tr.update([Vehicle(id="v2", lat=q[1], lon=q[0], trip_id="t1",
                           status="stopped", timestamp=base + i * 20)],
                  now=base + i * 20)
    # It travelled the 400 m to the platform, then held through 15 m of noise.
    assert abs(tr.tracks["v2"].last_s - 700) < 1e-6


def test_timestamp_zero_is_not_treated_as_missing():
    """`v.timestamp or now` silently substitutes wall clock for a zero epoch,
    which then rejects every later fix as out of order."""
    from isogrid.transit.realtime import Vehicle
    _, idx = _straight_track()
    tr = VehicleTracker(index=idx, max_age_s=10**9)
    tr.update([Vehicle(id="v1", lat=42.36, lon=-71.06, trip_id="t1",
                       timestamp=0)], now=0)
    assert tr.tracks["v1"].last_t == 0
    tr.update([Vehicle(id="v1", lat=42.36, lon=-71.0588, trip_id="t1",
                       timestamp=20)], now=20)
    assert tr.tracks["v1"].last_t == 20


def test_mode_filter_keeps_only_the_requested_kinds():
    from isogrid.transit.realtime import Vehicle
    _, idx = _straight_track()
    tr = VehicleTracker(index=idx,
                        route_kinds={"Red": "subway", "1": "bus"},
                        keep_kinds={"subway"})
    base = 1787000000
    tr.update([
        Vehicle(id="train", lat=42.36, lon=-71.06, route_id="Red",
                trip_id="t1", timestamp=base),
        Vehicle(id="bus", lat=42.36, lon=-71.06, route_id="1",
                trip_id="t1", timestamp=base),
    ], now=base)
    assert set(tr.tracks) == {"train"}





def test_mode_filter_spelling_does_not_drop_the_green_line():
    """`vehicle_modes` is written "light rail" in config; ROUTE_KIND, which
    produces the values compared against it, writes "light_rail". Unnormalised
    the two never matched, and every route_type 0 — all four Green Line
    branches and Mattapan — was filtered out of a map whose stated purpose is
    to show rapid transit."""
    from isogrid.transit.realtime import Vehicle
    _, idx = _straight_track()
    tr = VehicleTracker(index=idx,
                        route_kinds={"Green-B": "light_rail", "Red": "subway",
                                     "1": "bus"},
                        keep_kinds={"subway", "light rail"})
    base = 1787000000
    tr.update([Vehicle(id="green", lat=42.36, lon=-71.06, route_id="Green-B",
                       trip_id="t1", timestamp=base),
               Vehicle(id="train", lat=42.36, lon=-71.06, route_id="Red",
                       trip_id="t1", timestamp=base),
               Vehicle(id="bus", lat=42.36, lon=-71.06, route_id="1",
                       trip_id="t1", timestamp=base)], now=base)
    assert set(tr.tracks) == {"green", "train"}


def test_added_trip_resolves_its_shape_by_route_and_direction():
    """MBTA publishes most Green Line runs as ADDED trips, whose trip_id is in
    no static table. Measured live, that is 45 of 72 rapid-transit vehicles and
    every Green Line car among them. Resolved by trip_id alone they all fall
    back to straight chords between fixes — the exact corner-cutting this
    module exists to prevent."""
    from isogrid.transit.realtime import Vehicle
    sh, _ = _straight_track()
    idx = ShapeIndex({"canonical-8000013": sh}, {"scheduled-1": "canonical-8000013"},
                     {("Green-B", "0"): ["canonical-8000013"]})
    tr = VehicleTracker(index=idx)
    base = 1787000000
    for i, along in enumerate((100, 400)):
        q = sh._at(along)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], route_id="Green-B",
                           trip_id="ADDED-1584869373", direction_id=0,
                           timestamp=base + i * 20)], now=base + i * 20)
    assert tr.tracks["v1"].shape_id == "canonical-8000013"
    assert tr.stats["no_shape"] == 0
    assert tr.tracks["v1"].legs[-1]["on_track"] is True


def test_the_wrong_direction_is_not_borrowed_from_another_pattern():
    """Direction is part of the key, so an inbound car cannot be placed on the
    outbound shape."""
    from isogrid.transit.realtime import Vehicle
    sh, _ = _straight_track()
    idx = ShapeIndex({"out": sh}, {}, {("Green-B", "0"): ["out"]})
    tr = VehicleTracker(index=idx)
    q = sh._at(200)
    tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], route_id="Green-B",
                       trip_id="ADDED-1", direction_id=1, timestamp=1787000000)],
              now=1787000000)
    assert tr.tracks["v1"].shape_id is None and tr.stats["no_shape"] == 1


def test_consecutive_legs_join_exactly_where_the_last_one_ended():
    """A gap between one leg's final point and the next leg's first is the
    rubber-band: the browser has nothing to interpolate across it and can only
    jump. It appeared whenever a leg switched between on-track and
    straight-line geometry, because the two endpoints were derived
    independently — 7% of live leg handovers, the worst of them 1.1 km. Legs
    now start from the previous leg's endpoint by construction."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    mx, my = _local_scale(42.36)
    # On track, then a fix ~600 m off it, then back on: the transition that
    # produced every one of the worst measured gaps.
    fixes = [(0, sh._at(100)), (20, sh._at(400)),
             (40, [-71.0530, 42.3655]), (60, sh._at(800)), (80, sh._at(1000))]
    for dt, q in fixes:
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="in_transit", timestamp=base + dt)],
                  now=base + dt)
    legs = list(tr.tracks["v1"].legs)
    assert len(legs) == 4
    assert any(not l["on_track"] for l in legs), "the off-shape case must fire"
    for a, b in zip(legs, legs[1:]):
        p, q = a["path"][-1], b["path"][0]
        gap = math.hypot((q[0] - p[0]) * mx, (q[1] - p[1]) * my)
        assert gap < 0.5, f"{gap:.0f} m gap between legs {a['id']} and {b['id']}"


def test_a_dwelling_vehicle_spends_the_time_instead_of_dropping_the_leg():
    """A held position must still publish a leg. Dropping it leaves the
    browser with nothing to play for those seconds, so playback runs ahead of
    the feed and the queue it is meant to be one fix behind runs dry."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    for i, along in enumerate((400, 405, 410)):
        q = sh._at(along)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="stopped", timestamp=base + i * 30)],
                  now=base + i * 30)
    legs = list(tr.tracks["v1"].legs)
    assert [l["dur"] for l in legs] == [30, 30]
    for l in legs:
        assert l["path"][0] == l["path"][-1], "a dwell leg goes nowhere"


def test_direction_id_is_read_from_the_feed():
    """Without it the route/direction fallback cannot tell inbound from
    outbound, and every ADDED trip would be a coin flip between two shapes."""
    v = parse_vehicle_positions(_vp_bytes(42.36, -71.06))[0]
    assert v.direction_id is None       # absent from this minimal fixture
    assert hasattr(v, "direction_id")


def test_published_legs_never_change_once_emitted():
    """The property the bouncing came from lacking. Rebuilding legs each poll
    let the *past* change under a vehicle midway through playing it, which the
    browser could only render as a jump. Immutability is now structural: a leg
    is built when its closing fix is accepted and never recomputed."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    snapshots = []
    # Irregular intervals as measured live: 47, 15, 12, 43 seconds.
    for dt, along in zip((0, 47, 62, 74, 117), (0, 200, 300, 350, 550)):
        q = sh._at(along)
        v = Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                    status="in_transit", timestamp=base + dt)
        tr.update([v], now=base + dt)
        segs = tr.segments([v], now=base + dt)
        if segs and segs[0]["legs"]:
            snapshots.append([(l["id"], l["dur"], l["path"]) for l in segs[0]["legs"]])

    assert len(snapshots) >= 3
    for a, b in zip(snapshots, snapshots[1:]):
        n = min(len(a), len(b))
        assert a[:n] == b[:n], "an already-published leg changed"
        assert len(b) >= len(a), "legs must only be appended"


def test_leg_duration_is_the_real_gap_between_fixes():
    """Playing a leg over this duration moves the vehicle at the speed it
    actually travelled, rather than at an invented constant."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    for dt, along in ((0, 0), (37, 400)):
        q = sh._at(along)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="in_transit", timestamp=base + dt)],
                  now=base + dt)
    leg = tr.tracks["v1"].legs[-1]
    assert leg["dur"] == 37
    assert leg["id"] == base + 37


def test_leg_ids_increase_so_the_browser_can_deduplicate():
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    for i, along in enumerate((0, 150, 300, 450)):
        q = sh._at(along)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="in_transit", timestamp=base + i * 20)],
                  now=base + i * 20)
    ids = [l["id"] for l in tr.tracks["v1"].legs]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_trip_change_clears_legs_as_well_as_fixes():
    """Otherwise a stale leg would be played across a route the vehicle has
    left."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    for i, along in enumerate((0, 150)):
        q = sh._at(along)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="in_transit", timestamp=base + i * 20)],
                  now=base + i * 20)
    assert tr.tracks["v1"].legs
    q = sh._at(300)
    tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t2",
                       status="in_transit", timestamp=base + 40)], now=base + 40)
    assert not tr.tracks["v1"].legs


def test_leg_paths_never_double_back():
    """Every vertex must advance along the route. `searchsorted` returns the
    first index at or past a value, so including it overshoots the endpoint by
    up to one vertex spacing and the path then reverses to reach the true end.
    Played at 60 fps that is a forward surge and a snap back at the end of
    every leg — the bouncing."""
    sh, _ = _straight_track()
    for s0, s1 in ((0, 200), (150, 500), (333, 777), (500, 150),
                   (0, 9), (900, 987)):
        path = sh.slice(s0, s1, max_points=24)
        arc = [sh.project(p[1], p[0])[0] for p in path]
        deltas = [b - a for a, b in zip(arc, arc[1:])]
        if s1 >= s0:
            assert all(d > -0.5 for d in deltas), (s0, s1, arc[-3:])
        else:
            assert all(d < 0.5 for d in deltas), (s0, s1, arc[-3:])
        # And the endpoints must still be where they were asked for.
        assert abs(arc[0] - s0) < 2 and abs(arc[-1] - s1) < 2


def test_leg_path_endpoints_match_the_fix_positions():
    """Trimming interior vertices must not shorten the leg itself."""
    sh, _ = _straight_track()
    path = sh.slice(120.0, 640.0)
    a, b = sh._at(120.0), sh._at(640.0)
    assert abs(path[0][0] - a[0]) < 1e-9 and abs(path[0][1] - a[1]) < 1e-9
    assert abs(path[-1][0] - b[0]) < 1e-9 and abs(path[-1][1] - b[1]) < 1e-9


# -- station markers ------------------------------------------------------

from isogrid.transit.shapes import build_station_geojson  # noqa: E402


def _feed_with_stations(tmp):
    import zipfile
    path = write_feed(os.path.join(tmp, "t.zip"))
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("stops.txt",
            "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
            "place-alfa,Alpha Station,42.3600,-71.0580,1,\n"
            "A-1,Alpha Platform 1,42.3600,-71.0581,0,place-alfa\n"
            "A-2,Alpha Platform 2,42.3601,-71.0579,0,place-alfa\n"
            "B,Bravo,42.3650,-71.0620,0,\n"
            "FAR,Faraway,43.9000,-70.0000,0,\n"
            "BUS,Bus Stop,42.3655,-71.0625,0,\n")
        zf.writestr("routes.txt",
            "route_id,route_short_name,route_long_name,route_type,route_color\n"
            "RED,Red,Red Line,1,DA291C\n"
            "GRN,Green,Green Line,0,00843D\n"
            "BUS1,1,Bus One,3,FFC72C\n")
        zf.writestr("trips.txt",
            "route_id,service_id,trip_id,direction_id\n"
            "RED,WEEKDAY,RED_0800,0\nGRN,WEEKDAY,GRN_T,0\n"
            "BUS1,WEEKDAY,BUS_T,0\n")
        zf.writestr("stop_times.txt",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "RED_0800,08:00:00,08:00:00,A-1,1\n"
            "RED_0800,08:10:00,08:10:00,B,2\n"
            "RED_0800,08:20:00,08:20:00,FAR,3\n"
            "GRN_T,08:00:00,08:00:00,A-2,1\n"
            "BUS_T,08:00:00,08:00:00,BUS,1\n")
    return path


def test_stations_collapse_platforms_onto_the_parent():
    """Park Street is one marker, not six stacked platforms."""
    with tempfile.TemporaryDirectory() as d:
        gj = build_station_geojson(_feed_with_stations(d), 42.36, -71.058,
                                   radius_km=6)
        ids = {f["id"] for f in gj["features"]}
        assert "place-alfa" in ids
        assert not (ids & {"A-1", "A-2"}), ids


def test_stations_exclude_bus_and_out_of_range_stops():
    with tempfile.TemporaryDirectory() as d:
        gj = build_station_geojson(_feed_with_stations(d), 42.36, -71.058,
                                   radius_km=6)
        ids = {f["id"] for f in gj["features"]}
        assert "BUS" not in ids, "bus stops would bury the map"
        assert "FAR" not in ids, "outside the radius"


def test_interchange_stations_report_every_route():
    """A station served by two lines should say so — it is drawn heavier."""
    with tempfile.TemporaryDirectory() as d:
        gj = build_station_geojson(_feed_with_stations(d), 42.36, -71.058,
                                   radius_km=6)
        alfa = next(f for f in gj["features"] if f["id"] == "place-alfa")
        assert alfa["properties"]["n_routes"] == 2
        assert set(alfa["properties"]["routes"].split(", ")) == {"Red", "Green"}


def test_station_geojson_is_cached():
    with tempfile.TemporaryDirectory() as d:
        feed = _feed_with_stations(d)
        cache = os.path.join(d, "stations.json")
        a = build_station_geojson(feed, 42.36, -71.058, 6, cache=cache)
        assert os.path.isfile(cache)
        b = build_station_geojson(feed, 42.36, -71.058, 6, cache=cache)
        assert a == b


def test_trip_change_still_publishes_the_vehicle():
    """A trip change empties the leg list, and trip changes happen at
    terminals. If that removed the vehicle from the response, the browser
    would forget it and replay on its return."""
    from isogrid.transit.realtime import Vehicle
    sh, idx = _straight_track()
    tr = VehicleTracker(index=idx)
    base = 1787000000
    for i, along in enumerate((0, 150)):
        q = sh._at(along)
        tr.update([Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t1",
                           status="in_transit", timestamp=base + i * 20)],
                  now=base + i * 20)
    assert tr.segments([Vehicle(id="v1", lat=42.36, lon=-71.06, trip_id="t1",
                                timestamp=base + 20)], now=base + 20)[0]["legs"]

    q = sh._at(300)
    v2 = Vehicle(id="v1", lat=q[1], lon=q[0], trip_id="t2",
                 status="in_transit", timestamp=base + 40)
    tr.update([v2], now=base + 40)
    segs = tr.segments([v2], now=base + 40)
    assert len(segs) == 1, "vehicle must stay in the response"
    assert segs[0]["legs"] == [], "but with no legs to replay"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  pass  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
