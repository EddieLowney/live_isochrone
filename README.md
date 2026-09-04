# isogrid

A live travel-time surface for one origin, rendered as a hex grid coloured by
delay against free-flow, built to run inside a hard API quota.

The prototype ships with a **synthetic provider**, so the whole system runs,
simulates, and benchmarks with no network and no API key. Swapping in Mapbox is
a one-line config change.

```
python3 -m isogrid.cli --ephemeral bench --cycles 260

reconstructor   holdout MAE
passthrough       18.9 s
idw               13.3 s
factor            10.3 s  <-- best
```

## Quick start

```bash
pip install numpy pyyaml requests          # scipy/sklearn optional
python3 -m isogrid.cli baseline            # measure the free-flow reference once
python3 -m isogrid.cli serve               # http://127.0.0.1:8000
```

Other commands:

| Command | What it does |
|---|---|
| `preflight` | Two-transaction check of auth, response shape, and budget fit |
| `mask` | Show which cells are excluded and why, before spending anything |
| `baseline` | Sweep the provider's baseline profile once (free-flow for driving, walk-only for transit) |
| `seed` | Derive the baseline from a live sweep where the provider returns free-flow alongside it (TomTom); falls back to `baseline` otherwise |
| `run --cycles N` | Run N cycles in real time |
| `simulate --cycles N` | Fast-forward N cycles in virtual time, free with the synthetic provider |
| `bench` | Compare reconstructors on held-out cells |
| `budget` | What the ledger thinks is left this period |
| `serve` | HTTP API plus the renderer |

Any config value can be overridden without editing the file:

```bash
python3 -m isogrid.cli --set grid.rings=5 --set sampling.kind=volatility simulate
```

## How a cycle works

```
pacer      -> how many billable units this cycle can afford
sampler    -> which cells are worth that budget, in priority order
provider   -> batches them into requests and prices each batch
ledger     -> decrements BEFORE the call; refuses when the cap would break
provider   -> measures
store      -> appends to the observation log
reconstructor -> estimates every unmeasured cell from the measured ones
surface    -> published as JSON and GeoJSON
```

The ordering in the middle is the whole point. Spend is committed before the
request goes out, atomically, and refunded only when a request fails in a way
the provider would not bill. Any other ordering makes the quota advisory.

## Extension points

Each role is an ABC with a name-keyed registry. Add a class, decorate it, name
it in YAML.

| Role | Registry | Ships with |
|---|---|---|
| Grid geometry | `GRIDS` | `hex`, `square` |
| Routing provider | `PROVIDERS` | `synthetic`, `tomtom`, `mapbox` |
| Sampling strategy | `SAMPLERS` | `probe`, `volatility`, `staleness`, `round_robin` |
| Reconstruction | `RECONSTRUCTORS` | `factor`, `idw`, `passthrough` |
| Cell masking | `MASKS` | `polygon`, `quarantine`, `composite`, `none` |
| Persistence | `STORES` | `sqlite`, `memory` |
| Quota ledger | `LEDGERS` | `sqlite`, `memory` |
| Pacing policy | `PACERS` | `proportional`, `flat`, `schedule` |

Adding a TomTom provider, for example:

```python
@PROVIDERS.register("tomtom")
class TomTomProvider(TravelTimeProvider):
    budget_key = "tomtom"

    def plan(self, cells):
        # TomTom bills max(origins, destinations) x 5 once either dimension
        # exceeds 5, so 1xN is punitive. Batch in fives to stay in the cheap
        # branch: one transaction per cell instead of five.
        return [Batch(tuple(cells[i:i+5]), len(cells[i:i+5]))
                for i in range(0, len(cells), 5)]

    def fetch(self, origin, batch, profile, depart_at=None):
        ...
```

The registry passes only the kwargs a class accepts, so one config file can
carry options for several alternatives while you compare them.

## Transit mode

A second, quite different configuration: walk plus public transit, routed by a
local OpenTripPlanner instance against MBTA's GTFS and GTFS-RT feeds.

```bash
cd otp
bash setup.sh
bash run.sh
bash check.sh
python3 -m isogrid.cli --config config/mbta-transit.yaml baseline
python3 -m isogrid.cli --config config/mbta-transit.yaml serve
```

**The rationing machinery goes unused.** OTP runs locally, so every cell is
measured every cycle. `plan()` returns one batch, the pacer allowance exceeds
the cell count, and the sampler is degenerate by design. The ledger is kept
only as a runaway guard against local load — the numbers are not money.

**Walking is not a separate layer.** OTP routes access, transfer and egress
legs over the OSM pedestrian network inside the same search, and returns a
plain walk when transit would be slower. Bolting on a separate pedestrian API
is not possible even in principle: the router has to know walk times to every
candidate boarding stop *before* it can choose where you board.

### Departure-window profiling

Transit travel time is a step function of when you leave. Miss a bus on a
15-minute headway and the answer jumps by 15 minutes. A single query is a
snapshot of one arbitrary moment and would make the map flicker between cycles
for no real reason.

So each cell is queried at several departure times across a window
(`profile_samples` over `profile_window_minutes`) and reported as a
distribution. The median is the headline number; **p90 − p10 is the headway
penalty**, and it is arguably the more useful of the two. A cell reachable in
12 minutes with good timing and 27 without is a materially different place from
one that is reliably 18.

Walking is not profiled — it is deterministic, so sampling it is waste.

### Why the baseline is walking

For driving, the baseline was free-flow and the delta was congestion. For
transit that has no analogue, so the baseline profile is `walk` and the delta
becomes **transit minus walking**. Negative means transit is worth taking. Near
the origin it is positive everywhere, because no stop is worth the wait.

### Why reconstruction is off

`reconstruct: passthrough`, and this is a correctness constraint rather than a
default. A transit surface is piecewise, with creases where it becomes better
to walk to a different stop. Those creases are the most informative feature on
the map and any smooth reconstructor — IDW or the factor model — erases exactly
them. Everything is measured anyway, so there is nothing to reconstruct.

Scaling past a few hundred cells is where that changes. The decomposition is

    T(p) = min( walk(origin, p), min over stops s of [ arrive(s) + walk(s, p) ] )

with `walk(s, p)` static and precomputable, and `arrive(s)` the only part that
moves. That makes the per-cycle update a min-plus matrix product and the
resolution effectively free. Not implemented — at 127 cells the local queries
are cheap enough that it would be premature.

### Overlays: route lines and live vehicles

Neither changes a number the map computes — they explain the numbers. Route
lines say *where* the transit is, so a fast region reads as "the Red Line runs
through here" rather than as an unexplained patch. Vehicles say why a cell just
jumped: you can watch the train pull away.

**Neither needs an API key.** Route geometry and official colours come from
`shapes.txt` and `routes.txt` in the GTFS zip already downloaded for RAPTOR,
which also keeps the drawn lines consistent with the timetable being routed on.
Vehicle positions come from `cdn.mbta.com/realtime/VehiclePositions.pb`, the
same unauthenticated path as TripUpdates. Buses, ferries, Commuter Rail and
subway all appear.

A free key from `api-v3.mbta.com` raises the V3 API limit from 20 to 1,000
requests per minute and unlocks server-sent event streaming. Worth having for
`/live_facilities` later; not needed for either overlay here.

`shapes.txt` covers every bus variant in eastern Massachusetts, so
`build_route_geojson` reduces in three passes: drop shapes outside the radius,
keep the longest few per route and direction (routes carry many variants
differing only at the ends), then Douglas-Peucker each polyline. Built once and
cached to disk, then served with a day-long browser cache.

| Endpoint | Cadence |
|---|---|
| `/api/routes.geojson` | static, cached on disk and in the browser |
| `/api/vehicles.geojson` | live, with a server-side refresh floor |

`vehicles_min_refresh_s` bounds upstream polling regardless of how many browser
tabs are open, and the fetch is conditional, so an unchanged feed costs a 304.
Each vehicle carries `age_s`; markers older than two minutes render faded rather
than as confidently as fresh ones.

### Vehicle interpolation

Positions arrive every 6 s for buses and 21 s for subway — but the interval on
a single vehicle ranges from 8 s to 95 s, and a subway car covers ~204 m
between fixes. Drawn as they arrive, rail markers jump.

The model is deliberately small:

* The server keeps recent fixes and, **when a fix is accepted**, builds the leg
  from the previous fix to it: the stretch of route polyline between them, plus
  the real elapsed time. That leg is then never recomputed.
* The server publishes each vehicle's completed legs in order. It does not
  decide where anything is drawn.
* The browser plays the current leg over its own duration, takes the next from
  its queue, and **halts at the newest fix** until something new arrives.

So a vehicle is exactly one fix behind, which is the most that can be shown
without guessing. There is no playback buffer and no wall-clock playhead.

**Why legs are immutable.** An earlier version rebuilt the geometry on every
poll. That let the *past* change underneath a vehicle midway through playing
it, and the browser could only render the change as a jump — which looked like
bouncing forwards and backwards. Building each leg once, at the moment its
closing fix arrives, makes "the past cannot change" a property of the data
rather than a rule to remember. A test asserts that a published leg is
byte-identical on every later poll.

**Interpolation is in arc length**, not coordinates. A chord between two fixes
cuts corners — measured at 137 m of shortcut on one 90-degree turn.

**Legs join up by construction.** Each leg's path *starts* at the previous
leg's final point rather than at a position projected afresh. Derived
independently, the two disagreed whenever a leg switched between on-track and
straight-line geometry, and the browser has nothing to interpolate across a gap
— it can only jump to the start of the next leg. Measured against the live
feed, 7% of leg handovers had such a gap, the worst of them 1.1 km. It is now
0%, and `diagnose_vehicles.py` reports the figure on every run.

**Finding the shape matters more than the interpolation.** A `trip_id` resolves
through `trips.txt` only if the trip was scheduled, and MBTA publishes most
Green Line runs as `ADDED-*` trips whose id exists nowhere in the static feed —
45 of 72 live rapid-transit vehicles when measured, every Green Line car among
them. Those all fell back to straight chords, which is the corner-cutting the
arc-length machinery exists to prevent. Route plus `direction_id` is always
present, so unscheduled trips resolve to the route's `canonical-` shape for the
direction reported. Where a line genuinely branches, the candidates are
separated by projecting onto each and taking the nearest. Legs on real track
geometry went from 36% to 99%.

**The projection is windowed.** Nearest-point-on-polyline is ambiguous wherever
a route runs beside itself. The search is restricted to a window around the
previous position, sized by how far the vehicle could have travelled since —
one fixed width is either too tight after a long gap (fix intervals reach 54 s
at p90) or wide enough after a short one to catch the opposite track. A match
landing more than 120 m off the shape falls back to a straight line rather than
teleporting the vehicle onto a route it is not following.

**Correction is one comparison.** GTFS draws each shape in the direction of
travel and MBTA gives inbound and outbound distinct `shape_id`s — verified
against every rapid-transit shape in the feed. Arc length is therefore
monotonic, so `s - last_s < 25 m → hold` absorbs both a bad re-match and
receiver noise on a stationary car, and there is no direction sign to infer.

An earlier version inferred a direction sign per vehicle, counted consecutive
backward fixes, and re-anchored after three of them by clearing the track. That
re-anchor was itself a teleport followed by a replay of ground already covered,
and a single noisy fix could lock the sign inverted for a whole run. On live
data the machinery fired **zero times in 144 consecutive fix pairs** — there
were no backward fixes at all once the projection was windowed. It is gone.

`STOPPED_AT` no longer freezes the marker. Holding arc length at the *previous*
fix stopped the train wherever it last reported in transit: 17% of
rapid-transit fixes, discarding a median of 253 m and a p90 of 534 m of real
travel. The train halted a quarter-kilometre short of the platform, then jumped
past the station on departure without ever appearing at it — the missed
station. A vehicle that genuinely is not moving is already covered by the 25 m
deadband.

**Known residual.** About 0.8% of legs imply a speed above 35 m/s, the worst
seen being 387 m over 6 s, caused by irregular feed timestamps rather than by
the projection. The marker stays on the track and glides fast rather than
teleporting, so it is left uncorrected; clamping it would mean stretching leg
durations, which desynchronises playback from the feed and grows the queue.

| Setting | Default | Why |
|---|---|---|
| `vehicles_min_refresh_s` | 5 | below every measured fix interval |
| `vehicle_max_age_s` | 180 | past this a vehicle is not reporting |
| `vehicle_modes` | subway, light rail | buses move 37 m per fix, under one cell |

Mode names are matched with spaces and underscores treated alike. They were
not: `ROUTE_KIND` produces `light_rail`, config says `light rail`, nothing
matched, and every `route_type` 0 — all four Green Line branches and Mattapan,
45 of 116 rapid-transit vehicles — was silently absent from the map.

### The continuous field

The hex fill asserts a step change at every cell boundary, and that step is an
artefact of the sampling lattice rather than anything in the city. Cell centres
are 44.2 m apart, and the pedestrian graph they were measured on has nodes
10-40 m apart — the config already notes that neighbouring cells increasingly
snap to the same node. Linear interpolation between two adjacent *measured*
centres therefore claims strictly less than the step does.

This is a rendering change, not a reconstruction. `reconstruct: passthrough` is
about refusing to invent values where none were measured, and that still holds:
every cell in the surface is measured. Nothing below adds a data point.

**The mesh is free.** With the axial layout in `grid.py`, `{(q,r), (q+1,r),
(q,r+1)}` and `{(q,r), (q+1,r-1), (q+1,r)}` are both equilateral with side
`cell_km·√3` — two triangles per cell, ~7,450 of them, no Delaunay pass. Each is
shaded barycentrically from its three vertices.

Two rules keep it honest:

* **The scalar is interpolated, never the colour.** Blending RGB through a ramp
  produces colours the ramp maps to no value at all. On the diverging `delta`
  ramp, mixing a strong negative with a strong positive lands on the pale
  midpoint and asserts the opposite of both inputs.
* **A measured value is never altered, and unreachable ground is never drawn
  at full opacity.** Nothing is extrapolated past the measured hull, and the
  field ends half a cell inside the hex footprint because the outermost centres
  are its boundary.

**Unreachable ground fades rather than being cut out.** Leaving the 116 masked
cells transparent was correct and looked wrong: the surface ended in hard
hexagonal edges, and larger regions became dark angular patches punched through
it — a hex lattice reasserting itself in the one layer built to not have one.
Instead, unreachable cells are filled by breadth-first diffusion outward from
the measured boundary, each ring averaging the ring inside it, with opacity
falling linearly to zero over `FIELD_FADE_RINGS` (4 cells, ~175 m). The surface
dissolves into the Charles instead of being cut by it.

This is the layer's one deliberate inaccuracy, and it is bounded three ways: it
only ever touches cells that carry no measurement, it never modifies a measured
value, and it is never drawn at an opacity that could pass for measured.
`verifyField()` asserts the last of those — `holesAtFullAlpha` must be 0. A
one-ring fade was tried first and was not enough; the interiors of the larger
regions still read as hard-edged patches. With four rings, 6 triangles out of
7,776 remain undrawn, all of them deep water.

The grid's own outer boundary is deliberately *not* faded. It marks where the
measured data ends, which is information worth keeping crisp.

**Creases are not washed out, and this was measured rather than assumed.**
Across 11,345 mesh edges the median adjacent-cell gap is 0.53 min, p99 is
3.53 min and the largest is 7.72 min. The ramp spans 18 minutes over nine
stops, so a 7.7 min crease swings four ramp stops across 44 m — a steep
gradient, not a smoothed one. 4.6% of edges exceed `cliff_threshold_s`. An
edge-preserving guard (render a triangle nearest-neighbour where the gap
exceeds the threshold) was considered and left out: the numbers do not justify
it. Conversely the *median* gap of 32 s is largely node-snapping noise, and not
drawing a hard edge around each sample of it is a fair reading of the data.

**Verification is runnable, not asserted.** A cell centre is a mesh vertex, so
interpolation there must return that cell's own measurement. `verifyField()` in
the browser console samples the texture at all 3,997 centres and compares:

```
cellsChecked 3810      exactMatch 3384 (88.8%)   worstChannelDelta 3/255
unreachableCells 116   worstHoleAlpha 192/255    holesAtFullAlpha 0
trianglesMeasured 7454 trianglesFeathered 316    trianglesDropped 6
```

`exactPct` below 100 is texel rounding — a centre lands up to half a texel from
the exact vertex — bounded at 3/255, which is sub-perceptual. The measured
cells' figures are unchanged by the fade, which is the point: it reaches into
unmeasured ground only. `holesAtFullAlpha` must be 0. The `Field`/`Cells`
toggle makes smooth-versus-raw a one-click comparison for the same reason.

**Implementation.** MapLibre cannot shade a polygon per vertex — `fill-color`
is flat and `line-gradient` applies only to lines — so the mesh is rasterised
into an offscreen canvas and supplied as an `image` source. The grid never
moves, so this runs once per data update (~95 ms at 2048 px wide, ~1.6 m/px)
and MapLibre resamples on pan and zoom for free. A `canvas` source would avoid
the PNG encode, but with `animate: false` MapLibre does not pick up redraws,
and `animate: true` re-uploads the whole texture every frame.

Rebuilds are coalesced over 150 ms, because `updateImage` aborts any load still
in flight and MapLibre reports that as a map error. A lone poll never trips it;
a mode switch landing in the same tick as one does. Layer visibility is applied
outside the debounce so the `Field`/`Cells` toggle stays instant. An `AbortError`
is no longer surfaced in the UI at all — a load cancelled because something
newer replaced it is the intended behaviour, not a fault.

Currently applied to **travel time only**. The other three modes keep the hex
fill: their diverging ramps make the scalar-versus-colour rule load-bearing in
ways worth verifying separately, and `cliff` uses null to mean "no cliff in the
window" — real information, not a missing value, and not something to
interpolate through.

### Render modes

| Mode | Shows |
|---|---|
| Travel time | Absolute median journey time |
| vs walking | Transit minus walking; negative means transit wins |
| Reliability | p90 − p10, the headway penalty |

The transit panel also reports how many cells transit actually beats walking
in, plus median wait and transfer count.

### RAPTOR: transit without OTP in the loop

`config/mbta-raptor.yaml` replaces the OTP provider with a built-in RAPTOR
implementation reading the same MBTA GTFS and GTFS-RT feeds.

The reason is structural. OTP's `plan` runs a search that computes arrival at
*every* stop, then discards all but the one destination asked about — so an
isochrone through it pays for the same search once per cell. RAPTOR keeps the
whole result:

| | OTP provider | RAPTOR provider |
|---|---|---|
| Per cycle | cells x departures HTTP queries | 1 search + 1 GTFS-RT fetch |
| 547 cells | ~2700 queries | 1 search |
| Cost of more cells | linear | flat |
| Departure profiling | 1 query per sample | nearly free |

Hence the shipped config runs 547 cells at 346 m rather than 127 at 727 m: four
times the resolution for the same work.

**Layers, and how often each is paid for**

| Layer | Frequency |
|---|---|
| GTFS parse into RAPTOR arrays | once per service day, cached to `.npz` |
| Footpaths, access, egress matrix | once per grid and walk source, cached |
| GTFS-RT trip updates | once per cycle, one HTTP request |
| range-RAPTOR + min-plus compose | once per cycle, milliseconds |

**Walk source.** Three options, and the default is now `graph`.

| Source | Accuracy | Cost for N sources x M points |
|---|---|---|
| `straight_line` | ~4x optimistic at barriers | instant |
| `otp` | network-accurate | one HTTP request per pair |
| `graph` | network-accurate | one bounded Dijkstra per source |

The straight-line error is not noise around a mean. On open street grid a 1.35
detour factor is close to right; across the Charles it is optimistic by a
factor of four. The error is structured and concentrated exactly where a
Boston map most needs to be correct.

`graph` builds a pedestrian network from the OSM extract `otp/setup.sh`
already cropped, using the `osmium` CLI that setup also requires — no geopandas
stack. Nodes are identified by rounded coordinates, since the GeoJSON export
drops OSM ids; 7 decimal places is about a centimetre, well below the scale at
which two ways genuinely share a node.

What makes it fast is the distance limit: capping at a 15-minute walk touches
roughly 2% of a metro-scale graph, so a source costs ~12 ms rather than ~56 ms.
The 1,437 stops within reach of a 5 km grid take about 17 seconds in total.
Sources are chunked at 50 because scipy returns a dense (sources x nodes)
array regardless of how few columns are wanted — 1,000 sources over 300k nodes
would be 2.4 GB.

The graph is the cached artifact, not the walk matrix. It is origin-, grid- and
resolution-independent, so it survives every change except an OSM refresh.
Matrices derived from it are cheap and disposable.

**Realtime.** `TripUpdates.pb` is parsed by a small wire-format reader rather
than `gtfs-realtime-bindings`, to avoid a protobuf runtime dependency on the
one component that has to keep working unattended. Delays shift trip times;
cancellations are pushed beyond reach rather than deleted. A failed fetch falls
back to the timetable and records the error — never silently, since
schedule-only output looks identical to working.

**The headline number is "leaving now", not a median.** Journey time as a
function of departure is a sawtooth: it falls with slope -1 while a departure
is still catchable, then leaps by roughly a headway the moment it is not.
Reporting the median across a departure window answers a *planning* question —
"if I leave sometime in the next half hour, typically how long" — and erases
the temporal edge this map exists to show. Someone who has just missed a bus is
genuinely fifteen minutes worse off. It also makes the map nearly static
between cycles, which would look like evidence that a live map is unnecessary
when the fault is in the aggregation.

So `seconds` is the value at the *first* sampled departure. The percentiles are
kept as a reliability layer: p10 is best-case timing, p90 worst-case.

**`next_jump_s` — the deadline.** Seconds until the journey time jumps past
`cliff_threshold_s`, computed on the composed per-cell curve rather than on
stop arrivals, since a cell served by two stops on different headways has a
cliff neither stop has alone. Rendered as the **Leave by** mode, red meaning
go now. `null` means no cliff in the window — either service is frequent
enough that timing barely matters, or the cell is walk-dominated.

**Why range-RAPTOR still.** Sweeping departures newest-first lets each run
reuse the previous one's labels, so 31 samples cost little more than one
search. That is what makes fine sampling affordable — and fine sampling is
required, because 6 samples over 30 minutes aliases a 10-minute headway into
noise. In the OTP provider each sample is a separate HTTP query, which is why
it profiles coarsely.

### Offline testing

`synthetic_transit` models radial lines with real headways, so the entire
pipeline — profiling, spread, the walk-vs-transit comparison — runs without
Java, without a graph, and without network. It reproduces the departure-time
discontinuity deliberately; a smooth simulator would hide every bug worth
catching.

```bash
python3 -m isogrid.cli --config config/mbta-transit.yaml \
  --provider synthetic_transit --ephemeral run --cycles 1
```

## The archive

The running system only reads the last few hundred cycles. A model trained later
needs everything, and nothing can be backfilled — you cannot re-request last
Tuesday's traffic. The schema therefore keeps more than the live system uses.

| Table | Holds | Why it would otherwise be lost |
|---|---|---|
| `observation` | every measurement, with free-flow and route distance | free-flow and distance arrive free with each TomTom call and were previously discarded |
| `failure` | cells requested that returned nothing | only a count survived, not which cells or when |
| `cycle_log` | allowance, spend, holdout error, refits | lived only in the transient surface, overwritten each cycle |
| `baseline_history` | append-only free-flow snapshots | the live baseline is an EWMA that overwrites itself |
| `cell_geometry`, `run_meta` | where each cell was, keyed by grid fingerprint | axial ids get reused when the grid moves, so exports would be unmappable |
| `estimate` | what the reconstructor predicted | lets a future model be scored against what was actually deployed |

Distance matters more than it looks. Duration is not comparable across cells,
but distance over duration is a speed, and speed is what generalises.

```bash
python3 -m isogrid.cli --config config/tomtom.yaml archive
python3 -m isogrid.cli --config config/tomtom.yaml export --out obs.csv
python3 -m isogrid.cli --config config/tomtom.yaml export --format parquet --out obs.parquet
```

Export emits the `observation_features` view: raw seconds, free-flow, distance,
derived delta / ratio / speed, hour and weekday, and the cell's coordinates.
Derived columns live in a SQL view rather than in storage, so redefining one
does not require rewriting history.

Volume is modest — roughly 2,000 observations a day at the shipped cadence, a
few hundred MB per year including estimates.

`seed --reset` now clears the baseline but **keeps** the archive; the roads were
doing what they were doing regardless of how the area was tiled. Pass
`--wipe-archive` to delete measurements too.

Schema changes go through `_migrate`, and the version is stored in the file. A
v1 database is rebuilt in place with its measurements carried across.

## The basemap

MapLibre GL JS over TomTom raster tiles, with the hex grid sandwiched between
the base tiles and a transparent label overlay. TomTom's `labels` layer carries
the same labels as the standard map, precisely aligned, so street names render
*above* the colour and stay readable through it. That ordering is the difference
between an overlay and a blanket.

Raster rather than vector because the raster endpoint is fully documented and
needs no style JSON. MapLibre handles raster sources natively, and the GeoJSON
machinery — which is what makes the grid and any future overlays work — is
identical either way.

**Tiles are metered separately.** TomTom's free tier is 2,500 non-tile requests
per day *plus* 50,000 tile requests. `budget.quotas.tomtom_tiles` guards the
second allowance through the same ledger, so the basemap can never eat the
routing budget.

**The key never reaches the browser.** `/tiles/{layer}/{style}/{z}/{x}/{y}.png`
is proxied server-side, injecting the key from the environment, and cached to
disk under `render.basemap.cache_dir`. A metro area at one zoom is a few dozen
tiles, so after the first load nearly every request is a cache hit. Failed
fetches refund the ledger and serve a transparent tile, so a network blip
degrades the basemap rather than breaking the page.

### Adding roads or transit

The renderer has an `OVERLAYS` array near the top of `web/index.html`. Each
entry becomes a MapLibre source plus layers, refreshed on the same poll as the
grid:

```js
const OVERLAYS = [{
  id: 'roads',
  url: '/api/roads.geojson',
  before: 'labels',              // draw beneath the label tiles
  layers: [{ id: 'roads-line', type: 'line',
             paint: { 'line-color': ['get','_color'], 'line-width': 2.5 } }],
  colour: (f, mode, cfg) => rampColour(mode, normalise(mode, f.properties, cfg)),
}];
```

Serve the geometry from a new endpoint and add the entry. Nothing else changes.
The hard part of road-segment colouring was never the rendering — it is deciding
which measured travel time belongs to which segment.

## Masking: not paying for water

A hex in Boston Harbour returns nothing and consumes a transaction every cycle,
forever. Two masks handle it, and they fail in opposite directions, so use both.

**`polygon`** takes geography you supply and excludes cells before the first
request. It samples the cell's vertices as well as its centre and keeps the cell
only if `min_land_fraction` of those points are on land — a hex straddling a
shoreline with its centre just offshore still contains routable roads, and
testing one point would throw it away. Free sources:

- **US Census TIGER/Line AREAWATER** — per county, authoritative, the right
  choice for a US metro. Suffolk County covers the Harbour and the Charles.
- **OpenStreetMap water polygons** from `osmdata.openstreetmap.de` — global
  coastline plus inland water.
- **Natural Earth** — global, and far too coarse for a city. It will call the
  Charles dry land.

GeoJSON is read with no dependencies. Shapefiles and GeoPackages go through
geopandas if installed, reprojected to EPSG:4326 automatically. Shapely is used
for an STRtree index when present; the fallback is ray casting with a
bounding-box pre-filter, which is fast enough for a few hundred cells.

Setting `mode: include` inverts it, turning the same mask into a study-area
boundary.

**`quarantine`** learns from the provider instead. A cell requested but never
returned is unroutable for some reason, and the engine detects that
generically — no provider-specific error parsing. After three consecutive
failures the cell is excluded, then re-probed after a backoff that doubles each
time, so a transient outage does not permanently delete it and a single success
clears the record. It costs a few transactions per discovery, once, versus one
per cycle forever.

Masked cells render as hatched holes rather than estimates. Claiming a travel
time for open water would be worse than showing a gap.

```bash
python3 -m isogrid.cli --config config/tomtom.yaml mask
```

## The three algorithms worth understanding

**Sampling** decides where the money goes. `probe` keeps a fixed
high-information set measured every cycle and rotates the leftover budget
through everything else. `volatility` ranks by sigma x staleness, which is
Neyman allocation — under a fixed budget, sampling frequency proportional to a
cell's delta volatility minimises total squared staleness.

`max_probe_fraction` is not a tuning knob, it is a correctness constraint. Set
the probe count equal to the allowance and the rotation starves, no non-probe
cell is ever measured, the factor model has no history for the cells it is
supposed to predict, and it silently underperforms plain spatial interpolation.
That failure is covered by `test_factor_model_beats_spatial_interpolation`.

**Reconstruction** works in ratio space — live seconds over baseline seconds —
which removes the dominant spatial gradient and leaves the congestion signal,
the part that is actually correlated across cells. `factor` runs PCA on the
historical ratio surface, then each cycle regresses the observed ratios on the
components restricted to observed cells to recover factor scores, and projects
back out to the full grid. On the synthetic corridors the first component
carries roughly 80% of variance.

**Pacing** converts remaining period budget into a per-cycle allowance.
`proportional` divides what is left by the cycles remaining, so underspending
one hour quietly raises every later allowance. `schedule` weights by hour of
day, which matters because delta variance overnight is near zero and uniform
cadence spends most of the quota confirming nothing is happening.

## Validation

Every cycle holds out a few measured cells, excludes them from the
reconstructor's input, and scores the estimate against ground truth. It works
against any provider, not just the synthetic one, and it is the number to watch
when changing anything — reported as `holdout_mae_s` on each cycle and
surfaced in the renderer.

```bash
python3 tests/test_isogrid.py     # 19 tests, no pytest required
```

The budget tests include a threaded race against the sqlite ledger, because a
quota that is merely usually respected is not a quota.

## Going live with TomTom

The recommended first provider: the free tier needs no credit card, and TomTom
returns `trafficDelayInSeconds` with every live measurement, so free-flow
arrives in the same request as the live value and the delta carries no
cross-call drift.

```bash
export TOMTOM_API_KEY=...
python3 -m isogrid.cli --config config/tomtom.yaml preflight   # ~2 transactions
python3 -m isogrid.cli --config config/tomtom.yaml seed        # 37 transactions
python3 -m isogrid.cli --config config/tomtom.yaml serve
```

**Batch size is a billing constraint, not a tuning knob.** TomTom charges on
matrix dimensions: at most five in both directions costs origins x
destinations, otherwise it costs the larger dimension times five. A single
1 x 37 request bills 185 transactions for 37 answers; eight batches of five
bill 37. Raising `batch_size` above 5 multiplies the bill by five for identical
data, which is why the provider clamps it.

**The quota is daily, so `budget.period` must be `daily`.** The shipped config
plans 15 transactions per cycle at a 10-minute cadence: 2,160/day against 2,250
usable after the 0.9 safety factor. `test_tomtom_config_fits_its_daily_quota`
holds that invariant.

Headroom matters here more than with Mapbox, because TomTom counts
`NO_ROUTE_FOUND` and `MAP_MATCHING_FAILURE` cells against your transaction
quota. Note that "billed" throughout TomTom's docs means "counted against the
daily allowance", not "charged" — the free tier has no payment method attached,
and exceeding the quota returns HTTP 403/429 rather than generating a bill. A
cell in water therefore wastes one of 2,500 daily transactions and returns
nothing. The provider counts them in `describe()['billed_failures']`; if that
climbs, move or drop the offending cells so the allowance buys real
measurements.

What the ledger protects here is uptime, not spend. Exhausting the daily quota
means the map stops updating until midnight UTC, and TomTom's terms allow
suspension for repeatedly exceeding plan limits.

## Going live with Mapbox

```yaml
provider:
  kind: mapbox
  token_env: MAPBOX_TOKEN
  max_coords_per_request: 25
```

Four things to check before you do:

1. **Coordinate cap on `driving-traffic`.** Some accounts see 10 rather than 25
   until Mapbox raises it. Billing is per element either way, so this changes
   request count, not cost.
2. **Terms on storing derived results.** The baseline is a stored derivative of
   Mapbox data. The Isochrone endpoint carries an explicit requirement to
   display results on a Mapbox map; the Matrix docs do not state the same
   restriction, but read the terms rather than assuming.
3. **Baseline from the same engine.** `bootstrap_baseline` deliberately uses the
   same provider on the `driving` profile. A baseline from a self-hosted engine
   would inject a spatial bias into every delta that never cancels.
4. **Set the real quota.** `budget.quotas.mapbox` defaults to 100000 with a 0.9
   safety factor. Headroom covers retries and unroutable-but-billed cells.

## Scaling up

At `rings: 3` this is 37 cells and nothing is under pressure. The knobs that
matter as it grows:

- `grid.rings` — cell count is `3n(n+1)+1`, so rings 8 is 217 cells
- `budget.pacer` — switch to `schedule` and weight the rush windows
- `sampling.n_probes` — keep it well under the per-cycle allowance
- `reconstruct.n_components` — refit and watch `explained_variance_ratio`;
  add components only while they earn their place on holdout MAE

Move the scheduler out of the web process before running this for real. The
sqlite ledger is already safe across processes; the in-thread loop in
`server.py` is a prototype convenience.

## Known limitations

- The synthetic provider's congestion is smoother than reality, so holdout MAE
  here is optimistic. Re-benchmark against real data before trusting the
  reconstructor's numbers.
- No multi-provider support. The provider partition and cross-provider
  calibration discussed earlier are deliberately out of scope; `budget_key`
  already separates ledgers, which is the piece that would need to exist.
- The Mapbox adapter is untested against the live API — written without network
  access. Verify the response shape on first run.
- The renderer projects with a local equirectangular approximation. Fine for a
  metro area, wrong at continental scale.
