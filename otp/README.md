# OpenTripPlanner setup

Local transit router for the walk + public transit map. Everything here is
free and unmetered — OTP runs on your machine against MBTA's published feeds.

## Prerequisites

- **Java 17+** (`brew install openjdk@21`)
- **~8 GB free disk**, **~6 GB RAM** for the build
- `osmium-tool` is optional but halves the build (`brew install osmium-tool`)

## Setup

```bash
cd otp
bash setup.sh
bash run.sh       # leave this one running
bash check.sh     # from another terminal
```

Invoking through `bash` rather than `./setup.sh` sidesteps the executable bit,
which some download and extraction paths drop. `chmod +x *.sh` if you prefer
the shorter form.

`check.sh` is the important one. It confirms three separate things: OTP is up,
the MBTA feed is loaded, and **GTFS-RT is actually being consumed**. That last
check matters because OTP starts happily with broken updaters and serves
schedule-only results that look completely normal. If it reports `0/N
departures carry realtime`, the map is showing timetables, not live conditions.

## What gets downloaded

| File | Source | Size |
|---|---|---|
| `otp-2.5.0-shaded.jar` | Maven Central | ~120 MB |
| `mbta.gtfs.zip` | `cdn.mbta.com/MBTA_GTFS.zip` | ~80 MB |
| `massachusetts-latest.osm.pbf` | Geofabrik | ~250 MB |
| `graph.obj` | built locally | ~1-2 GB |

MBTA updates its GTFS several times a month. Re-run `setup.sh` after
`rm graph.obj` every few weeks; GTFS-RT trip updates reference trip IDs from
the static feed, so a stale graph quietly loses realtime coverage as IDs rotate.

## The realtime updaters

`router-config.json` polls three feeds every 30-60 seconds:

- `TripUpdates.pb` — delays and cancellations. **This is the one that changes
  the map.** Without it you have a timetable.
- `VehiclePositions.pb` — where vehicles are. Not used in routing, but useful
  if you later want live vehicles on the map.
- `Alerts.pb` — service disruptions.

These are plain static files with no key and no rate limit. Verify the URLs
against MBTA's GTFS-Realtime page if a feed 404s — they are the documented
paths but the site is the authority.

Use of MBTA data is governed by the MassDOT Developers License Agreement.

## Bounding box

`setup.sh` crops to `-71.30,42.20,-70.90,42.50`, deliberately wider than the
grid. Transit lines leaving the area still need their downstream stops or the
router will not know where a train goes. Override with `BBOX=...`.

## Tuning

`build-config.json`:
- `subwayAccessTime` (minutes) — time to get from street to platform. The
  single largest systematic error in transit isochrones, and a crude constant
  beats zero. 2.0 is a starting point; time yourself at Park Street.
- `platformEntriesLinking` — uses OSM entrance nodes where they exist.

`router-config.json`:
- `walkSpeed` — 1.33 m/s is the usual planning default. Match whatever you set
  in `config/mbta-transit.yaml`, or walk and transit will disagree.
- `accessEgress.maxDuration` — how far the router will walk to reach a stop.
  Raising it widens coverage and slows every query.

## Elevation (optional)

Boston has real grade. To apply slope penalties, fetch a USGS 3DEP DEM for the
area, drop the `.tif` in this directory, and add to `build-config.json`:

```json
"demDefaults": { "elevationUnitMultiplier": 1.0 },
"dem": [{ "source": "boston-dem.tif" }]
```

Then rebuild. Free, and worth doing before chasing better sidewalk data.
