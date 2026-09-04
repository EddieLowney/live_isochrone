#!/usr/bin/env bash
# One-time setup: fetch OTP, an OSM extract, and MBTA GTFS, then build the graph.
#
# Run from the otp/ directory. Expect 15-30 minutes, mostly downloading.
# Re-run after `rm graph.obj` to rebuild against fresher data.
set -euo pipefail
cd "$(dirname "$0")"

OTP_VERSION="${OTP_VERSION:-2.5.0}"
OTP_JAR="otp-${OTP_VERSION}-shaded.jar"
# Bounding box around Boston. Wider than the grid on purpose: transit lines
# leaving the area still need their downstream stops for the router to know
# where a train goes.
BBOX="${BBOX:---bbox=-71.30,42.20,-70.90,42.50}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Checking prerequisites"
command -v java >/dev/null || { echo "Java not found. brew install openjdk@21"; exit 1; }
JAVA_MAJOR=$(java -version 2>&1 | head -1 | sed -E 's/.*"([0-9]+).*/\1/')
if [ "${JAVA_MAJOR:-0}" -lt 17 ]; then
  echo "Java $JAVA_MAJOR found; OTP 2.5 needs 17 or newer."
  echo "  brew install openjdk@21 && export PATH=\"/opt/homebrew/opt/openjdk@21/bin:\$PATH\""
  exit 1
fi
echo "Java $JAVA_MAJOR — ok"

say "Fetching OpenTripPlanner ${OTP_VERSION}"
if [ ! -f "$OTP_JAR" ]; then
  curl -fL --progress-bar -o "$OTP_JAR" \
    "https://repo1.maven.org/maven2/org/opentripplanner/otp/${OTP_VERSION}/otp-${OTP_VERSION}-shaded.jar"
else
  echo "already present"
fi

say "Fetching MBTA GTFS schedule"
if [ -f mbta.gtfs.zip ]; then
  curl -fL --progress-bar -z mbta.gtfs.zip -o mbta.gtfs.zip "https://cdn.mbta.com/MBTA_GTFS.zip"
else
  curl -fL --progress-bar -o mbta.gtfs.zip "https://cdn.mbta.com/MBTA_GTFS.zip"
fi
unzip -l mbta.gtfs.zip | tail -3

say "Fetching Massachusetts OSM extract"
if [ ! -f massachusetts-latest.osm.pbf ]; then
  curl -fL --progress-bar -o massachusetts-latest.osm.pbf \
    "https://download.geofabrik.de/north-america/us/massachusetts-latest.osm.pbf"
else
  echo "already present (delete to refresh)"
fi

say "Cropping to the Boston area"
# Not cosmetic. The statewide graph needs roughly double the heap of the metro
# area, which is the difference between building and swapping on a laptop.
BUILD_HEAP=4G
RUN_HEAP=3G
if ! command -v osmium >/dev/null && command -v brew >/dev/null; then
  echo "osmium-tool is not installed and makes this build much smaller."
  read -r -p "Install it with brew now? [Y/n] " reply
  case "${reply:-Y}" in [Yy]*) brew install osmium-tool ;; esac
fi
if command -v osmium >/dev/null; then
  osmium extract $BBOX massachusetts-latest.osm.pbf -o boston.osm.pbf --overwrite
  ls -lh boston.osm.pbf
else
  echo "Proceeding with the full state extract."
  echo "Raising the heap; if the build dies with OutOfMemoryError, install"
  echo "osmium-tool and re-run rather than raising it further."
  cp massachusetts-latest.osm.pbf boston.osm.pbf
  BUILD_HEAP=8G
  RUN_HEAP=6G
fi

say "Building the graph (heap $BUILD_HEAP)"
java "-Xmx$BUILD_HEAP" -jar "$OTP_JAR" --build --save .

say "Done"
echo "Graph built: $(ls -lh graph.obj | awk '{print $5}')"
echo
echo "Next:  HEAP=$RUN_HEAP bash run.sh    then bash check.sh"
echo
echo "Verify realtime is actually flowing before trusting the map:"
echo "  curl -s -X POST http://127.0.0.1:8080/otp/gtfs/v1 \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\":\"{ feeds { feedId } }\"}'"
