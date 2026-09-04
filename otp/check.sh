#!/usr/bin/env bash
# Is OTP up, does it have the feed loaded, and is realtime actually flowing?
set -uo pipefail
URL="${OTP_URL:-http://127.0.0.1:8080}"
GQL=""
for p in /otp/gtfs/v1 /otp/routers/default/index/graphql; do
  if curl -fsS -m 5 -X POST "$URL$p" -H 'Content-Type: application/json' \
       -d '{"query":"{ feeds { feedId } }"}' >/dev/null 2>&1; then GQL="$p"; break; fi
done
[ -n "$GQL" ] || { echo "FAIL  no GraphQL endpoint at $URL — is OTP running?"; exit 1; }
echo "ok    GraphQL at $GQL"

curl -fsS -X POST "$URL$GQL" -H 'Content-Type: application/json' \
  -d '{"query":"{ feeds { feedId agencies { name } } }"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin)['data']['feeds']; [print(f'ok    feed {f[\"feedId\"]}: {len(f[\"agencies\"])} agencies') for f in d]"

# A trip update with a nonzero delay proves GTFS-RT is being consumed, not just
# the static timetable. If every arrival is exactly on schedule, the updaters
# are probably not connected.
echo -n "      realtime: "
curl -fsS -X POST "$URL$GQL" -H 'Content-Type: application/json' \
  -d '{"query":"{ stops(name:\"Park Street\") { name stoptimesWithoutPatterns(numberOfDepartures:8) { realtime realtimeDeparture scheduledDeparture } } }"}' \
  | python3 -c "
import sys, json
try:
    stops = json.load(sys.stdin)['data']['stops']
except Exception:
    print('could not parse — check the query against your OTP version'); raise SystemExit
rt = [t for s in stops for t in s['stoptimesWithoutPatterns'] if t.get('realtime')]
tot = [t for s in stops for t in s['stoptimesWithoutPatterns']]
if not tot: print('no departures found (service may not be running right now)')
elif rt:   print(f'{len(rt)}/{len(tot)} departures carry realtime — GTFS-RT is live')
else:      print(f'0/{len(tot)} departures carry realtime — updaters may not be connected')
"
