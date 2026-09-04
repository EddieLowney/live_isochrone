#!/usr/bin/env bash
# Start OTP against the built graph. Leave this running in its own terminal.
set -euo pipefail
cd "$(dirname "$0")"

OTP_VERSION="${OTP_VERSION:-2.5.0}"
OTP_JAR="otp-${OTP_VERSION}-shaded.jar"
HEAP="${HEAP:-4G}"

[ -f "$OTP_JAR" ] || { echo "No $OTP_JAR — run ./setup.sh first"; exit 1; }
[ -f graph.obj ]  || { echo "No graph.obj — run ./setup.sh first"; exit 1; }

echo "Starting OTP on http://127.0.0.1:8080 (heap $HEAP, Ctrl-C to stop)"
exec java "-Xmx${HEAP}" -jar "$OTP_JAR" --load .
