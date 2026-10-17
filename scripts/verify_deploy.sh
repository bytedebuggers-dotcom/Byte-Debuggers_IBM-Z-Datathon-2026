#!/usr/bin/env bash
BASE="$1"
if [ -z "$BASE" ]; then BASE="http:127.0.0.1:8000"; fi
echo "Architecture: $(uname -m)"
echo "Health:"
curl -s "$BASE/health" 
echo
echo "Latest event:"
curl -s "$BASE/events/latest"
echo
echo "SMS export:"
curl -s "$BASE/exports/sms"
echo
echo "Prediction test:"
curl -s -X POST "$BASE/predict" -H "Content-Type: application/json" \
    -d '{"magnitude":6.1, "dist_km":12, "shaking":5.2, "pipe_age":40,"sensor_var":0.6}'
echo 
