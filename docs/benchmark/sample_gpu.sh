#!/bin/bash
# Sample nvidia-smi memory/util for one GPU while a run is in flight.
# Needed because Soup's own bench command refuses to measure a run it did not
# launch, and we want the same metric captured for both arms by the same tool.
GPU="${1:-0}"
OUT="${2:-/tmp/gpu0.csv}"
INTERVAL="${3:-2}"
echo "ts,used_mib,total_mib,util_pct" > "$OUT"
while true; do
  line=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.total,utilization.gpu \
         --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$line" ] && sleep "$INTERVAL" && continue
  echo "$(date +%s),$(echo "$line" | tr -d ' ')" >> "$OUT"
  sleep "$INTERVAL"
done
