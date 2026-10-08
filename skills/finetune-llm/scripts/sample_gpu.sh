#!/bin/bash
# Sample GPU memory/utilization to CSV while a training run is in flight.
# Use the same sampler for every arm so measurements are comparable.
#
#   ./sample_gpu.sh 0 logs/run_gpu.csv &
#   <training command>
#   kill %1
#
# Args: [gpu_index=0] [out_csv=/tmp/gpu.csv] [interval_s=2]
set -u
GPU="${1:-0}"
OUT="${2:-/tmp/gpu${GPU}.csv}"
INTERVAL="${3:-2}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; nothing to sample" >&2
  exit 1
fi

echo "ts,used_mib,total_mib,util_pct" > "$OUT"
while true; do
  line=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.total,utilization.gpu \
         --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$line" ]; then
    sleep "$INTERVAL"
    continue
  fi
  echo "$(date +%s),$(echo "$line" | tr -d ' ')" >> "$OUT"
  sleep "$INTERVAL"
done
