#!/bin/bash
# Smoke test per batch candidate: runs train.py --smoke (smoke.ttl_steps steps) while sampling nvidia-smi utilization; prints mean util per batch.
# Usage: ./smoke.sh   (candidates, seconds and target from config.json gpu.*)
cd "$(dirname "$0")"
T=$(python3 -c "import json;g=json.load(open('config.json'))['gpu'];print(g['util_target_pct'])")
for b in $(python3 -c "import json;print(*json.load(open('config.json'))['gpu']['batch_candidates'])"); do
  base=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  ./venv/bin/python train.py --smoke --run smoke_b$b --set smoke.batch=$b > runs/smoke_b$b.out 2>&1 &
  PID=$!; sleep 25; vals=()  # skip model load / baseline sample
  while kill -0 $PID 2>/dev/null; do vals+=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)); sleep 1; done
  u=$(printf "%s\n" "${vals[@]}" | awk '{s+=$1;n++} END{if(n)printf "%.0f", s/n; else print -1}')
  echo "batch $b: gpu util ${u}% over ${#vals[@]} s (idle before: ${base}%, target ${T}%)  $(grep -c 'audio loss' runs/smoke_b$b/progress.log) log lines  $(tail -1 runs/smoke_b$b.out | cut -c1-80)"
done
