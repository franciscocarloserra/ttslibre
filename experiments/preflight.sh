#!/bin/bash
# Preflight for a training run, executed from the experiment dir: smoke test of train.py (does the script work end to end?) while measuring GPU
# utilization; if it is outside config gpu.util_target_pct ± gpu.util_tolerance_pct the batch is rescaled (batch * target / measured, rounded to
# gpu.batch_step, capped at gpu.batch_max) and the smoke repeats, up to gpu.calib_max_tries times. Last stdout line: BATCH=<chosen>.
# Exit 1 only when the smoke itself fails. Usage: preflight.sh [--set key=value ...]
read T TOL SKIP B STEP BMAX TRIES < <(python3 -c "import json;c=json.load(open('config.json'));g=c.get('gpu',{});print(g.get('util_target_pct',0),g.get('util_tolerance_pct',0),g.get('smoke_skip_seconds',25),c['ttl']['batch'],g.get('batch_step',8),g.get('batch_max',128),g.get('calib_max_tries',3))")
mkdir -p runs/todelete; trap 'kill $PID 2>/dev/null; exit 130' INT TERM
for try in $(seq 1 $TRIES); do
  rm -rf runs/todelete/preflight_b$B; mv runs/preflight_b$B runs/todelete/ 2>/dev/null
  ./venv/bin/python train.py --smoke --run preflight_b$B "$@" --set smoke.batch=$B > runs/preflight_b$B.out 2>&1 &
  PID=$!; sleep $SKIP; vals=()
  while kill -0 $PID 2>/dev/null; do vals+=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)); sleep 1; done
  wait $PID || { echo "preflight: smoke FAILED (runs/preflight_b$B.out):"; tail -5 runs/preflight_b$B.out; exit 1; }
  u=$(printf "%s\n" "${vals[@]}" | awk '{s+=$1;n++} END{if(n)printf "%.0f", s/n; else print -1}')
  echo "preflight $try/$TRIES: smoke ok, batch $B, gpu util ${u}% over ${#vals[@]} s (target ${T}±${TOL}%)"
  [ "$T" -eq 0 ] || [ "$u" -le 0 ] && break
  [ "$u" -ge $((T-TOL)) ] && [ "$u" -le $((T+TOL)) ] && break
  NB=$(( (B * T / u + STEP / 2) / STEP * STEP )); [ $NB -lt $STEP ] && NB=$STEP; [ $NB -gt $BMAX ] && NB=$BMAX
  [ $NB -eq $B ] && break
  echo "preflight: rescaling batch $B -> $NB"; B=$NB
done
echo "BATCH=$B"
