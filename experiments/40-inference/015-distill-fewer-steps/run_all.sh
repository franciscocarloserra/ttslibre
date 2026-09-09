#!/bin/bash
# Runs ON the rented instance, from this directory: baseline -> distill rounds -> student eval, each timed into
# $LOGS/timings.log; GPU util/VRAM sampled into $LOGS/gpu.log; touches $DONE_FILE at the end (contains FAILED if a stage failed).
# Env (from vast-test/params.env, exported by 05_run.sh): REMOTE_DIR, DONE_FILE, GPU_SAMPLE_S; SKIP_BASELINE=1 to relaunch from distill; DISTILL_ARGS="--resume-rounds" to skip finished rounds. Whisper token: $REMOTE_DIR/secrets.env.
cd "$(dirname "$0")"
LOGS=${REMOTE_DIR:?}/logs; mkdir -p "$LOGS" runs; [ -f "$REMOTE_DIR/secrets.env" ] && source "$REMOTE_DIR/secrets.env"; export WHISPER_TOKEN
T=$LOGS/timings.log; rm -f "$DONE_FILE"; status=OK
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)"; sleep "${GPU_SAMPLE_S:-30}"; done ) >> "$LOGS/gpu.log" 2>/dev/null & GP=$!
stage() { local n=$1; shift; local t0=$(date +%s); echo "== $n start $(date '+%F %T')" | tee -a "$T"
  "$@" >> "$LOGS/run_all.log" 2>&1 || { status=FAILED; echo "$n FAILED (see logs/run_all.log)" | tee -a "$T"; }
  echo "T_run_$n=$(( $(date +%s)-t0 ))s" | tee -a "$T"; }
[ -n "$SKIP_BASELINE" ] && [ -f runs/baseline/grid.md ] && echo "baseline skipped (exists)" | tee -a "$T" || stage baseline python3 baseline.py
stage distill python3 distill.py $DISTILL_ARGS
BEST=$(ls -d runs/distill/r*to* 2>/dev/null | sort -t o -k2 -n | tail -1)/best.pt
[ -f "$BEST" ] && stage eval python3 baseline.py --section eval --run "$BEST" || { echo "no student checkpoint, eval skipped" | tee -a "$T"; status=FAILED; }
kill $GP 2>/dev/null; echo "$status $(date '+%F %T')" > "$DONE_FILE"; echo "DONE $status" | tee -a "$T"
