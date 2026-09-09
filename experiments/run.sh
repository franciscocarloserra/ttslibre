#!/bin/bash
# Launch a training run in the foreground (Ctrl-C stops it; progress is also in runs/<run>/progress.log).
# Runs preflight.sh first (smoke + GPU utilization calibration: the batch is rescaled into the config gpu.* band); the run starts with that batch.
# A run whose runs/<run>/ttl.pt exists is resumed (budget clock continues); --fresh moves it to runs/todelete/ and starts over.
# Skip the preflight with PREFLIGHT=0. Usage: ./run.sh <group/NNN-exp> <run-name> [--fresh] [--set key=value ...]
{ # whole body in one block: bash reads scripts incrementally, so editing run.sh while a run is up would otherwise break the running copy
set -o pipefail; cd "$(dirname "$0")/$1" || exit 1; R=$2; shift 2
if [ "$1" = --fresh ]; then shift; [ -d runs/$R ] && { mkdir -p runs/todelete; mv runs/$R runs/todelete/${R}_$(date +%Y%m%d%H%M); }; fi
echo "[run] experiment $(pwd), run $R, config.json + overrides: $*"
if [ -f runs/$R/ttl.pt ]; then echo "[run] runs/$R/ttl.pt exists: resuming (step, optimizer and budget clock continue; --fresh starts over)"; set -- --resume "$@"; else echo "[run] new run (no runs/$R/ttl.pt)"; fi
if [ "${PREFLIGHT:-1}" = 1 ]; then
  echo "[run] stage 1/2 preflight (PREFLIGHT=0 skips it)"
  ../../preflight.sh "$@" | tee runs/preflight.log || exit 1
  B=$(tail -1 runs/preflight.log | sed -n 's/^BATCH=//p'); [ -n "$B" ] && set -- "$@" --set ttl.batch=$B
fi
echo "[run] stage 2/2 training: train.py --run $R $* (foreground, Ctrl-C stops; runs/$R/progress.log has the same lines; panel http://localhost:7807)"
exec ./venv/bin/python train.py --run "$R" "$@"
}
