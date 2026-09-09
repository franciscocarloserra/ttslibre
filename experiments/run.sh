#!/bin/bash
# Launch a training run in the foreground (Ctrl-C stops it; progress is also in runs/<run>/progress.log).
# Runs preflight.sh first (smoke + GPU utilization calibration: the batch is rescaled into the config gpu.* band); the run starts with that batch.
# A run whose runs/<run>/ttl.pt exists is resumed (budget clock continues); --fresh moves it to runs/todelete/ and starts over.
# Skip the preflight with PREFLIGHT=0. Usage: ./run.sh <group/NNN-exp> <run-name> [--fresh] [--set key=value ...]
set -o pipefail; cd "$(dirname "$0")/$1" || exit 1; R=$2; shift 2
if [ "$1" = --fresh ]; then shift; [ -d runs/$R ] && { mkdir -p runs/todelete; mv runs/$R runs/todelete/${R}_$(date +%Y%m%d%H%M); }; fi
[ -f runs/$R/ttl.pt ] && { echo "run $R: resuming from runs/$R/ttl.pt (use --fresh to start over)"; set -- --resume "$@"; }
if [ "${PREFLIGHT:-1}" = 1 ]; then
  ../../preflight.sh "$@" | tee runs/preflight.log || exit 1
  B=$(tail -1 runs/preflight.log | sed -n 's/^BATCH=//p'); [ -n "$B" ] && set -- "$@" --set ttl.batch=$B
fi
echo "run $R: $@"
exec ./venv/bin/python train.py --run "$R" "$@"
