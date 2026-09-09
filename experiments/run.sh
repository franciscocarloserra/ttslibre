#!/bin/bash
# Launch a training run in the foreground (Ctrl-C stops it; progress is also in runs/<run>/progress.log). Usage: ./run.sh <group/NNN-exp> <run-name> [--set key=value ...]
cd "$(dirname "$0")/$1" || exit 1; R=$2; shift 2
exec ./venv/bin/python train.py --run "$R" "$@"
