#!/bin/bash
# Watchdog: if train.py dies before a final STOP (max_minutes / held-out success), relaunch it with --resume
# and the remaining budget; the failure stop is disabled on restarts. Usage: ./watch.sh <run> [max_restarts=5]
cd "$(dirname "$0")"; R=$1; N=${2:-5}; D=runs/$R
START=$(stat -c %Y "$D/config.effective.json"); BUDGET=$(python3 -c "import json;print(json.load(open('$D/config.effective.json'))['ttl']['max_minutes'])")
n=0
while sleep 60; do
  pgrep -f "^\./venv/bin/python train\.py --run $R( |$)" >/dev/null && continue
  grep -qE 'STOP: (max_minutes|heldout)' "$D/progress.log" && { echo "$(date)  finished"; exit 0; }
  left=$(( BUDGET - ($(date +%s) - START) / 60 )); [ $left -le 1 ] && { echo "$(date)  budget over"; exit 0; }
  [ $n -ge $N ] && { echo "$(date)  giving up after $N restarts"; exit 1; }
  n=$((n + 1)); echo "$(date)  restart $n, $left min left"
  setsid nohup ./venv/bin/python train.py --run "$R" --resume --set ttl.max_minutes=$left --set ttl.fail_after_minutes=100000 >> "$D.out" 2>&1 < /dev/null &
done
