#!/bin/bash
# Train es8h with the config budget, then whole-val eval of the init checkpoint and every partial/final checkpoint.
set -e
cd "$(dirname "$0")"
[ -f ../../datasets/openslr61/prep/latent_stats.pt ] || { ./venv/bin/python prepare.py 2>&1 | tail -1; ./venv/bin/python encode.py 2>&1 | tail -1; }
echo "$(date) launching train"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv
setsid nohup ./venv/bin/python train.py --run es8h > runs/es8h.out 2>&1 < /dev/null &
sleep 120
setsid nohup ./watch.sh es8h > runs/es8h.watch.log 2>&1 < /dev/null &
while pgrep -f "train.py --run es8h" >/dev/null; do sleep 120; done
echo "$(date) training done, evaluating"
./venv/bin/python eval_val.py 012 2>&1 | tail -1
for p in runs/es8h/ttl_*step*.pt; do ./venv/bin/python eval_val.py "014_$(basename $p .pt | sed 's/ttl_//')" $p 2>&1 | tail -1; done
./venv/bin/python eval_val.py 014best 2>&1 | tail -1
./venv/bin/python eval_val.py 014 2>&1 | tail -1
echo "$(date) all done"
