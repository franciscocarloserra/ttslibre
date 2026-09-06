#!/bin/bash
# Overnight pipeline: wait for data and for 011 to finish, prep, train until overnight.end_hour, then eval every checkpoint.
set -e
cd "$(dirname "$0")"
RAW=../../datasets/libritts-r/raw
while [ ! -f $RAW/dl_train360.ok ]; do sleep 60; done
echo "$(date) download done"
[ -d $RAW/LibriTTS_R/train-clean-360 ] || tar xzf $RAW/train_clean_360.tar.gz -C $RAW
echo "$(date) extracted"
./venv/bin/python prepare.py 2>&1 | tail -1
./venv/bin/python encode.py 2>&1 | tail -1
while ! grep -q "STOP:" ../011-continue-4h-low-lr/runs/cont4h/progress.log; do sleep 60; done
while pgrep -f "eval_val.py" >/dev/null; do sleep 60; done
EH=$(python3 -c "import json;o=json.load(open('config.json'))['overnight'];print(o['end_hour']*60+o['end_minute'])")
NOW=$(( $(date +%H)*60 + $(date +%M) )); LEFT=$(( (EH - NOW + 1440) % 1440 ))
echo "$(date) launching train, budget $LEFT min"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv
setsid nohup ./venv/bin/python train.py --run overnight --set ttl.max_minutes=$LEFT > runs/overnight.out 2>&1 < /dev/null &
sleep 120
setsid nohup ./watch.sh overnight > runs/overnight.watch.log 2>&1 < /dev/null &
while pgrep -f "train.py --run overnight" >/dev/null; do sleep 120; done
echo "$(date) training done, evaluating"
./venv/bin/python eval_val.py 011 2>&1 | tail -1
for p in runs/overnight/ttl_*step*.pt; do ./venv/bin/python eval_val.py "012_$(basename $p .pt | sed 's/ttl_//')" $p 2>&1 | tail -1; done
./venv/bin/python eval_val.py 012best 2>&1 | tail -1
./venv/bin/python eval_val.py 012 2>&1 | tail -1
echo "$(date) all done"
