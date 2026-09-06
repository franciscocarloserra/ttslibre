#!/bin/bash
# Chained after 012: style packs and a mix on 012's final checkpoint, WER test, panel restart on 013.
set -e
cd "$(dirname "$0")"
while ! grep -q "all done" ../012-english-235h-overnight/runs/pipeline.log 2>/dev/null; do sleep 120; done
CK=../012-english-235h-overnight/runs/overnight/ttl.pt; T=$(python3 -c "import json;print(json.load(open('config.json'))['style']['mix_t'])")
echo "$(date) 012 done, checkpoint $CK"
V="./venv/bin/python ../voice.py"
$V test 6209 --exp 013-style-vectors --run $CK | tail -1
$V style 6209 --exp 013-style-vectors --run $CK
$V test 6209.style --exp 013-style-vectors --run $CK | tail -1
$V test 4137 --exp 013-style-vectors --run $CK | tail -1
$V style 4137 --exp 013-style-vectors --run $CK
$V mix 6209 4137 $T mix6209x4137 --exp 013-style-vectors --run $CK
$V test mix6209x4137.style --exp 013-style-vectors --run $CK | tail -1
mkdir -p ../voices/todelete; mv ../voices/*.wav ../voices/todelete/ 2>/dev/null || true
{ echo "# 013 results"; echo; echo "Checkpoint: $CK. Whisper WER, 4 sentences x 3 draws (007 set). $(date)"; echo; echo "| voice | mean WER |"; echo "|---|---|"
  for n in 6209 6209.style 4137 mix6209x4137.style; do echo "| $n | $(python3 -c "import json;print(json.load(open('../voices/$n.test.json'))['mean_wer'])") |"; done; } > RESULTS.md
PID=$(pgrep -f "python panel.py" | head -1); [ -n "$PID" ] && kill $PID; sleep 2
setsid nohup ./venv/bin/python ../panel.py --exp 013-style-vectors > runs/panel.log 2>&1 < /dev/null &
echo "$(date) all done"; cat RESULTS.md
