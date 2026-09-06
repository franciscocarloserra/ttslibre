# 012 english 235h overnight

**Question.** With dev-clean + train-clean-100 + train-clean-360 (~235 h, ~1150 speakers), does whole-val-set WER keep falling with data, and does the long-sentence case finally move?

**Data.** `datasets/libritts-r/prep-clean460`, same `prepare.py` / `encode.py` as 010 (vocab and val split from dev-clean, so the 99 val sentences are the same as 006/010/011).

**Changes vs 011.** Data (+190 h). Warm start from 011 `cont4h/ttl.pt`, lr 1e-4 halved every 80k, budget = minutes until `overnight.end_hour` (04:00) at launch (`ttl.max_minutes` is overwritten by `pipeline.sh` with that number), samples every 6000 steps, 5 partial checkpoints, val loss every 20k.

**After training (chained in `pipeline.sh`).** `eval_val.py` on 011 final, every 012 partial checkpoint, `best.pt` and final: whole-val WER vs training hours, in `runs/evalval/`.

**Run.** `./pipeline.sh` (waits for the 360 download and for 011 to stop, extracts, preps, encodes, trains, evaluates). Log: `runs/pipeline.log`.

**Result.** pending (`RESULTS.md`)
