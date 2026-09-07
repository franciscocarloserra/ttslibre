# 011 continue 4h low lr

**Question.** Continuing 010 on the same 45 h with the learning rate it ended on (1e-4, no bump) and a 4-hour budget, does whole-val-set WER improve over 006 and 010?

**Two parts.**
1. `eval_val.py <name>`: Whisper WER over all 99 val sentences, `eval.val_draws` (3) draws each, speaker 1988 reference. Run on `006`, `010`, `010best` (`eval.val_checkpoints`) before training, then on this run's checkpoints. This is the calibration number that the 7-sentence rounds cannot give (010 lesson).
2. `train.py --run cont4h`: warm start from 010 `clean100/ttl.pt`, lr 1e-4 halved at 60k, 240 min, samples every 4000 steps (~23 rounds), 4 partial checkpoints.

**Changes vs 010.** lr 2.5e-4 → 1e-4 (010 regressed for 30 min after the bump), budget 120 → 240 min. Data unchanged.

**Reading.** `eval_val` mean WER: 006 vs 010 says whether the 45 h helped at all; 011 checkpoints vs 010 says whether more steps on the same data help. If 011 ≈ 010, the next lever is more data (train-clean-360), not more time.

**Run.**
```bash
for n in 006 010 010best; do ./venv/bin/python eval_val.py $n; done
./venv/bin/python train.py --run cont4h
./venv/bin/python eval_val.py 011 runs/cont4h/ttl.pt
```

**Result.** no gain over 010 (whole-val mean WER 0.291 vs 0.284-0.296): clean-100 saturated, expand data. See `RESULTS.md`.
