# 005 overfit 185

**Question.** Can the model overfit all 185 clips of one speaker? 004 overfit 16 in 5 min; 003 failed on 185 from scratch in 12 min.

**Changes vs 004.** All 185 clips of speaker 1988 (no length filter), warm start from 004 `w16`, `ttl.lr` 2e-3 (was 1e-3), `ttl.max_minutes` 60, failure stop at 20 min if the training sentence WER is still > 0.5.

**Samples every 200 steps.** `train` = a training sentence (the 002 one). `knownwords` = new sentence from words in the training set. `heldout1` = never-seen words.

**Reading.** `train` WER near 0 = overfits 185. `knownwords` dropping too = starts learning words. Spanish stays out of scope until English generalizes (reference implementation went one language at a time).

**Run.**
```bash
./venv/bin/python train.py --run all185
```

**Result.** pending (`RESULTS.md`)
