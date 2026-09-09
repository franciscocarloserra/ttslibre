# 005 results

Run `all185`: 185 clips of speaker 1988, warm start from 004 `w16`, lr 2e-3, budget 60 min.

**Outcome: crashed at ~12 min with NaN** (`ValueError: cannot convert float NaN to integer` in `synth.py`, duration predictor output). The run was launched twice by mistake, so two processes shared the GPU and wrote the same log; both went NaN (step ~2600 and ~3000). No `STOP` line; the budget was never reached.

WER per round before the crash (both processes, so two values per step):

| step | 200 | 800 | 1600 | 2200 | 2800 |
|---|---|---|---|---|---|
| train | 1.1–1.3 | 0.40–0.45 | 0.15–0.25 | 0.10–0.25 | 0.30 |
| knownwords | ~1.0 | ~1.0 | ~1.0 | 0.92 | 0.92 |
| heldout1 | ≥1.0 | ≥2.0 | ≥2.0 | 2.0 | 2.0 |

**Learned.** Train WER falls faster than 003 (0.10 at step 2200 vs 0.55 at 3871) thanks to warm start + higher lr, but lr 2e-3 diverges. knownwords/heldout do not improve.

**Not tested.** Whether 185 clips overfit with a stable lr and the full hour. No NaN guard existed; 006 adds one.
