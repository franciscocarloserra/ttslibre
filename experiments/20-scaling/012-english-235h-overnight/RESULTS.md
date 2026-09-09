# 012 results

Run `overnight`: 7h39m, 166068 steps, init from 011 `ttl.pt`, lr 1e-4 (halved every 80k), data dev-clean + train-clean-100 + train-clean-360 (`prep-clean460`: 131423 clips, 177.4 h after the 1-12 s filter). Audio loss 0.561 at stop.

Whole-val Whisper WER (same 99 val sentences x 3 draws as 010/011, `runs/evalval/*.summary.json`):

| checkpoint | hours into run | mean WER | median | frac > 0.5 |
|---|---|---|---|---|
| 011 (init) | 0 | 0.291 | 0.200 | 0.18 |
| step 36000 | 1h40m | 0.365 | 0.333 | 0.25 |
| step 72000 | 3h19m | 0.334 | 0.278 | 0.21 |
| step 102000 | 4h42m | 0.232 | 0.154 | 0.14 |
| step 138000 | 6h21m | 0.232 | 0.136 | 0.13 |
| best.pt | - | 0.266 | 0.182 | 0.15 |
| ttl.pt (final) | 7h39m | 0.254 | 0.176 | 0.15 |

- Adding 4x data first hurt (0.29 -> 0.37 at 1h40m: distribution shift to 900+ new speakers), then recovered and beat 011 from ~4.5 h on. Best whole-val number so far: 0.232 (step 102000 / 138000).
- Final ttl.pt (0.254) is slightly worse than the 4h42m / 6h21m checkpoints; within this eval's noise (3 draws) but not improving in the last 1.5 h. `best.pt` is selected by the 7-sentence probe and is not the best on whole-val.
- Recommended checkpoint for downstream use: `runs/overnight/ttl_6h21m_step138000.pt` (mean 0.232, median 0.136).
- Next: WER is still 0.23 on average with 13% of sentences above 0.5; the model (19.8M params) is likely the limit now rather than data. Options: longer schedule with lr decay to zero, or a wider model.
