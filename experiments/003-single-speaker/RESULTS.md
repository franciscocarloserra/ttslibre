# 003 results

Run `spk1988`, 2026-09-05. Speaker 1988, 185 clips, 002 recipe, from scratch. Stopped by `ttl.max_minutes` = 12 at step 3871.

Whisper WER per sample round (`train` = training sentence, `heldout1/2` = never seen):

| step | train | heldout1 | heldout2 |
|---|---|---|---|
| 200 | 1.00 | 2.00 | 1.33 |
| 400 | 1.15 | 3.00 | 1.00 |
| 600 | 1.25 | 3.00 | 1.08 |
| 800 | 1.00 | 1.33 | 1.00 |
| 1000 | 1.35 | 4.33 | 1.00 |
| 1200 | 1.05 | 2.00 | 1.00 |
| 1400 | 1.15 | 2.33 | 1.08 |
| 1600 | 1.00 | 1.67 | 1.00 |
| 1800 | 1.00 | 4.00 | 1.08 |
| 2000 | 0.90 | 1.00 | 1.00 |
| 2200 | 1.25 | 3.00 | 1.00 |
| 2400 | 1.15 | 3.00 | 1.00 |
| 2600 | 1.15 | 1.00 | 1.00 |
| 2800 | 0.90 | 1.67 | 1.00 |
| 3000 | 1.05 | 2.33 | 1.08 |
| 3200 | 0.80 | 2.33 | 1.00 |
| 3400 | 0.90 | 2.33 | 1.08 |
| 3600 | 0.75 | 1.33 | 1.00 |
| 3800 | 0.55 | 2.00 | 1.00 |

Last transcripts: train: "He was 19 years old, and who saw that, Frotts said, that, fat Frotts, it's a little flat face." / heldout1: "really, really rock and super lame."

## What we learned

- Same recipe, 45x more data than 002, same time: no alignment. The training sentence WER only started dropping in the last 3 rounds (1.0 -> 0.75 -> 0.55); the run was cut before it got there. Held-out never left >= 1.0.
- Compared with 002 on the same sentence it is worse (0.55 vs 0.10), but the curve was still going down: with more data, memorizing one sentence takes longer, it is not a code regression.

## Not tested

- Longer budget (the curve suggests 20-30 min would align the training sentence).
- Whether held-out would follow once aligned.
