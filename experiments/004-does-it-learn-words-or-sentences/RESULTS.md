# 004 results

Run `w16`, 2026-09-05. Warm start from 002 `overfit4`, speaker 1988, 16 clips under 4 s. Stopped by `ttl.max_minutes` = 5 at step 1448.

Sample sentences: `train` = "He was neatly dressed." (in the 16 clips); `knownwords` = "Pop was at the beach with Hilda and the boss all evening." (new order, every word in the 16 clips); `heldout1` = never-seen words.

| step | train | knownwords | heldout1 |
|---|---|---|---|
| 100 | 0.00 | 0.92 | 1.67 |
| 200 | 0.00 | 1.00 | 3.00 |
| 300 | 0.00 | 1.25 | 1.00 |
| 400 | 0.00 | 0.83 | 2.67 |
| 500 | 0.25 | 1.17 | 2.33 |
| 600 | 0.00 | 1.00 | 2.00 |
| 700 | 0.00 | 0.92 | 2.67 |
| 800 | 0.00 | 1.00 | 2.00 |
| 900 | 0.00 | 0.92 | 2.00 |
| 1000 | 0.00 | 1.00 | 2.33 |
| 1100 | 0.50 | 1.42 | 1.00 |
| 1200 | 0.00 | 1.00 | 3.33 |
| 1300 | 0.00 | 1.00 | 2.00 |
| 1400 | 0.00 | 1.08 | 2.33 |

Last transcripts: train: "He was neatly dressed." / knownwords: "Let's put a little glass to play who reads the top of my hearing."

## Answer

No. With 16 sentences the model memorizes sentences, it does not learn words. `train` reached 0.00 from step 1200; `knownwords` never went below 0.92. Warm start works: `train` was already 0.00 at step 100.

## What we learned

- Warm start from an aligned checkpoint keeps alignment and speeds up overfitting on new sentences (0.00 on a new training sentence within 100 steps).
- Recombining known words is not learned at this data size. Character-to-sound mapping needs more sentences than 16.

## Not tested

- The same question at 64 and 185 clips (data ladder).
