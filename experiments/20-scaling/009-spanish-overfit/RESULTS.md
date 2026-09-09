# Results: 009 spanish overfit from scratch

Run `sabato-scratch`, stopped by user at 1h33m (step 32650) of a 2 h budget.
Data: 775 clips (45 min) of one speaker (Sabato), 10% held out. No warm start.
Checkpoints: ttl.pt, best.pt, partials at steps 12000 / 22000 / 32000.

## WER per round (Whisper, accent-insensitive)

| step | time | train | novel1 | novel2 | heldout1 | heldout2 |
|---|---|---|---|---|---|---|
| 2000 | 5m | 1.40 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10000 | 28m | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 18000 | 51m | 0.70 | 1.00 | 1.11 | 1.00 | 1.67 |
| 26000 | 1h14m | 1.10 | 1.00 | 0.89 | 0.67 | 1.33 |
| 28000 | 1h20m | 0.20 | 1.12 | 1.22 | 1.11 | 0.67 |
| 30000 | 1h26m | 0.80 | 1.00 | 0.89 | 1.11 | 1.33 |
| 32000 | 1h32m | 0.20 | 1.00 | 1.00 | 1.11 | 1.00 |

## Losses

| step | train audio | val audio | train dur | val dur |
|---|---|---|---|---|
| 900 | 0.53 | - | 0.24 | - |
| 15000 | ~0.30 | 0.69 | ~0.03 | 0.13 |
| 30000 | ~0.22 | 0.79 | ~0.02 | 0.11 |

Train audio loss keeps falling, validation audio loss rises after step 15000.

## What Whisper heard (step 28000-30000)

- train: "y que no se necesitan mayores independencias de mi persona" (target: "...mayores explicaciones sobre mi persona")
- novel2: "El miembro de Polarán iba de Dios a romperse"
- heldout1: "Y la idea es que se le va a irse a Tijolán"
- Several rounds detected as Italian, Portuguese, French or English.

## Learned

- From scratch on 45 min of one speaker: Spanish-sounding fluent speech, real Spanish words, but text is not read. Same state as 008 (warm start, 10 min), reached slower: train WER first < 1.0 at step 12000 (34 min) vs step 1000 in 008.
- Validation audio loss rising while train loss falls = the model memorizes the clips. More steps on this data will not help; more text variety is needed.
- Accent judged by ear by the user (native): see README "Result".

## Not tested

- Full 2 h budget (stopped at 1h33m; trend was flat).
- Comparison of 008 vs 009 samples by Whisper language detection.
