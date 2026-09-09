# 010 results

Run `clean100`: dev-clean + train-clean-100 (33202 clips, 287 speakers, 44.6 h), warm start from 006 `devclean/ttl.pt` (step 160000), lr 2.5e-4 halved at 30k, batch 16×8. Stopped by `ttl.max_minutes` = 120 at step 46197. Peak VRAM 4.3 GiB. Checkpoints: `ttl.pt`, `best.pt`, partials at 12000 / 24000 / 36000.

Data prep: 38968 wavs, 33301 kept (1–12 s and characters inside the 006 vocab), 99 val rows copied from the dev-clean split. Prep 6 min (8 workers), encode 1 min.

Note: `heldout1..3` are 3 sentences of the same 99-row val set as 006, but not the same 3 (val rows are sorted by id here, shuffled in 006). The fair comparison with 006 is the step 0 row below (006 weights, untrained on the new data). heldout1 is the 27-word sentence ("Victorian novelists...").

## WER per round (speaker 1988 reference; one round every 2000 steps, subset shown)

| step | elapsed | train | knownwords | novel1 | novel2 | heldout1 | heldout2 | heldout3 |
|---|---|---|---|---|---|---|---|---|
| 0 (= 006 final) | 0 | 0.00 | 0.67 | 0.22 | 0.00 | 0.90 | 0.33 | 0.00 |
| 25 | 0m15s | 0.00 | 0.67 | 0.22 | 0.00 | 0.90 | 0.33 | 0.00 |
| 2025 | 5m27s | 0.50 | 0.58 | 0.44 | 0.86 | 1.00 | 0.50 | 0.75 |
| 4025 | 10m38s | 0.15 | 0.25 | 0.22 | 0.00 | 1.00 | 0.33 | 0.25 |
| 6025 | 15m49s | 0.35 | 0.50 | 0.22 | 0.43 | 1.45 | 0.50 | 0.58 |
| 8025 | 20m59s | 0.40 | 0.50 | 0.33 | 0.14 | 1.26 | 0.67 | 0.33 |
| 10025 | 26m10s | 0.30 | 0.25 | 0.33 | 0.00 | 1.16 | 0.17 | 0.17 |
| 12025 | 31m22s | 0.60 | 0.75 | 0.33 | 0.57 | 1.00 | 1.00 | 0.50 |
| 14025 | 36m32s | 0.45 | 0.50 | 0.22 | 0.29 | 0.97 | 0.17 | 0.58 |
| 16025 | 41m42s | 0.40 | 0.75 | 0.33 | 0.00 | 1.10 | 0.00 | 0.50 |
| 18025 | 46m53s | 0.40 | 0.58 | 0.33 | 0.00 | 0.90 | 0.00 | 0.50 |
| 20025 | 52m04s | 0.30 | 0.25 | 0.33 | 0.29 | 1.03 | 0.17 | 0.17 |
| 22025 | 57m15s | 0.45 | 0.50 | 0.33 | 0.14 | 1.03 | 0.17 | 0.33 |
| 24025 | 1h02m | 0.45 | 0.75 | 0.44 | 0.00 | 0.90 | 0.50 | 0.33 |
| 26025 | 1h07m | 0.15 | 0.58 | 0.78 | 0.00 | 0.94 | 0.17 | 0.75 |
| 28025 | 1h12m | 0.35 | 0.67 | 0.44 | 0.14 | 1.10 | 0.33 | 0.33 |
| 30025 | 1h18m | 0.45 | 0.33 | 0.44 | 0.00 | 0.87 | 0.50 | 0.17 |
| 32025 | 1h23m | 0.10 | 0.42 | 0.00 | 0.00 | 1.13 | 0.00 | 0.08 |
| 34025 | 1h28m | 0.30 | 0.33 | 0.22 | 0.00 | 0.90 | 0.00 | 0.17 |
| 36025 | 1h33m | 0.30 | 0.67 | 0.44 | 0.00 | 0.90 | 0.00 | 0.33 |
| 38025 | 1h38m | 0.15 | 0.33 | 0.33 | 0.14 | 1.16 | 0.33 | 0.00 |
| 40025 | 1h44m | 0.30 | 0.33 | 0.44 | 0.43 | 1.10 | 0.33 | 0.17 |
| 42025 | 1h49m | 0.20 | 0.33 | 0.33 | 0.00 | 1.55 | 0.50 | 0.25 |
| 44025 | 1h54m | 0.35 | 0.17 | 0.22 | 0.00 | 0.90 | 0.00 | 0.42 |
| 46025 | 1h59m | 0.10 | 0.58 | 0.67 | 0.00 | 0.81 | 0.00 | 0.00 |

Mean of the last 9 rounds (steps 30000–46000): train 0.27, knownwords 0.37, novel1 0.31, novel2 0.07, heldout1 1.06, heldout2 0.21, heldout3 0.20.

## Losses

| step | train audio (mean over 5k steps) | val audio | val duration |
|---|---|---|---|
| 0–5k | 0.595 | – | – |
| 20k | 0.573 | – | – |
| 35k | 0.564 | 0.604 | 0.090 |
| 45k | 0.560 | 0.625 | 0.087 |

006 ended at val audio 0.882 / duration 0.103 on the same 99 val clips.

## Answer

Not measurable in 2 hours with this probe. Short unseen sentences stay in the 0.0–0.5 band of 006 with the same round-to-round noise; the long sentence (heldout1) never left 0.8–1.5. The training sentence, memorized in 006 (0.00), is now 0.1–0.6: it is 1 clip in 33k instead of 1 in 4.9k. Validation audio loss dropped from 0.88 to 0.60 by step 35k and is flat since, while train audio loss is still falling slowly (0.595 → 0.560): the model fits the audio better, the 7-sentence WER probe cannot resolve it.

## Learned

- Prep scales: 45 h in 7 min. Vocab and val copied from the earlier prep keep the checkpoint loadable and the val set comparable.
- lr 2.5e-4 on a checkpoint that ended at 1.25e-4 costs ~30 min of regression (train 0.50, novel2 0.86 at step 2000) before it recovers. A continuation should keep or lower the lr.
- 7 sentences × 1 draw per round is too noisy to see a change of ~0.1 WER. Comparing runs needs WER over the whole val set (99 sentences, several draws) at fixed checkpoints.

## Not tested

- Longer budget on the same data (val loss flat from 35k, train loss still falling: unclear if more steps help).
- train-clean-360 (further 190 h).
- Whole-val-set WER on 006 `ttl.pt` vs 010 `ttl.pt` / `best.pt`.
- Other reference speakers.
