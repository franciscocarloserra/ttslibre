# 014 results

Run `es8h`: stopped by the success criterion at 5h13m / step 108000 (mean held-out probe WER of the last 3 rounds < 0.2), from scratch, batch 16, lr 5e-4 halved every 60k. 19.8M params, peak VRAM 4.1 GB. Validation at step 100000: audio loss 0.496, duration loss 0.114.

## Probe WER during training (`runs/es8h/progress.log`, 1 draw per sentence per round)

| elapsed | step | train | heldout mean (3) | novel mean (2) | english |
|---|---|---|---|---|---|
| 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1h10m | 24000 | 0.17 | 0.78 | 0.59 | 1.71 |
| 1h46m | 36000 | 0.00 | 0.84 | 0.36 | 1.29 |
| 2h21m | 48000 | 0.50 | 0.38 | 0.42 | 1.14 |
| 2h55m | 60000 | 0.17 | 0.48 | 0.24 | 1.57 |
| 3h30m | 72000 | 0.00 | 0.10 | 0.06 | 1.29 |
| 4h04m | 84000 | 0.33 | 0.45 | 0.00 | 1.29 |
| 4h39m | 96000 | 0.00 | 0.32 | 0.19 | 1.57 |
| **5h04m** | **105000** | 0.33 | **0.05** | **0.00** | 1.43 |
| 5h13m (final) | 108000 | 0.00 | 0.21 | 0.28 | 0.86 |

- Answer to the question: yes. From scratch, 8 h of Argentinian Spanish is enough for held-out and novel sentences to drop under 0.3 from ~3h30m on. Spanish letter-to-sound is learned faster than English was (006: 5 h of data, WER 0.39 on whole-val).
- `english` stays >= 0.86 the whole run: expected, the model never sees English.
- Rounds are noisy (1 draw, 3+2 sentences); the 5h04m partial checkpoint is the best round and is the one published.

## Whole-val WER (`eval_val.py`, 286 val sentences x 3 draws, single reference voice)

Only the 1h01m checkpoint was measured (699 of 858 draws, run stopped by the user): mean 0.88, median 0.87, 83% of draws above 0.5. Consistent with the probe at that point (0.78). The other checkpoints were not evaluated; the published checkpoint is chosen on the probe, not on whole-val.

## Model-card samples (`card_samples.py`, checkpoint 5h04m, `runs/card/`)

6 dataset voices (3 female, 3 male, `card.voices`) x 8 sentences (4 hand-written + 4 held-out) x 3 draws, steps 16, cfg 2.0. Best draw per (voice, sentence) kept if WER <= 0.1: 48 of 48 kept, mean WER 0.005. Published under `samples/` (ogg) and on the Hugging Face card. Voicepacks written to `../voices/<voice>.pt`.

## What we learned

- One reference voice at eval time is a pessimistic measurement: with a voice chosen from the dataset and 3 draws, almost every sentence is intelligible.
- The success stop fired on a noisy 3-round window; a partial checkpoint (5h04m) beats the final on the same probe.

## Not tested

- Whole-val WER of the partial checkpoints and of `best.pt` / `ttl.pt`.
- Warm start from the English checkpoint (012) instead of scratch.
- Cloning the author's own voice (`voices/francisco*.pt`, local, excluded from the repo until the clone is good; then it joins the voicepack).
