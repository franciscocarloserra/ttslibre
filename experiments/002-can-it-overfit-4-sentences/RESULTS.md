# 002 results

Facts from `runs/overfit4/` (progress.log, summary.json, config.effective.json). 2026-09-05.

## Run `overfit4`

| Setting | Value |
|---|---|
| Data | LibriTTS-R dev-clean, speaker 1988, first 4 utterances |
| Model | 001 architecture, 19.8M params (DP 0.54M), AE reused from 001 (`runs/ae/ae.pt`) |
| Optimizer | lr 1e-3, warmup 100, grad_clip 5, batch 16, batch_expand 8, p_uncond 0 |
| Sampling | 16 Euler steps, cfg 1.0, reference clip `ref_default.wav` (speaker 1988) |
| Steps / time / VRAM | 1000 steps, 186 s, 1.36 GiB peak |

Whisper WER on the training sentence ("He was nineteen years old, short and broad backed, with a close cropped, flat head, and a wide, flat face."):

| Step | WER | Whisper transcript |
|---|---|---|
| 50-300 | ≥1.0 | unrelated speech ("Thank you very much for joining us...") |
| 650 | 0.35 | He was 19 years old, short and wide, flat head, and a wide, flat face. |
| 750 | 0.10 | He was 19 years old and broad-backed with a close-cropped flat head and a wide flat face. |
| 850 | 0.15 | He was 19 years old, short and broad-backed, with a close head and a wide, flat face. |
| 1000 | 0.15 | He was 19 years old, short, ground-backed, with a close-cropped flat head and a wide flat face. |

WER oscillates between 0.10 and 0.55 from step 650 on (not monotonic). Final val fm 1.15 (val set is not held out here: 4 training rows are used).

## What we learned

- The 001 architecture (character input, no phonemizer, flow matching over the compressed AE latent, style from a reference clip) learns text-audio alignment. Criterion WER < 0.2 met at steps 750, 850, 1000.
- The knobs that made it work vs 001's failed overfit (185 clips, WER 1.0 at 2200 steps): 4 utterances instead of 185, batch_expand 8, p_uncond 0, cfg 1.0, lr 1e-3. Which of these mattered individually was not isolated.
- WER on the sample is noisy step to step; a single reading is not a stopping criterion. Use the best-of-last-N or an average.
- Cost of viability check: ~3 min on GPU, 1.4 GiB VRAM.

## Not tested

- Generalization: no held-out sentence, no held-out speaker. This run measures memorization only.
- Whether the same recipe scales to more utterances (001's 185-clip run did not align in 2200 steps with weaker settings).
