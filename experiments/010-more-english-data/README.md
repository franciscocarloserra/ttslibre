# 010 more english data

**Question.** Starting from the 006 checkpoint (6.5 h, 40 speakers), does adding LibriTTS-R train-clean-100 (~54 h, 247 speakers) improve held-out sentences in a 2-hour run? Long sentences (`heldout3`) and `knownwords` were the weak spots of 006.

**Data.** `datasets/libritts-r/prep-clean100`: dev-clean + train-clean-100 (CC BY 4.0, attributed track). `prepare.py` (mels, 8 workers) → `encode.py` (latents with the 002 AE, dev-clean latent stats). Vocab and val split are copied from the dev-clean prep so the character table and the held-out sample sentences are identical to 006; clips with characters outside that vocab are dropped.

**Changes vs 006.** Data only, plus what a 2-hour continuation needs: warm start from 006 `devclean/ttl.pt` (step 160000), `ttl.lr` 2.5e-4 (006 ended at 1.25e-4 after two halvings), halve every 30k steps, budget 120 min, failure stop at 20 min if the training sentence WER > 0.8, 4 partial checkpoints.

**Test sets (samples every 2000 steps, ~5 min; ~22 rounds).** Same as 006, speaker 1988 reference clip: `train`, `knownwords`, `novel1`, `novel2`, `heldout1..3`.

**Reading.** Compare the WER table with 006's last rounds (`../006-generalize-40-speakers/RESULTS.md`). `heldout3` and `knownwords` lower = more data generalizes better. Everything flat = 2 h is not enough to see the effect of the data.

**Run (once).**
```bash
./venv/bin/python prepare.py && ./venv/bin/python encode.py
./venv/bin/python train.py --run clean100
```

**Result.** no measurable WER change in 2 h; val audio loss 0.88 → 0.60 (`RESULTS.md`).
