# 006 english dev-clean

**Question.** With all of LibriTTS-R dev-clean (40 speakers, 4881 clips, 6.5 h) and an 8-hour budget, does the model start to say sentences it never saw?

**Changes vs 005.** All speakers (`data.speaker` ""), warm start from 004 `w16`, `ttl.lr` 5e-4 (2e-3 diverged in 005), warmup 500, lr halves every 60k steps, `ttl.max_minutes` 480, NaN guard (stops the run, last checkpoint stays on disk), `best.pt` saved whenever the held-out mean WER improves.

**Test sets (samples every 2000 steps, ~6 min; ~80 rounds).** All synthesized with the speaker 1988 reference clip.
- `train` = a training sentence (the 002 one).
- `knownwords` = new sentence built from training-set words.
- `heldout1..3` = val sentences never seen in training (speakers 1988, 1462, 2035).
- `novel1`, `novel2` = hand-written everyday sentences.

**Stops.** 8 h; or held-out mean WER < 0.2 for 3 rounds (success); or train WER still > 0.8 after 60 min (failure); or NaN.

**Run (once).**
```bash
./venv/bin/python train.py --run devclean
```

**Result.** works: unseen short sentences at WER 0–0.3 after ~3 h; long sentences still weak (`RESULTS.md`).
