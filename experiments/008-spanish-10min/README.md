# 008 spanish 10min

**Question.** Warm-started from 006 (English), does 10 minutes of training on 60 Spanish clips push the model towards Spanish sounds?

**Data.** `datasets/rioplatense`: one recording (Ernesto Sabato reading *El túnel*, YouTube b0GKWQi5d7w, 45 min), cut at silences into 60 clips (225 s, 1.5–10 s each), transcribed by Whisper. Copyrighted: local proof of concept only, never distributed. `segment.py` → `prepare.py` (vocab = English vocab + `¿ á é í ñ ó ú ü`) → `encode.py` (latents with the existing AE, English latent stats).

**Changes vs 006.** Warm start copies the old character embeddings and leaves the 8 new rows random. WER ignores accents. lr 5e-4, samples every 500 steps, budget 10 min, no failure stop, 4 partial checkpoints.

**Test sets.** `train` = a training clip. `heldout1..2` = clips of the same recording not trained on. `novel1..2` = hand-written Spanish. `english` = a 006 held-out sentence, to measure forgetting. Reference clip: `sabato_0005.wav`.

**Reading.** Spanish WER falling below ~0.5 on `train` and moving on `heldout`/`novel` = direction is right. `english` rising = forgetting.

**Run (once).**
```bash
./venv/bin/python train.py --run sabato60
```

**Result.** voice and Spanish sounds in minutes, no Spanish text yet; English forgotten (`RESULTS.md`).
