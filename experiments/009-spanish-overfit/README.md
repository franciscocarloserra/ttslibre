# 009 spanish overfit

**Question.** From scratch (no English warm start), how fast does the model reach something recognizable in Spanish on one speaker's full recording?

**Data.** `datasets/rioplatense`: the whole Sabato recording (45 min), cut at silences, Whisper transcripts. Copyrighted: local proof of concept only, never distributed. `segment.py` → `prepare.py` → `encode.py` (existing AE, English latent stats).

**Changes vs 008.** No `init_from` (random init, English vocab kept as base ids). Budget 120 min, samples every 2000 steps, 4 partial checkpoints. Reference clip: `sabato_0005.wav`.

**Test sets.** `train` = a training clip. `heldout1..2` = clips of the same recording not trained on. `novel1..2` = hand-written Spanish. Accent is judged by ear (native listener), Whisper WER measures the text.

**Reading.** First ~hour is alignment (003 did not align in 12 min from scratch). `train` WER falling = alignment found. `heldout`/`novel` moving = rules emerging. Accent: compare by ear with 008's warm-started samples.

**Run (once).**
```bash
./venv/bin/python train.py --run sabato-scratch
```

**Result.** pending (`RESULTS.md`)
