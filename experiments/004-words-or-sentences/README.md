# 004 word-level generalization

**Question.** Does the model learn to pronounce words, or only whole sentences? Warm start from the aligned 002 checkpoint, overfit 16 short clips, then test a new sentence built only from training words.

**Data.** Speaker 1988, clips under `data.max_clip_seconds` (4 s), first `data.max_utts` (16). Init weights: `ttl.init_from` (002 `overfit4`).

**Samples every 100 steps.** `train` = a training sentence (control). `knownwords` = new sentence, every word appears in the 16 clips. `heldout1` = sentence with words never seen.

**Stops.** `ttl.max_minutes` 5; success when held-out WER mean < `ttl.stop_wer`; failure if `train` WER > `ttl.fail_wer` after `ttl.fail_after_minutes`.

**Reading.** `wer/train` low, `wer/knownwords` low: learns words. `wer/train` low, `wer/knownwords` ~1: memorizes sentences. `wer/train` ~1 at 3 min: recipe broke, stop.

**Run.**
```bash
./venv/bin/python train.py --run w16
```

**Result.** pending (`RESULTS.md`)
