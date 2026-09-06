# 007 style refs

**Question.** Does the 006 model transfer voice zero-shot? Same model, same sentences, three reference clips from different speakers (1988 = the training-time reference, 6313 and 7976 never used as reference).

**No training.** `eval_refs.py` synthesizes 4 unseen sentences × 3 references × 3 repeats, transcribes with Whisper, writes wavs and a WER table to `runs/refs/`. Whether the *voice* actually changes is judged by ear in the panel (advanced → ref clip path).

**Reading.** WER similar across references = the style path does not break intelligibility. Voice audibly different per reference = zero-shot style works.

**Run.**
```bash
./venv/bin/python eval_refs.py
```

**Result.** intelligible with new references on 3/4 sentences (WER 0.2–0.5 vs 0.14); the 3-word sentence breaks (`RESULTS.md`).
