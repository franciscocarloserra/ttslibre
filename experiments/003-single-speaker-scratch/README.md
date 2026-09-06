# 003 single-speaker generalization

**Question.** Does the 002 recipe, trained on one full speaker, say sentences it never saw?

**Data.** LibriTTS-R dev-clean, speaker 1988: 185 training clips, 3 held-out clips (val split from `prepare.py`). Same AE and latents as 001/002.

**Recipe.** 002 `overfit4` settings (lr 1e-3, batch 16 x batch_expand 8, p_uncond 0, cfg 1.0). Budget: `ttl.max_minutes` 12 or `ttl.steps` 4000, whichever first. Sample every 200 steps (at least 10 sample rounds in the run).

**Samples (TensorBoard).** Each round synthesizes 1 training sentence (`train/`) and `ttl.sample_heldout_n` held-out sentences (`heldout1/`, `heldout2/`). Per sentence: `<name>/text`, `<name>/original` (step 0), `<name>/generated` (per step), `<name>/whisper` (transcript), scalar `wer/<name>`. `wer/heldout_mean` is the criterion.

**Stop.** `ttl.stop_wer` (0.3) on the mean of the last `ttl.stop_wer_window` (3) held-out rounds; else time/steps.

**Success.** `wer/heldout_mean` < 0.3. If it stays >= 1.0 while `wer/train` drops, the model memorizes but does not generalize at this data size.

**Run.**
```bash
./venv/bin/python train.py --run spk1988            # add --resume to continue from runs/spk1988/ttl.pt
../panel.py (run from experiments/: ./004-words-or-sentences/venv/bin/python panel.py)  # http://localhost:7807
```
TensorBoard: see repo AGENTS.md (add `003:experiments/003-single-speaker-scratch/runs` to `--logdir_spec`).

**Result.** pending (see `RESULTS.md`)
