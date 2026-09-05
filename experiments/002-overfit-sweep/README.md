# 002 overfit sweep

**Goal.** Get an intelligible overfit as fast as possible with the 001 architecture, and find out which optimizer knobs make the text-audio alignment appear. If no learning-rate setting reaches intelligibility, the bug is in the code, not in the compute.

**Data.** Same prep as 001 (LibriTTS-R dev-clean). Speaker 1988, first 16 utterances (`data.speaker`, `data.max_utts`). The sample sentence is a training sentence, so this measures memorization only.

**Criterion.** Whisper WER of the sample sentence, logged every 100 steps (`sample/wer` in TensorBoard). Intelligible = WER under 0.2. Kokoro and Supertonic score ~0 on the same sentence.

**Config.** `config.json`, overridden per run with `--set key=value`; each run writes `runs/<name>/config.effective.json`. Changes from 001: batch_expand 8, p_uncond 0, cfg 1.0, warmup 100, grad_clip 5, 1500 steps.

**Runs.**
```bash
./venv/bin/python train.py --run lr5e-4 --set ttl.lr=5e-4
./venv/bin/python train.py --run lr1e-3 --set ttl.lr=1e-3
./venv/bin/python train.py --run lr2e-3 --set ttl.lr=2e-3
```
TensorBoard over 001 and 002: see repo AGENTS.md. Resume a run: same command plus `--resume`.

**Result.** See `RESULTS.md`. Run `overfit4` (4 utterances, lr 1e-3) reached WER 0.10-0.15 on the training sentence at steps 750-1000, ~3 min.

**Verdict.** Architecture viable: character input aligns to audio without G2P. Generalization untested (experiment 003).
