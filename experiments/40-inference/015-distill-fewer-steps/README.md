# 015 distill fewer steps

**Short.** Distill the 014 teacher to fewer Euler steps without CFG: 4 passes give WER 0.20 vs 0.48 for the teacher at equal cost.

**Question.** Can the 014 checkpoint (`runs/teacher/ttl_5h04m_step105000.pt`) be distilled into a student that reads with the same WER in 4 Euler steps and a single pass per step (no classifier-free guidance), i.e. 32 → 4 vector-field runs per sentence?

**Why now.** In the browser PoC (`projects/browser-inference/ttslibre`) the 014 model is 14x faster than Supertonic 3 per vector-field run but 3x slower per sentence, only because it needs 16 steps x 2 passes (CFG) vs Supertonic's 5 x 1. Distillation is a post-process on a finished checkpoint; if it works it is a recipe for every later checkpoint, independent of generalization work.

**Fact to keep in mind.** 014 was trained with `p_uncond = 0.0`: the unconditional text/style embeddings were never trained (zero init). CFG 2.0 at synthesis (the `synth.cfg` default, used for the model card) therefore extrapolates against an untrained branch. Stage 0 measures whether CFG is doing anything at all.

**Method.**
- Stage 0, baseline (no training): WER grid of the teacher over `baseline.sentences` held-out sentences, `baseline.draws` draws each, for every (steps, cfg) in `baseline.grid`. Fixes the reference WER at 16/2.0 and what 8, 4, 2 steps cost, with and without CFG. If cfg 1.0 already matches, the guidance half of the problem is free.
- Stage 1, progressive distillation (Salimans & Ho 2022, applied to the flow-matching Euler sampler): student initialized from the teacher. One round halves the step count: for a random training sentence (text, reference from the 014 prep set, fresh noise), the teacher takes 2 Euler steps of size 1/N (with `distill.teacher_cfg`), the student takes 1 step of size 2/N with cfg 1 (unconditional inputs never used), loss = MSE on the resulting latent. Rounds `distill.rounds` (e.g. 16→8, 8→4). Each round has its own time budget, keeps partial checkpoints, and logs the same WER probe as 014 on `distill.probe_steps` steps.
- Stage 2, same WER grid as stage 0 on the student; whole-val WER on the best student.

**Success criterion** (`distill.success`): student at 4 steps / cfg 1.0 within `max_wer_delta` of the teacher at 16 steps / cfg 2.0 on the same sentences and draws. Secondary: the browser PoC re-exported with the student (`export_web.py`) reports the RTF.

**Not in scope.** Retraining 014, changing the architecture, quantization (not the bottleneck: the time is in the number of runs, not in the weights).

**Run.**
```bash
./venv/bin/python baseline.py            # stage 0, writes runs/baseline/grid.jsonl + grid.md
./venv/bin/python distill.py             # stage 1 (to be written after stage 0 is agreed)
```
