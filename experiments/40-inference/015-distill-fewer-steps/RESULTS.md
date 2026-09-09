# 015 results

Teacher: `runs/teacher/ttl_5h04m_step105000.pt` (014, 5h04m checkpoint). Student: `runs/distill/r16to8/best.pt`, one progressive-distillation round 16→8 (teacher_cfg 2.0, batch 64, lr 1e-4, 5453 training sentences), trained on a Vast 3090, cut by hand at step ~9000 (45 min). `best.pt` = step 1500, chosen by the training probe (3 held-out sentences x 3 draws at 4 steps / cfg 1.0). All WER grids below: same 20 held-out sentences (`baseline.seed` 0), 3 draws, one reference voice, whisper on the rig (6969), measured on the rig 3090.

## Method

1. Stage 0, teacher grid (`baseline.py`, `runs/baseline/grid.md`): WER over (steps, cfg) to fix the reference (16/2.0) and the cost of fewer steps with and without CFG.
2. Stage 1, distillation (`distill.py`, `runs/distill/r16to8/`): student initialized from the teacher; per training sentence the teacher takes 2 Euler steps with CFG 2.0, the student 1 step without CFG, MSE on the latent. Probe every 500 steps.
3. Stage 2, student grid (`baseline.py --section eval --run .../best.pt`, `runs/student_eval/grid.md`): same sentences, [4,1.0], [8,1.0], [2,1.0].
4. Stage 2b, teacher at equal cost (`--section baseline_equal`, `runs/baseline_equal/grid.md`): the teacher run with the same number of vector-field passes as the student (CFG doubles the passes), so the comparison is at equal generation time.

## Probe during distillation (`runs/distill/r16to8/progress.log`, 3 sentences x 3 draws, 4 steps / cfg 1.0; teacher ref at 16/2.0 = 0.059)

| step | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000-5500 | 6000 | 6500 | 7000 | 7500 | 8000 | 8500 | 9000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe WER | 0.151 | 0.111 | 0.108 | **0.049** | 0.062 | 0.154 | 0.12-0.14 | 0.056 | 0.142 | 0.062 | 0.093 | 0.111 | 0.148 | 0.077 |

The probe oscillates 0.05-0.15 from step 1500 on: 9 draws is too few to pick a checkpoint. Loss kept falling (0.053 → 0.036-0.049).

## Teacher vs student, equal cost (20 sentences x 3 draws)

| vector-field passes | teacher (steps/cfg) | WER | gen s/draw | student (steps/cfg) | WER | gen s/draw |
|---|---|---|---|---|---|---|
| 2 | 1/2.0 | 0.947 | 0.05 | 2/1.0 | 0.302 | 0.05 |
| 2 | 2/1.0 | 0.567 | 0.05 | | | |
| 4 | 2/2.0 | 0.479 | 0.08 | 4/1.0 | **0.202** | 0.08 |
| 4 | 4/1.0 | 0.288 | 0.10 | | | |
| 8 | 4/2.0 | 0.266 | 0.13 | 8/1.0 | 0.216 | 0.13 |
| 8 | 8/1.0 | 0.240 | 0.16 | | | |
| 16 | 8/2.0 | 0.132 | 0.29 | | | |
| 16 | 16/1.0 | 0.217 | 0.29 | | | |
| 32 (reference) | 16/2.0 | **0.118** | 0.60 | | | |

Teacher 2/1.0 measured twice (0.587 in stage 0, 0.567 in stage 2b): run-to-run noise of the grid is ~0.02.

## Findings

- **Success criterion not met**: student 4/1.0 = 0.202 vs teacher 16/2.0 = 0.118, delta 0.084 > `max_wer_delta` 0.05.
- **At equal cost the student wins everywhere**: 0.202 vs 0.479 at 4 passes, 0.216 vs 0.266 at 8, 0.302 vs 0.947 at 2. The student at 4 passes (0.08 s/draw, 7.5x faster than the reference) matches the teacher at 16/1.0 (0.217, 0.29 s) and beats the teacher at 4/2.0 (0.220).
- **CFG is what the student absorbed**: the teacher's gap between cfg 1.0 and 2.0 (0.10 WER at 16 steps) is gone in the student; the remaining gap to 0.118 is step count, not guidance.
- **Student at 4 steps beats 8** (0.202 vs 0.216) although it was distilled for 8: sign that `best.pt` (step 1500) is under-trained and the probe pick is noisy, not that 8 is a converged optimum. A local 8→4 round on this checkpoint is not worth it.
- **Single-sentence listening is misleading**: easy short sentences read at WER 0 with the teacher at 2/2.0 in the panel, while the same setting gives 0.479 over the 20 held-out sentences (e.g. "¿Sabés escanear libros?" mean 1.67 at 2/1.0). Judge on the grid, listen on the worst sentences.
- Supertonic 3 reference: 5 steps (`total_step`) x 1 pass in the core, 8 in its CLI. The student at 4-8 passes is in the same range.

## Next

- Continue the 16→8 round longer, with the probe on the 20 grid sentences (not 3) to pick the checkpoint; then re-run stage 2. Only after that, an 8→4 round.
- Panel (`experiments/panel.py`): compare mode now takes a checkpoint, steps and cfg per side; synth info reports `ckpt`, `steps`, `cfg`.
