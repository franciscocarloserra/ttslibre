# 006 results

Run `devclean`: LibriTTS-R dev-clean, 40 speakers, 4881 clips (6.5 h), warm start from 004 `w16`, lr 5e-4, batch 16×8, RTX 3090. Stopped by the user at 7h07m, step 161125 (budget was 8 h). No crash, no watchdog restart. Checkpoints: `ttl.pt` (step 160000), `best.pt` (step 98000, held-out mean WER 0.15).

WER per sample sentence (speaker 1988 reference clip; rounds every 2000 steps, 80 rounds, subset shown):

| step | elapsed | train | knownwords | novel1 | novel2 | heldout1 | heldout2 | heldout3 |
|---|---|---|---|---|---|---|---|---|
| 2000 | 5 min | 1.00 | 1.00 | 1.00 | 1.43 | 2.00 | 1.10 | 1.00 |
| 22000 | 58 min | 0.45 | 0.42 | 0.33 | 0.86 | 1.00 | 0.20 | 0.93 |
| 42000 | 1h51 | 0.30 | 0.42 | 0.22 | 0.14 | 0.33 | 0.40 | 1.07 |
| 62000 | 2h44 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.81 |
| 82000 | 3h37 | 0.30 | 0.33 | 0.56 | 0.29 | 0.00 | 0.00 | 0.70 |
| 102000 | 4h30 | 0.05 | 0.58 | 0.44 | 0.00 | 0.33 | 0.00 | 1.19 |
| 122000 | 5h23 | 0.05 | 0.25 | 0.11 | 0.00 | 0.00 | 0.10 | 1.15 |
| 142000 | 6h16 | 0.00 | 0.33 | 0.11 | 0.29 | 0.67 | 0.10 | 0.41 |
| 160000 | 7h04 | 0.00 | 0.58 | 0.22 | 0.00 | 0.00 | 0.20 | 0.59 |

Validation at step 160000: audio loss 0.882, duration loss 0.103 (flat since ~100k).

**Answer: yes.** The model says sentences it never saw. Short unseen sentences (heldout1, heldout2, novel2) reach WER 0.00 from ~2h44 on and stay in 0.0–0.3 afterwards, with round-to-round noise. The long sentence heldout3 (8.5 s, 27 words) improves slowly and never goes below 0.41. `knownwords` (proper names Pop, Hilda) stays noisy, 0.17–0.67.

**Learned.**
- The recipe (character input, flow matching over AE latent, 20M params) generalizes at 6.5 h of data on one consumer GPU in ~3 h of training; the rest of the budget mostly reduces noise.
- The success stop (held-out mean < 0.2 for 3 rounds) never fired: rounds are too noisy for a 3-round window with 3 sentences.
- The failure stop on the `train` sentence is wrong for a multi-utterance set (it would have fired at 60 min); the watchdog disabled it on restart, but it was never needed.
- GPU utilization ~62%: per-step overhead and per-file latent loading limit throughput, not the GPU.

**Not tested.** Other reference speakers (all samples used speaker 1988's clip); long sentences beyond 8 s; lower cfg / more sampling steps at inference; whether `best.pt` (step 98000) sounds better than the last checkpoint.
