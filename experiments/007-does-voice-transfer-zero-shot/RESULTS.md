# 007 results

Model: 006 `devclean` step 160000. 4 unseen sentences × 3 references × 3 repeats, 16 steps, cfg 1.0. Whisper WER, mean of 3:

| ref | heldout1 | heldout2 | novel1 | novel2 | mean |
|---|---|---|---|---|---|
| 1988 (training-time ref) | 0.11 | 0.13 | 0.19 | 0.14 | 0.14 |
| 6313 | 1.44 | 0.20 | 0.26 | 0.48 | 0.59 |
| 7976 | 0.89 | 0.17 | 0.33 | 0.38 | 0.44 |

**Answer.** Style embeddings from other speakers keep the model intelligible on 3 of 4 sentences (WER 0.17–0.48), worse than the reference it was sampled with during training (0.14). The 3-word sentence `heldout1` breaks with new references (0.9–1.4: extra or wrong words). Whether the voice actually sounds like 6313/7976 is not measured here: listen in `runs/refs/*.wav` or the panel.

**Learned.** Zero-shot reference works partially without any training for it. Short inputs are the fragile case.

**Not tested.** Speaker similarity (no metric); references longer than 4 s; cfg > 1.
