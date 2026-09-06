# 001 results

Facts from `runs/` logs. 2026-09-05.

## Data prep (`prepare.py`)

LibriTTS-R dev-clean: 4980 utterances, 40 speakers, 6.64 h, char vocab 67. Mels 24 kHz / hop 256 / 100 bins.

## Stage 1: latent AE (`train_ae.py`, `runs/ae`)

6.9M params, 12240 steps, 449 s, 667 MiB peak. val L1 0.2256. AE reconstruction of a clip is transcribed correctly by Whisper (mel -> 24-d latent -> mel -> Vocos is not the bottleneck).

## Stage 2: text-to-latent (`train.py`)

- `runs/ttl` (speaker 1988, 185 clips, batch 16, batch_expand 4, p_uncond 0.05, lr 5e-4, cfg 2.0): WER 1.0 or worse through step 2200. Output is fluent-sounding but unrelated speech. Note: `runs/ttl/ttl.pt` was later overwritten by the multi-speaker run below; the log is the reliable record.
- `runs/todelete/ttl_multispeaker_step4000` (all 40 speakers, batch 32, resumed): step 9350, train fm 0.60, 4.5 GiB VRAM, no alignment (WER ≥ 1.0).

## Export / eval

`export.py` produces the 5 ONNX components (style_encoder, text_encoder, vector_estimator, duration_predictor, latent_decoder), max abs diff vs PyTorch ≤ 1.4e-6. `eval.py` (WER vs Kokoro and Supertonic, CPU RTF) exists but was not run on an aligned model.

## What we learned

- Pipeline (prep, AE, training, sampling, Whisper WER in TensorBoard, ONNX export) works end to end.
- With these settings, alignment did not emerge within the step budget on 185 clips or on 6.6 h multi-speaker. Alignment was first obtained in experiment 002 with a 4-utterance overfit and stronger settings (see `../002-can-it-overfit-4-sentences/RESULTS.md`).
