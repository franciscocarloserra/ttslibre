# Candidate architectures for a sub-100M TTS

Facts from the linked sources, read on 2026-09-05. Raw dumps of every source are under `research/` (papers as text in `research/papers/`, web pages in `research/web/`). "Not disclosed" means the source does not say it. Anything marked *unverified* comes from a secondary summary and was not confirmed in a primary source.

Scope: architectures with a credible open lineage at or under ~100M parameters. Models above that (F5-TTS 336M, Chatterbox, CosyVoice, OmniVoice, Qwen3-TTS, Dia, Orpheus) are out of scope for this project's parameter budget and are not tabulated.

## Summary table

| | StyleTTS 2 + iSTFTNet (Kokoro lineage) | SupertonicTTS (flow matching over compressed latent) | Pocket TTS / CALM (continuous audio LM) | Matcha-TTS (flow matching to mel) | ZipVoice (flow matching, Zipformer) | MOSS-TTS-Nano (discrete-token LM) |
|---|---|---|---|---|---|---|
| Params | 82M (Kokoro) | 44M (paper v1), 66M (v1 release), ~99M (v3) | 100M (90M backbone + 20M VAE) | 18.2M acoustic + 13.9M HiFi-GAN | ~123M | ~100M incl. ~20M tokenizer |
| Text input | Phonemes (phonemizer/espeak-ng; Kokoro: misaki) | Raw characters | SentencePiece 4k subword tokens | Phonemes (phonemizer/espeak-ng) | Phonemes (Emilia) or characters (LibriTTS) | Not disclosed |
| Aligner | Learned in-model (text aligner pretrained, then monotonic differentiable upsampling); phoneme durations needed | None external; cross-attention alignment; utterance-level duration only | Forced alignments required for training data (tooling shipped) | None external; MAS-style monotonic alignment learned | None external; average-upsampling initial alignment, infilling | None (autoregressive) |
| Acoustic target | Mel or direct waveform via iSTFTNet/HiFi-GAN decoder | 24-dim continuous latent, ~14 Hz after 6x temporal compression, 44.1 kHz | 32-dim VAE latent, 12.5 Hz, 24 kHz | 80-bin mel | Mel, Vocos vocoder | Discrete codec tokens, 12.5 Hz, 48 kHz stereo |
| Vocoder | iSTFTNet (LJSpeech model) or HiFi-GAN | Own Vocos-style latent decoder (part of the 44M) | Own VAE decoder (part of the 100M) | HiFi-GAN | Vocos (LibriTTS-trained) | Own tokenizer decoder |
| Style / expressiveness levers | Style vector from a reference (acoustic + prosodic style encoders); optional style diffusion; predicted pitch, energy, durations per phoneme; Kokoro voices are stored style vectors | Reference-speech embedding (zero-shot); CFG scale; NFE; Supertonic 3 adds `<laugh>` `<breath>` `<sigh>` tags; speed | Voice prompt (zero-shot cloning); sampling temperature; CFG baked into student | None beyond dataset; single speaker in paper | Reference prompt; CFG; NFE | Reference audio (cloning); sampling |
| Training data reported | StyleTTS 2 paper: LJSpeech 24 h, VCTK, LibriTTS train-clean-460 (245 h). Kokoro: "a few hundred hours", partly synthetic from closed TTS | Autoencoder: 11,167 h (public + internal). Text-to-latent: 945 h (LJSpeech, VCTK, Hi-Fi TTS, LibriTTS). v2/v3: not disclosed | Released model: 88k h (AMI, Earnings22, GigaSpeech, SPGISpeech, TED-LIUM, VoxPopuli, LibriHeavy, Emilia). Recipe example: HiFiTTS-2, 2,000 h or 31,700 h | LJSpeech 24 h | Emilia 100k h; LibriTTS 585 h for ablations | Not disclosed |
| Training compute reported | StyleTTS 2: 4x A40; LJSpeech 100 + 60 epochs, batch 16. Kokoro: ~1000 A100-80GB hours, ~$1000 | AE 1.5M iters batch 128 on 4x RTX 4090; text-to-latent 700k iters batch 64 (Ke=4) on 4x 4090; duration predictor 3k iters on 1x 4090 | Teacher 400k steps: 1x H100 ~22 h or 8x H100 ~10.5 h; distill 200k steps 1x H100 ~7.3 h. Consumer GPU: batch 16 x accum 4 in ~16 GB | 500k updates on 2x RTX 3090, batch 32 | 1M updates batch 4k s (Emilia); 60k updates (LibriTTS); GPUs not stated | Not disclosed |
| Training code | Yes, MIT (yl4579/StyleTTS2). Kokoro's own recipe: not released | No (inference only, MIT). Experiment 001 in this repo reimplements it | Yes, MIT (kyutai-labs/pocket-tts, `training/`) | Yes, MIT | Yes, Apache-2.0 | Fine-tuning code, Apache-2.0 |
| Weights license | Kokoro Apache 2.0 | OpenRAIL-M (use restrictions; derivatives inherit them) | CC-BY-4.0 | MIT (LJSpeech checkpoint) | Apache-2.0 | Apache-2.0 (card metadata) |
| CPU latency evidence | Third-party 2026-06-22, 4-core Xeon 8272CL: Kokoro ONNX mean RTF 0.571, first audio 0.649 s; PyTorch 0.787 / 1.211 s. iSTFTNet paper: vocoder alone ×20-68 real time on a 2.7 GHz i7 laptop | Supertonic 3 card: RTF 0.200 on 16-thread CPU. Third-party same bench: 2-step RTF 0.178, first audio 0.417 s (UTMOS 1.53); 5-step 0.316 / 0.672 s (UTMOS 4.37). Paper v1: RTF 0.02 on 4090, 0.05 on 3090 (GPU) | README: ~200 ms first chunk, ~6x real time on MacBook Air M4 using 2 cores; ~2.3-2.5x on 4-vCPU x86 VM. Paper: faster than real time on Apple M3 and Core Ultra 7 165H | GPU only in paper (RTF 0.015-0.038 on 3090). No CPU figure | Single thread Xeon 8457C: 16 NFE RTF 9.55; distill 8 NFE 2.42; 4 NFE 1.22 (all slower than real time single-threaded) | "streaming on a 4-core CPU"; ONNX build "single core M4". No RTF figure |
| Intelligibility evidence | Kokoro WER 1.93 on LibriSpeech test-clean (Pocket paper, Whisper-large-v3) | Paper v1: WER 2.64 / CER 0.83 on LS-clean, WER 2.41 on LS-PC-clean, 32 NFE | WER 1.84 same protocol; recipe: 0.76-0.94 % with Granite ASR on cross-sentence set | WER 2.09 (MAT-10, Whisper medium, LJSpeech test) | See paper Table I (not extracted) | Not disclosed |

Sources per column:
- StyleTTS 2: `docs/references/papers/styletts2-2306.07691.pdf`; https://github.com/yl4579/StyleTTS2 (MIT). iSTFTNet: `docs/references/papers/istftnet-2203.02395.pdf`; https://github.com/rishikksh20/iSTFTNet-pytorch (Apache 2.0). Kokoro: https://huggingface.co/hexgrad/Kokoro-82M.
- SupertonicTTS: `docs/references/papers/supertonictts-2503.23108.pdf`; https://github.com/supertone-inc/supertonic; https://huggingface.co/Supertone/supertonic-3.
- Pocket TTS / CALM: https://arxiv.org/abs/2509.06926 (appendix D, F); https://github.com/kyutai-labs/pocket-tts (README and `training/README.md`); https://huggingface.co/kyutai/pocket-tts.
- Matcha-TTS: https://arxiv.org/abs/2309.03199; https://github.com/shivammehta25/Matcha-TTS.
- ZipVoice: https://arxiv.org/abs/2506.13053; https://github.com/k2-fsa/ZipVoice.
- MOSS-TTS-Nano: https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M; https://github.com/OpenMOSS/MOSS-TTS-Nano; report arXiv 2603.18090 (not read).
- CPU benchmark: https://heyneo.com/blog/kokoro-supertonic-inflect-nano-cpu-tts-benchmark (third party, 2026-06-22).

## Per-candidate notes

### 1. StyleTTS 2 + iSTFTNet (Kokoro lineage)

- Two-stage training. Stage 1: text encoder, text aligner (pretrained ASR-style aligner), pitch extractor, style encoder, decoder. Stage 2: duration and prosody predictors, style diffusion, SLM adversarial loss with a frozen WavLM discriminator. Kokoro dropped diffusion and ships only the decoder path.
- Needs: phonemizer (espeak-ng is GPL-3; Kokoro uses misaki, Apache 2.0, with espeak-ng fallback), a pretrained text aligner and pitch extractor (the repo ships them, trained on LibriTTS), phoneme-level durations from the aligner.
- Expressiveness: the richest explicit control surface of the six. Style vector is a real, editable object (this is what kokoro-voicelab already exploits), plus per-phoneme pitch, energy, duration. Style diffusion can sample diverse prosody from text alone.
- Cost: cheapest known recipe to a usable model (Kokoro: ~$1000 of A100 time on a few hundred hours). Two-stage training with several auxiliary models is the complexity price.
- Risk for this project: the aligner, pitch extractor and phonemizer are extra dependencies to license-audit and to retrain for Spanish. GPL espeak-ng cannot be linked into an MIT runtime; misaki covers en, es and a few others.

### 2. SupertonicTTS (flow matching over a compressed latent)

- Three parts: speech autoencoder (mel 228 bins at 44.1 kHz, hop 512 -> 24-dim latent at ~86 Hz, then 6x temporal compression to ~14 Hz), text-to-latent flow-matching module with ConvNeXt blocks and cross-attention, utterance-level duration predictor (~0.5M params).
- No G2P, no aligner, no phoneme durations. Character input. This is the simplest data pipeline: audio plus transcript only.
- Expressiveness: reference embedding only in the paper. Supertonic 3 added expression tags and a "Voice Builder" that derives a style from reference audio. No per-phoneme controls. The third-party review pasted by the user (`research/user-notes/`) rates Supertonic 3 expressive control 1/10 and voice quality below Kokoro; that is an opinion, not a measurement.
- Cost: paper AE trained 1.5M iterations on 11k hours, 4x 4090. Text-to-latent 700k iterations on 945 h, 4x 4090. Total GPU time not stated. Context-sharing batch expansion (Ke=4) is the trick that makes alignment converge from characters.
- Training code: none upstream. Experiment 001 in this repo is a reimplementation; the AE is the expensive part and Vocos (MIT) is the shortcut used there.
- Quality gate: WER 2.64 at 32 NFE on LS-clean in the paper. Fewer NFE on CPU trades quality hard (third-party bench: 2-step UTMOS 1.53 vs 5-step 4.37 for Supertonic 3).

### 3. Pocket TTS / CALM (continuous audio language model)

- Autoregressive Transformer over continuous VAE latents (12.5 Hz, 32-dim, Mimi-derived VAE with WavLM semantic distillation), MLP consistency head predicts the next frame in one step. 24-layer 313M teacher distilled to a 6-layer 90M student with CFG folded in.
- Text as SentencePiece subwords, fed as a prefix. Training data must be word-aligned; Kyutai ships the aligner tooling.
- Expressiveness: voice prompt and temperature. No style vector, no prosody controls. Multilingual (en, fr, de, es, pt, it) since 2026-05-04.
- This is the only sub-100M candidate with a published, reproducible, MIT training recipe that reports numbers on a CC BY dataset: on 2,000 h of HiFiTTS-2 the 24-layer teacher reaches WER 0.94 %, speaker sim 0.929, UTMOS 4.32 after 400k steps (about 22 h on one H100). Distillation to 6 layers costs nothing on WER in their table.
- Caveats: the released weights were trained on 88k h that includes non-commercial sources (GigaSpeech, Emilia); a CC0 model must retrain from scratch. Teacher needs ~41.5 GiB at batch 64 on one GPU, or batch 16 x accum 4 in ~16 GB; the 3090 fits the latter. Streaming first-chunk latency ~200 ms on an M4, above this project's "well under 100 ms" target (*README figure; not measured here*).

### 4. Matcha-TTS

- Encoder-decoder, OT-CFM decoder to mel, U-Net with Transformer blocks, HiFi-GAN vocoder. 18.2M acoustic params. Phoneme input, learned monotonic alignment, no external aligner. MIT. Single-speaker LJSpeech in the paper; multi-speaker VCTK config exists in the repo (*unverified*).
- Expressiveness: none built in. Would need a style encoder added.
- Cheapest to train (500k updates, 2x 3090). No CPU numbers published; vocoder is a separate model.
- Position: a minimal flow-matching baseline, useful to calibrate the bake-off, not a winner candidate on expressiveness.

### 5. ZipVoice

- Flow matching with a Zipformer estimator, F5-style infilling, Vocos vocoder. ~123M, above budget; a smaller config would have to be trained. Apache-2.0, full training code, character or phoneme input.
- CPU: slower than real time on a single thread even distilled at 4 NFE (RTF 1.22). Multi-thread and ONNX/INT8 paths exist but no numbers were found.
- Position: strong open recipe, but the parameter count and CPU RTF both miss the targets as published.

### 6. MOSS-TTS-Nano and other discrete-token LMs at ~100M

- Autoregressive LM over codec tokens at 12.5 Hz, 20 languages, Apache-2.0, released 2026-04-10. Training data and compute not disclosed. Only fine-tuning code.
- Similar closed-recipe entries: KittenTTS (15M/40M/80M, Apache-2.0 code, architecture and data not disclosed, https://github.com/KittenML/KittenTTS); Vui 100M (MIT, "40k hours", sources not disclosed, superseded by a 300M model, https://huggingface.co/fluxions/vui).
- Position: no reproducible recipe, so not bake-off candidates. Useful as CPU baselines only.

## Baselines already on this machine

- Kokoro-82M: Apache 2.0, WER 1.93 on LS test-clean (Pocket paper), RTF ~0.57 ONNX on 4 cores (third party).
- Supertonic 3: OpenRAIL-M, ~99M, RTF 0.200 on 16 threads (card), 0.178-0.316 on 4 cores depending on steps (third party).

## Recommendation for the bake-off

Run three candidates on the same 100 clean hours, same text set, same eval script (`experiments/001-*/eval.py`):

1. **Supertonic lineage** (experiment 001, already in progress). Simplest data pipeline, character input, the project's stated preference. Weakest published control surface.
2. **StyleTTS 2 decoder-only (Kokoro recipe)** with iSTFTNet. Best expressiveness levers and the cheapest known path to Kokoro-level intelligibility. Needs misaki (Apache) not espeak-ng, and its own aligner and pitch extractor retrained on the CC0 data.
3. **Pocket TTS recipe** (CALM), trained from scratch on the CC0 data with the MIT `training/` code. The only candidate with published small-data numbers on a CC BY corpus and a 3090-sized config. Expressiveness is the weak spot; the bake-off should show whether its intelligibility margin is worth adding a style path later.

Matcha-TTS is optional as a 20M sanity baseline if compute allows.

**Single comparison metric: Whisper WER on the held-out sentence set**, exactly as `AGENTS.md` defines it, reported next to Kokoro and Supertonic 3 measured the same way. Params, CPU RTF (8 threads, single sentence) and training VRAM are reported alongside but do not decide. Expressiveness is not measurable by one number yet; the winner must expose a reference or style embedding that a later voice tool can edit, and that is a hard filter, not a score.

## Open questions

- Exact per-candidate param counts at the 100 h scale depend on config; the table gives the published sizes.
- CPU first-audio latency under 100 ms has no published evidence for any candidate on a laptop CPU at full quality. The closest are Supertonic 3 2-step (0.417 s on 4 cores, poor UTMOS) and Pocket TTS (~200 ms on M4). This target needs measuring here, not citing.
- Pocket TTS six-language variants use 24 layers; whether a 6-layer student reaches the same Spanish quality is not published.
