# References

Facts below are taken from the linked sources on 2026-09-05. Anything not stated there is marked as not disclosed.

## Supertonic

**SupertonicTTS paper** (v1 research model, 44M params)
https://arxiv.org/abs/2503.23108 — Kim et al., submitted 2025-03-29, v3 2025-09-23.
- Three components: speech autoencoder (continuous latent), text-to-latent module with flow matching, utterance-level duration predictor.
- Low-dimensional latent with temporal compression, ConvNeXt blocks, raw character input (no G2P, no external aligner), cross-attention alignment.
- "Context-sharing batch expansion" for faster convergence and stable alignment.
- Training data: not specified in abstract. Read full paper for details.

**Supertonic v1 release** (66M params) — https://github.com/supertone-inc/supertonic
- ONNX, on-device, up to 167x real-time on consumer hardware.

**Supertonic 2** — released 2026-01-06, 5 languages. Code kept on branch `release/supertonic-2`.

**Supertonic 3** — released 2026-04-29, ~99M params, 31 languages.
- https://huggingface.co/Supertone/supertonic-3
- https://supertonic3.github.io/
- Weights: OpenRAIL-M. Code: MIT.
- RTF 0.200 on 16-thread CPU.
- Training data and compute: not disclosed. No paper published for v2 or v3.

## Kokoro

**Kokoro-82M** — https://huggingface.co/hexgrad/Kokoro-82M
- Apache 2.0 weights. Training code not released.
- Architecture: StyleTTS 2 + ISTFTNet decoder. Decoder only, no diffusion.
- Data: "a few hundred hours", public domain, Apache/MIT licensed, and synthetic audio from closed commercial TTS. Excludes audio from open TTS or cloning tools.
- Compute: about $1000 for 1000 A100-80GB hours total; v1.0 was 500 A100 hours.
- 8 languages, 54 voices in v1.0. No technical report exists on arXiv.

**Upstream papers Kokoro builds on**
- StyleTTS 2 — https://arxiv.org/abs/2306.07691
- ISTFTNet — https://arxiv.org/abs/2203.02395

## Takeaway

Both lineages show sub-100M models reaching usable quality on hundreds, not thousands, of hours. Compute is a four-figure budget. The open gaps are the training recipe and a fully licensed dataset.

## In this repo

Papers, under `references/papers/`:
- `supertonictts-2503.23108.pdf` (22 pages)
- `styletts2-2306.07691.pdf` (28 pages)
- `istftnet-2203.02395.pdf` (6 pages)

Implementations, as git submodules under `references/`:
- `supertonic/` — supertone-inc/supertonic, inference only, ONNX, MIT code.
- `kokoro/` — hexgrad/kokoro, inference only, Apache 2.0.
- `styletts2/` — yl4579/StyleTTS2, full training code. Kokoro is a fine-tune of this lineage.
- `istftnet/` — rishikksh20/iSTFTNet-pytorch, the vocoder Kokoro uses.

Run `git submodule update --init` after cloning.
