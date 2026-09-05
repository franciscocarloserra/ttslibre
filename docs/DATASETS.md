# Dataset options

Facts from the linked sources, read on 2026-09-05. Raw dumps in `research/web/` and `research/papers/`. License rules for the two tracks are in `docs/COORDINATION.md`; the audit of what is already used is in `docs/LICENSES.md`. Not legal advice.

Two tracks:
- **Main track (CC0 / public domain).** Everything here can be redistributed, relabeled and published with the model as CC0.
- **Attributed track (CC BY 4.0).** Attribution required; derived datasets stay CC BY. CC BY-SA is listed separately because ShareAlike propagates to adaptations and does not fit either track cleanly.

Non-commercial sources (Emilia core, Expresso, EARS, GigaSpeech, VoxPopuli terms vary) are excluded and listed at the end only so nobody re-checks them.

## Main track: CC0 or public domain

| Dataset | Language | Hours | Speakers | Sample rate | License | Caveats | Source |
|---|---|---|---|---|---|---|---|
| LJSpeech | en | 24 | 1 | 22.05 kHz | Public domain (US) | Single speaker; audiobook prosody; some clips end mid-sentence | https://keithito.com/LJ-Speech-Dataset/ |
| LibriVox raw recordings | en, es, many | Not counted here; MLS found ~60k h of English readings with matching text | Thousands | Mostly 64 kbps MP3 at 44.1 kHz (*unverified*) | Public domain (LibriVox releases all recordings to the public domain; page not fetched this session, HTTP 403) | Requires own segmentation, alignment and transcript matching; variable mic quality; no per-clip provenance out of the box. HiFiTTS-2 and Libri-Light provide the download lists | https://librivox.org |
| Common Voice Scripted Speech 26.0 (2026-06) | en, es, 294 locales | 42,388 total / 28,893 validated across all languages. Per-locale en and es totals not obtained; one filtered subset (US English, male, validated) alone is 390 h / 5,705 speakers. Spanish at v17.0 was 2,220 h total, ~562 h user-validated | Tens of thousands | MP3, 48 kHz source (*unverified*) | CC0-1.0 | Crowd mic quality, background noise, short prompts, many speakers with seconds each; needs quality filtering (DNSMOS or similar) before TTS use. Distribution now via Mozilla Data Collective, not direct download | https://github.com/common-voice/cv-dataset ; https://mozilladatacollective.com |
| CSS10 Spanish (speaker "Tux") | es | 23.8 | 1 | Not stated on README | Apache-2.0 repackaging of public-domain LibriVox audio | Single male speaker, Galdós novels; effectively the Spanish LJSpeech | https://github.com/Kyubyong/css10 |
| Libri-Light (unlabelled) | en | 60k+ | Not stated | 16 kHz | Code MIT; audio public domain (LibriVox) | No transcripts; download list only. Repo archived 2025-07-17 | https://github.com/facebookresearch/libri-light |

Note on LibriVox-derived corpora: the *audio* is public domain, but the *packaging* (segmentation, transcripts, manifests) carries the packager's license. LibriTTS, LibriTTS-R, MLS, CML-TTS, Hi-Fi TTS, HiFiTTS-2 and Libriheavy are all LibriVox audio under CC BY 4.0 or Apache 2.0 packaging. Re-deriving segments and transcripts from raw LibriVox with this project's own pipeline yields CC0-eligible data; reusing their manifests does not.

## Attributed track: CC BY 4.0

| Dataset | Language | Hours | Speakers | Sample rate | License | Caveats | Source |
|---|---|---|---|---|---|---|---|
| LibriTTS | en | 585 | 2,456 | 24 kHz | CC BY 4.0 | Audiobook read speech; some noisy "other" subsets | https://www.openslr.org/60/ |
| LibriTTS-R | en | 585 (restored) | 2,456 | 24 kHz | CC BY 4.0 | Speech-restoration artifacts possible; used by experiment 001 | https://www.openslr.org/141/ |
| Hi-Fi TTS | en | 291.6 | 10 (>=17 h each) | 44.1 kHz | CC BY 4.0 | Few speakers, good for a high-bandwidth vocoder | https://www.openslr.org/109/ |
| HiFiTTS-2 (NVIDIA) | en | 31.7k (44.1 kHz subset), ~36.7k (22 kHz subset) | 5,000 | 22.05 / 44.1 kHz | CC BY 4.0 (manifests; audio fetched from LibriVox) | ASR transcripts with per-utterance WER; 4 TB for the 44 kHz set; the Pocket TTS recipe trains on it | https://huggingface.co/datasets/nvidia/hifitts-2 |
| Libriheavy | en | ~50k | Not stated | 16 kHz | Apache 2.0 (card) | Punctuated, cased transcripts; 16 kHz limits vocoder quality | https://huggingface.co/datasets/pkufool/libriheavy |
| Multilingual LibriSpeech (MLS) | en, es, de, nl, fr, it, pt, pl | en 44,659.74 train; es 917.68 train, ~10 dev, ~10 test | Not extracted | 16 kHz | CC BY 4.0 | 16 kHz; ASR-oriented segmentation, no punctuation | https://www.openslr.org/94/ ; https://arxiv.org/abs/2012.03411 |
| CML-TTS | es, de, nl, fr, it, pt, pl | es train 443.2 (279.2 male, 164.1 female), dev 5.7, test 4.8; total 3,176 | es 77 train (35 M, 42 F); total 613 | 24 kHz | CC BY 4.0 | Derived from MLS, re-segmented for TTS with punctuation; gender-unbalanced overall | https://www.openslr.org/146/ ; https://arxiv.org/abs/2306.10097 |
| VCTK 0.92 | en (accents) | ~44 | 110 | 48 kHz | CC BY 4.0 | Short sentences, no narrative context; p280/p315 mic issues, p315 transcripts lost | https://datashare.ed.ac.uk/handle/10283/3443 |
| Emilia-YODAS | en, zh, de, fr, ja, ko | en 92.2k | In-the-wild | Not stated | CC BY 4.0 | No Spanish. YouTube-CC sourced, ASR transcripts, variable quality (DNSMOS field provided) | https://huggingface.co/datasets/amphion/Emilia-Dataset |
| People's Speech | en | 30,000+ (CC-BY and CC-BY-SA subsets; clean 1.55M rows, dirty 5.53M rows) | Many | Not stated | CC-BY / CC-BY-SA per item | archive.org audio, misaligned transcripts acknowledged by the authors, US-accent bias; ASR grade, not TTS grade | https://huggingface.co/datasets/MLCommons/peoples_speech |

## CC BY-SA 4.0 (ShareAlike; separate decision)

| Dataset | Language | Hours | Speakers | Sample rate | License | Source |
|---|---|---|---|---|---|---|
| Google crowdsourced Latin American Spanish (SLR61 ar, 71 cl, 72 co, 73 pe, 74 pr, 75 ve) | es (6 dialects) | 37.79 total; ar 8.0, cl 7.2, co 7.6, pe 9.2, pr 1.0, ve 4.8 | 174 total | 48 kHz mono, studio-grade mic | CC BY-SA 4.0 | https://www.openslr.org/61/ (and 71-75); paper https://aclanthology.org/2020.lrec-1.801.pdf |

## Excluded: non-commercial or restricted

| Dataset | Why excluded | Source |
|---|---|---|
| Emilia (core, 101k h incl. en 46.8k) | CC BY-NC 4.0 | https://huggingface.co/datasets/amphion/Emilia-Dataset |
| Expresso (45.9 h, 4 speakers, 26 styles, 48 kHz) | CC BY-NC 4.0. The only studio expressive-style corpus found; would matter for the expressiveness goal if a licence exception were negotiated | https://huggingface.co/datasets/ylacombe/expresso |
| GigaSpeech, SPGISpeech, Earnings22, AMI, TED-LIUM, VoxPopuli | Used in Pocket TTS's 88k h mix; licenses not audited here, several are research-only or CC BY-NC-ND (*unverified*) | arXiv 2509.06926 appendix D |
| Supertonic 3 synthetic output | OpenRAIL-M derivative clause | `docs/LICENSES.md` |

## Recommendation

**English, main track.** LJSpeech (24 h) plus a self-derived LibriVox slice. Use the HiFiTTS-2 speaker and chapter list (CC BY) only as a pointer to which LibriVox chapters have clean 44.1 kHz audio and matching Gutenberg text, then re-segment and re-transcribe with the project's own pipeline so the resulting manifest is CC0. Target the first 100 clean hours from a few dozen speakers with >=2 h each. Common Voice English as a filtered supplement for speaker diversity once a quality filter exists.

**Spanish, main track.** CSS10 Tux (23.8 h, single speaker) plus a self-derived LibriVox Spanish slice; CML-TTS shows ~440 h of usable Spanish LibriVox exists from 77 speakers, so a CC0 re-derivation of the same chapters is feasible. Common Voice Spanish (hundreds of validated hours) as the diversity supplement.

**Attributed track.** LibriTTS-R (already in experiment 001), Hi-Fi TTS, CML-TTS Spanish. Keep the Google LatAm Spanish set out until the project decides whether ShareAlike is acceptable.

**Expressive data.** Nothing CC0 or CC BY with labelled styles was found. Expressiveness will have to come from prosody diversity in audiobook data and from the model's style path, or from recordings contributed under the rules in `docs/COORDINATION.md`.

## Not obtained this session

- Common Voice 26.0 per-locale en and es hours and speaker counts (Data Collective pages are per-subset; the aggregate JSON path was not found).
- LibriVox public-domain statement text (page returned HTTP 403).
- MLS per-language speaker counts (table did not extract cleanly from the PDF; see `research/papers/mls-2012.03411.txt`).
- Kyutai blog bodies (JavaScript-rendered); facts taken from the GitHub READMEs and the CALM paper instead.
