# 016 mixed es en

**Short.** Spanish + English interleaved, 4 h + 4 h, fewest speakers per language, from scratch, 2 h budget: does one model learn both at the pace 014 learned Spanish?

**Question.** Trained from scratch on an interleaved 50/50 mix of Spanish (OpenSLR 61) and English (LibriTTS-R), does the 014 recipe reach held-out WER < 0.3 in both languages within the time the Spanish-only run needed to start falling (014: 0.38 at 2h21, 0.10 at 3h30)?

**Why.** 014 (8 h Spanish) and 012 (177 h English) landed at similar WER: at this model size data hours are not the limit. Before scaling anything, know whether two languages share the model without one slowing the other.

**Data.** `prepare.py` builds `datasets/mixed-es-en/prep` from the two existing preps (same AE, same latent stats, Spanish vocab = English vocab + 13 chars with unchanged ids; latents symlinked, nothing re-encoded). Per language the fewest speakers, most hours first, that reach `data.hours_per_lang` (one speaker per language was the wish: the largest speaker has 0.46 h in either corpus, so it is not possible at these hours). train/val rows alternate es, en, es, en so every batch is balanced in expectation.

**Method.** 014 `train.py` with two changes: held-out probes per language (`heldout_es1..3`, `heldout_en1..3`, stop criterion on their mean over `stop_wer_window` rounds) and GPU utilization from `nvidia-smi` on the log line every `gpu.util_every_logs` log lines (TensorBoard `train/gpu_util`). Budget `ttl.max_minutes` = 120, 4 partial checkpoints. English probes use the Spanish reference voice (`synth.ref_clip`): the voice is zero-shot, the language comes from the text and its tag.

**GPU utilization.** Target `gpu.util_target_pct` = 70. `smoke.sh` runs `--smoke` at each `gpu.batch_candidates` batch for `gpu.smoke_measure_seconds` and reports the mean utilization; the run is launched with the smallest batch that reaches the target (006 measured ~62% at batch 16: per-file latent loading and per-step overhead).

**Language tags.** Supertonic conditions on explicit tags (`<es>...</es>` as single tokens, `supertonic/core.py::_add_language_token`); this run does the same: `prepare.py` wraps every text in `<lang>...</lang>`, the 4 tag tokens are appended to the vocab (80 -> 84, so the char embedding is 4 rows larger; trained from scratch, nothing to resize), the tokenizer treats them as one id each and the WER normalizer strips them. Sample texts in `config.json` carry the tags.

**Iterations.** 1) `mix4h4h` (2026-09-09, aborted at 40 min, in `runs/todelete/mix4h4h_notag`): mistake, text entered with no language marker; every probe still WER > 1 at 40 min, GPU 15–25 % at batch 16. 2) `mix4h4h_tag` (2026-09-09, 2 h): same data and recipe with the tags above. Not met: at 2 h novel_en 0.43, novel_es 1.44, held-out 0.35–1.3; GPU 15–27 %. 3) `mix4h4h_b32`: latents cached in RAM (`train.py load_latent`) and batch 32, calibrated with `smoke.sh` (400 steps each: b16 47 %, b24 61 %, b32 70 %; target 70 ± 10 %). Inference tags the text programmatically (`Synth.tag`, default `synth.lang`, panel field `lang`).

**Success.** Both languages under 0.3 held-out WER within the 2 h; secondary: Spanish curve not slower than 014 at equal wall time.

**Run.**
```bash
./venv/bin/python prepare.py         # once
./smoke.sh                           # 60-step smoke per batch candidate, prints GPU util per batch
./venv/bin/python train.py --run mix4h4h --set ttl.batch=<chosen>
```
