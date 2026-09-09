# 013 style vectors

**Short.** Voice as an editable style tensor: save, load, interpolate without the reference encoder at synthesis.

**Question.** Can a voice be handled as an editable tensor (saved, loaded, interpolated) without the reference encoder at synthesis, and does that keep intelligibility? No training: code plus a measurement on the 012 checkpoint.

**Code (checkpoint-compatible, no new parameters).** `common.py`: `TTL.style(zref, rmask)` returns the voice as tensors (`style` 50×128 for text and vector field, `dp` 8×64 for durations); `encode` and `dp` accept them precomputed. `synth.py`: `Synth.style(ref)` accepts a wav, a voicepack (`voices/x.pt`, latents) or a style pack (`voices/x.style.pt`, tensors). `../voice.py`: `style` (voicepack → style pack for a checkpoint), `mix` (interpolate two voices), `test` (Whisper WER on the 007 sentences).

**Measurement (`pipeline.sh`, chained after 012).** On 012's final checkpoint: voicepack vs style pack of the same clip (must be identical WER, same math), and the 50/50 mix of 6209 and 4137 (`style.mix_t`). Results in `../voices/*.test.json`, summary in `RESULTS.md`.

**Reading.** Style pack WER == voicepack WER: plumbing is right. Mix WER in the same band: the style space is smooth enough for sliders. Mix WER ≥ 1: interpolation leaves the manifold, sliders need to be constrained (per-token or PCA directions).

**Run.** `./pipeline.sh` (waits for 012 `all done`).

**Result.** pending (`RESULTS.md`)
