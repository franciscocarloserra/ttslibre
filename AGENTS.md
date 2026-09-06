# AGENTS.md

Read before working in this repo. README.md is the pitch. This file is the operating brief.

## Goal of the prototype phase

A minimal, working, end-to-end TTS training pipeline that we own: data prep, train, export, evaluate. Not good yet. Working. Built as `experiments/001-*/`.

## Repo structure

- `experiments/NNN-<slug>/` : every experiment in its own numbered directory with a `README.md` (goal, config, result, verdict), a `config.json` with every numeric knob, and its scripts. Scratch, downloads and checkpoints under `<exp>/runs/` and `<exp>/data/`, git-ignored. **Nothing runs outside an experiment directory.**
- `datasets/` : symlink to `/media/usuario/hdd-unencrypted/ttslibre/datasets/`, one subdir per source (`libritts-r/raw`, `libritts-r/prep`, ...) plus `manifests/` (provenance, no audio). The single place where data lives. Git-ignored. Experiments reference it by relative path in their `config.json`. Polished datasets and manifests are published to Hugging Face (namespace `ttslibre`, nothing uploaded yet) from here.
- `docs/references/` : upstream implementations as submodules. Read-only.
- `docs/` : sourced facts (`ARCHITECTURES.md`, `DATASETS.md`, `REFERENCES.md`), coordination rules. `docs/research/` is a local-only, git-ignored scratch of raw fetched sources so nothing is looked up twice.
- `docs/references/papers/` : local PDF cache, git-ignored (third-party copyright). Links in `docs/REFERENCES.md`.

## Acceptance metric

**Whisper WER on held-out sentences.** Synthesize a fixed set of test sentences, transcribe with the local Whisper server, compute word error rate against the input text. Lower is better. Report next to two baselines measured the same way on the same sentences: Kokoro-82M and Supertonic 3 (both installed on this machine, see below).

Secondary, always reported: parameter count, real-time factor on CPU (single sentence, 8 threads), peak VRAM during training.

## Hard constraints

- Under 100M parameters. Externalize every numeric knob to the experiment's `config.json`. No hardcoded hyperparameters, paths or thresholds.
- All dependencies MIT/Apache/BSD compatible. No code copied from OpenRAIL or non-commercial sources. Reading `docs/references/` for ideas is fine; copying from `docs/references/styletts2` or `docs/references/istftnet` (MIT) is fine with attribution.
- Datasets: CC0 or public domain for the main track; CC BY only in the attributed track. Candidates and caveats in `docs/DATASETS.md`; license conclusions in `docs/LICENSES.md`. Experiment 001 uses LibriTTS-R dev-clean (CC BY, attributed track).
- Python 3, PyTorch (2.10, CUDA available), ONNX export required by the end of the phase.
- Nothing is published, pushed or made public without the user's explicit authorization. Commit locally. Do not push.

## Machine rules

- One RTX 3090, 24 GB, shared. Persistent services live on it: TTS 6971 (audio.cpp 6972), Whisper 6969, llama-server 8081. **Never kill, restart or reconfigure any of them.** Never start a second TTS or STT model server.
- Check `nvidia-smi` before every run. Size batch and model to the free VRAM. If the free VRAM is too small for a meaningful run, do everything else, run a tiny smoke test that fits, and report the blocker with the exact numbers instead of waiting.
- Never touch PipeWire, WirePlumber or PulseAudio.
- Never delete files. Move to `todelete/`.
- Ports for anything ephemeral: 78xx.
- Checkpoints, logs and samples go under `experiments/NNN-*/runs/` (symlink to `/media/usuario/hdd-unencrypted/ttslibre/NNN/runs/`, git-ignored). Root disk has ~5 GB free: nothing large on it.
- TensorBoard over all experiments, port 7806: `./venv/bin/python -m tensorboard.main --logdir_spec 001:<001 runs>,002:<002 runs> --port 7806 --reload_interval 5`. Per run: scalars `train/*`, `val/*`, `sample/wer`; audio `sample/original`, `sample/generated`; text `sample/text`, `sample/whisper`.

## Local tools

- Whisper server: port 6969, client and server code in `~/projects/know-how/local-whisper/`. Read its README for the HTTP contract.
- Supertonic 3 baseline: `~/projects/know-how/local-tts/tts "text"` speaks; `tts -o out.ogg "text"` saves. Server on 6971.
- Kokoro baseline: see `~/projects/kokoro-voicelab/` for how it is invoked on this machine.
- Reference implementations: `docs/references/`. Run `git submodule update --init` if empty.

## Deliverables of the prototype phase

0. `docs/LICENSES.md`: license audit (LJSpeech, Kokoro-82M output, Supertonic 3 output) before any synthetic data is generated.
1. `experiments/001-*/README.md`: what was built, how to run it, results table (WER, params, RTF, VRAM) for prototype vs Kokoro vs Supertonic 3, and what failed.
2. `experiments/001-*/config.json` with every knob.
3. `experiments/001-*/prepare.py`, `train.py`, `synth.py`, `export.py`, `eval.py`. Small files, stdlib plus torch plus what is strictly needed.
4. A few synthesized samples under `experiments/001-*/samples/` (small, git-tracked, ogg or mp3).
5. Local commits with clear messages. No push.

## Style

Ultraminimal. Only what the requested step needs. English everywhere. Report facts, not hopes: if a run failed, say so with the numbers.

## Experiment usability rules (from the user)

- Every experiment answers one question, stated in its `README.md`; only one thing changes vs the previous experiment. Results go in `RESULTS.md` (facts from logs, a WER table per sample round, "what we learned", "not tested") so a later agent can reason at a high level without re-reading logs.
- Whisper WER is the cut criterion. Standard definition, unbounded (insertions count); 0 = perfect, ≥1.0 = unintelligible. Normalize digits to words before comparing (Whisper writes "19", texts say "nineteen"); Kokoro, Supertonic and the original recording must score 0 on the sample sentence, that is the calibration check. Numbers, abbreviations and other edge cases are deferred; words only for now.
- Training log lines, in this order, no labels that need a manual: human-readable elapsed time, `(step)`, current WER per sample sentence, then losses named in plain words ("audio loss", "duration loss"). No VRAM or learning rate on every line.
- A run has a hard time budget in config (`ttl.max_minutes`), a success stop (held-out WER) and, when possible, an early failure stop. The user should never wait 15 minutes to learn a run failed.
- At least 10 sample rounds per run. Each round logs to TensorBoard, per sentence: text, original recording (once), generated audio, Whisper transcript, WER. Sample sentences: one training sentence (control) plus held-out ones.
- Interactive testing of any checkpoint with custom text is done through the panel (`experiments/panel.py`, port 7807, lists all experiments' checkpoints, model stays loaded on GPU), not through the CLI or TensorBoard.
- One aggressive run beats a three-command sweep. Give the user one command.
- Reports to the user: Spanish, short, plain words, no cryptic parameter names.

## Partial checkpoints (from the user)
Every training run keeps at least 4 partial checkpoints spread over its time budget (`ttl.keep_checkpoints`, weights only, `runs/<run>/ttl_<elapsed>_step<N>.pt`), plus `best.pt` (best held-out WER) and `ttl.pt` (latest, resumable). Reason: 006 stabilized at 3 h of a 7 h run and the 3 h weights were gone; partial checkpoints let the next experiment start from any point of the curve.

## Experiment names (from the user)
Directory names state the question the experiment answers, in plain words: `NNN-<question-as-a-slug>` (e.g. `006-generalize-40-speakers`). No cryptic names. Renamed on 2026-09-06; run dirs on the HDD keep the number only (`/media/usuario/hdd-unencrypted/ttslibre/NNN/runs`).
