# Experiments

Four groups by the question each experiment answers. Numbering `NNN` is global across groups and never reused: "014" means the same experiment everywhere (configs, logs, RESULTS, HDD run dirs `/media/usuario/hdd-unencrypted/ttslibre/NNN/runs`). Each experiment: `NNN-<question-as-a-slug>/README.md` (question, method), `RESULTS.md` (measured, verdict), `config.json` (every knob), scripts. `runs/` and `data/` are git-ignored. Each group has a `README.md` with its reading order and the checkpoint that came out of it.

| group | question | experiments |
|---|---|---|
| `10-proof-of-concept/` | does the architecture train, align and memorize? | 001, 002, 004, 005 |
| `20-scaling/` | does it generalize with more data, hours, speakers, languages? | 003, 006, 008, 009, 010, 011, 012, 014 |
| `30-voices/` | how is the voice controlled at synthesis? | 007, 013 |
| `40-inference/` | how fast can it read at the same WER? | 015 |

New experiment: next global number, in the group whose question it answers (a Spanish+English mixed run goes to `20-scaling`). Cross-experiment references are relative paths (`../../<group>/NNN-slug/...`); `venv` and `common.py`/`synth.py` are symlinks into the experiment they inherit from.

Shared tools at this level: `panel.py` (checkpoint browser + synthesis, port 7807, `--exp <group>/NNN-slug`), `voice.py` (voicepacks into `voices/`), `check_panel.sh`.
