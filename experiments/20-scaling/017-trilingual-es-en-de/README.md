# 017 trilingual es en de

**Short.** 016 plus German: Spanish + English + German interleaved, 4 h each, from scratch, 4 h budget: does a third language fit in the same 19.8M model without slowing the other two?

**Question.** Trained from scratch on an interleaved 1/3-1/3-1/3 mix, does the 016 recipe reach held-out WER < 0.3 in all three languages within 4 h, and are the es/en curves no slower than in 016 at equal wall time?

**Why.** 016 shows whether two languages share the model. Three is the smallest test of "one small multilingual model": German adds a new alphabet fragment (ä ö ß; ü is already in the Spanish vocab) and a single-speaker corpus, the wish 016 could not fulfil (one speaker per language).

**Data.** German: Thorsten-Voice v02 (OpenSLR 95, one male speaker, ~23 h, 22.05 kHz, CC0-1.0), LJSpeech-style layout (`metadata.csv` `id|text|...` + `wavs/`). *Download URL and inner folder names not verified this session (no web access)*: expected `https://www.openslr.org/resources/95/thorsten-de_v02.tgz`; `prepare_de.py` reads the layout from `data.sources.de` (`metadata`, `wav_dir`, `delimiter`, `id_col`, `text_col`), adjust those to whatever the archive contains. `prepare_de.py` resamples to 24 kHz, writes mels + latents with the shared AE (`runs/ae -> 014`), the manifest `datasets/manifests/thorsten-de.jsonl` and `vocab.json` = Spanish vocab + new German chars appended (ids unchanged). `prepare.py` is 016's, unchanged in logic, over three sources: per language the fewest speakers reaching `data.hours_per_lang`, hours trimmed to the smallest language, rows interleaved es, en, de, es, en, de; texts wrapped in `<es>`/`<en>`/`<de>` tags, the 6 tag tokens appended to the vocab.

**Method.** 016 `train.py` with two WER changes: numbers are spelled with `num2words` in the language of the row's tag (016 used `ttl.lang` for every row) and `ß -> ss` before the accent strip; the kept character class is `ttl.wer_chars`. Probes: `heldout_{es,en,de}1..3`, `novel_es`, `novel_en`, `novel_de`; stop criterion on their mean over `stop_wer_window` rounds. Budget `ttl.max_minutes` = 240, batch calibrated by the `run.sh` preflight from `ttl.batch` = 48 (016 ran at 90-100 % GPU with 48). All probes use the Spanish reference voice (`synth.ref_clip`): zero-shot voice, language from the tag.

**Success.** All three languages under 0.3 held-out WER within 4 h; secondary: es/en curves not slower than 016 at equal wall time.

**Run.**
```bash
mkdir -p ../../../datasets/thorsten-de/raw && cd ../../../datasets/thorsten-de/raw && wget https://www.openslr.org/resources/95/thorsten-de_v02.tgz && tar xf thorsten-de_v02.tgz && cd -   # verify the URL and the layout first
./venv/bin/python prepare_de.py     # once: mels, latents, manifest, vocab for the German source
./venv/bin/python prepare.py        # once: mixed prep (symlinks, no re-encoding)
cd ../.. && ./run.sh 20-scaling/017-trilingual-es-en-de mix4h3
```
