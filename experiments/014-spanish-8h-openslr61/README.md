# 014 spanish 8h openslr61

**Question.** From scratch (no English warm start), does 8 hours on OpenSLR 61 (8 h, 44 Argentinian Spanish speakers, CC BY-SA 4.0) produce a model that reads unseen Spanish sentences (held-out WER < 0.3)?

**Data.** `datasets/openslr61`: 5739 studio clips (48 kHz, median 4.9 s, short read prompts), `line_index_{female,male}.tsv` transcripts. Resampled to 24 kHz, mels with the 012 settings, latents with the existing AE and the English latent stats. 5% held out (all speakers seen). Vocab = English vocab + `¡ ¿ Á É Ú á é í ñ ó ú ü` (appended, English ids unchanged). Provenance manifest in `datasets/manifests/openslr61-es-ar.jsonl`.

**Changes vs 012.** No `init_from`: random init, English vocab kept as base ids so a later warm start stays compatible. WER ignores accents, `num2words` in Spanish. Batch 16, lr 5e-4 (006 from-scratch setting) halved every 60k steps, 8 h budget, 8 partial checkpoints, samples every 3000 steps, no failure stop. Reference voice: `arf_00295_00000740990.wav`.

**Test sets.** `train` = a training clip ("El pijama de rayas es azul"). `heldout1..3` = held-out clips. `novel1..2` = hand-written Spanish. `english` = an English sentence, to measure forgetting.

**Reading.** `heldout`/`novel` WER falling below 0.3 = Spanish letter-to-sound learned. `english` rising = forgetting (expected, not the question). Whole-val WER on every partial checkpoint at the end picks the best one.

**Run (once).**
```bash
./pipeline.sh    # prepare + encode already done; trains run es8h for 8 h, then whole-val eval of every checkpoint
```

**Result.** yes: held-out and novel probe WER under 0.3 from 3h30m, 0.05 / 0.00 at 5h04m; stopped by the success criterion at 5h13m. Published checkpoint: `runs/es8h/ttl_5h04m_step105000.pt`. See `RESULTS.md`.
