# 008 results

Run `sabato60`: 60 clips (225 s) of one Spanish speaker, warm start from 006 (English), lr 5e-4, 10 min, 3442 steps. Stopped by budget. Partial checkpoints at 1000/2000/3000 steps.

| step | elapsed | train | heldout1 | heldout2 | novel1 | novel2 | english |
|---|---|---|---|---|---|---|---|
| 500 | 1m23 | 0.40 | 1.33 | 1.20 | 1.25 | 1.00 | 1.40 |
| 1000 | 2m50 | 0.00 | 1.33 | 1.60 | 0.88 | 1.33 | 1.00 |
| 2000 | 5m48 | 0.00 | 1.33 | 1.00 | 0.75 | 1.11 | 1.10 |
| 3000 | 8m46 | 0.00 | 1.00 | 1.00 | 1.12 | 1.56 | 1.00 |

Whisper at step 3000, unseen sentences: "Dado el billete y después me cuándo me quedaron.", "Pero el astro más positivo sería que la típica tiene." (real Spanish words, wrong sentence). English sentence: Whisper heard Russian-like babble.

**Answer.** In 10 minutes the model memorizes the training clip (WER 0 from step 1000), copies the speaker's voice (by ear, from the first round), and produces Spanish-sounding output: Whisper transcribes fluent Spanish words, sometimes Italian. It does not say the requested text on any unseen sentence (WER ≥ 0.75). English is gone after 500 steps.

**Learned.**
- Voice and phonetics come almost free from the 006 weights; letter-to-sound rules for Spanish do not. Style is copied, language is learned.
- 60 clips are memorized, not generalized: same picture as 004 in English (16 clips).
- Catastrophic forgetting of English is immediate when training on Spanish only.

**Not tested.** More data (the full 45-min recording); mixing English batches to keep English; detected-language rate as a phonetics metric (Whisper server does not return it yet).
