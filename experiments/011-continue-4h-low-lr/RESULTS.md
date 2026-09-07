# 011 results

Run `cont4h`: 4h00m, 86644 steps, init from 010 `ttl.pt`, lr 1e-4 (halved every 60k). Audio loss 0.566 at stop (010 ended ~0.555 with more noise; flat).

Whole-val Whisper WER (99 val sentences x 3 draws, `runs/evalval/*.summary.json`):

| checkpoint | mean WER | median | frac > 0.5 |
|---|---|---|---|
| 006 (base) | 0.386 | 0.350 | 0.28 |
| 010 final | 0.296 | 0.227 | 0.20 |
| 010 best | 0.284 | 0.227 | 0.17 |
| 011 at 2h | 0.294 | 0.200 | 0.17 |
| 011 final | 0.291 | 0.200 | 0.18 |

4 more hours at low lr on train-clean-100 gave no measurable gain over 010 (mean 0.291 vs 0.284-0.296): the 44.5 h dataset was saturated at this model size. Answer to the question: expand data, not training time. Followed by 012.
