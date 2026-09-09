# 20 scaling

Does the model generalize to unseen sentences as data, hours and speakers grow, and across languages? English: 003 (one speaker, from scratch), 006 (40 speakers, 6.5 h), 010 (+54 h), 011 (continue at low lr), 012 (235 h overnight). Spanish: 008 (10 min warm-started from English), 009 (one speaker from scratch), 014 (8 h OpenSLR 61 from scratch, WER < 0.3 held-out). Reference checkpoints: English `012/runs/overnight/ttl_6h21m_step138000.pt`, Spanish `014/runs/es8h/ttl_5h04m_step105000.pt` (the 015 teacher). Next: mixed Spanish+English training run.
