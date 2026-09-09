# 10 proof of concept

Does the Supertonic-lineage prototype (latent autoencoder + text-to-latent flow matching, `001`) train at all, align text to audio, and memorize? Reading order: 001 (pipeline builds and trains), 002 (first alignment: 4 sentences, WER 0.15), 004 (learns words, not only sentences), 005 (overfits 185 clips of one speaker). Checkpoints from here (`004/runs/w16/ttl.pt`) were the warm start of the scaling runs.
