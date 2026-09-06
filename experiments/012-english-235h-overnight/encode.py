"""Encode prepared mels to latents with the existing AE (no AE training). Latent stats are copied from data.latent_stats."""
import json, os, shutil
import torch
from common import load_config, LatentAE, P

c = load_config(); d = c["data"]; prep = P(d["prep_dir"]); dev = "cuda" if torch.cuda.is_available() else "cpu"
ae = LatentAE(c).to(dev).eval(); ae.load_state_dict(torch.load(P("runs/ae/ae.pt"), map_location=dev))
os.makedirs(os.path.join(prep, "latents"), exist_ok=True)
rows = [json.loads(l) for f in ("train.jsonl", "val.jsonl") for l in open(os.path.join(prep, f))]
with torch.no_grad():
    for i, r in enumerate(rows):
        m = torch.load(os.path.join(prep, "mels", r["id"] + ".pt")).float()[None].to(dev)
        torch.save(ae.encode(m)[0].half().cpu(), os.path.join(prep, "latents", r["id"] + ".pt"))
        if i % 2000 == 0: print(f"{i}/{len(rows)}", flush=True)
shutil.copy(P(d["latent_stats"]), os.path.join(prep, "latent_stats.pt"))
print(f"encoded {len(rows)}")
