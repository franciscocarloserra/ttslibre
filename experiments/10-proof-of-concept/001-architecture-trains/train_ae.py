"""Stage 1: latent autoencoder mel -> 24-d latent -> mel (L1). Then encodes every mel to latents
and stores channel-wise latent stats. Usage: train_ae.py [--smoke]"""
import json, os, random, sys, time
import torch
from common import load_config, LatentAE, P, count_params

c = load_config()
smoke = "--smoke" in sys.argv
a, d = c["ae"], c["data"]
prep = P(d["prep_dir"])
run = P(os.path.join("runs", "ae_smoke" if smoke else "ae"))
os.makedirs(run, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = [json.loads(l) for l in open(os.path.join(prep, "train.jsonl"))]
val = [json.loads(l) for l in open(os.path.join(prep, "val.jsonl"))]
epochs, batch = (c["smoke"]["ae_epochs"], c["smoke"]["batch"]) if smoke else (a["epochs"], a["batch"])
all_rows = rows
if smoke:
    rows = rows[: batch * 4]
model = LatentAE(c).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a["lr"])
scaler_dtype = torch.bfloat16 if a["amp"] and dev == "cuda" else torch.float32
print(f"ae params={count_params(model)/1e6:.2f}M train={len(rows)} epochs={epochs} batch={batch} dev={dev}", flush=True)


def load(r, crop):
    m = torch.load(os.path.join(prep, "mels", r["id"] + ".pt")).float()
    if m.shape[1] > crop:
        s = random.randint(0, m.shape[1] - crop)
        m = m[:, s:s + crop]
    return m


def collate(batch_rows, crop):
    ms = [load(r, crop) for r in batch_rows]
    T = max(m.shape[1] for m in ms)
    out = torch.full((len(ms), ms[0].shape[0], T), float(torch.log(torch.tensor(d["log_clamp"]))))
    mask = torch.zeros(len(ms), 1, T)
    for i, m in enumerate(ms):
        out[i, :, :m.shape[1]] = m
        mask[i, :, :m.shape[1]] = 1
    return out.to(dev), mask.to(dev)


step, t0 = 0, time.time()
log = open(os.path.join(run, "progress.log"), "a")
for ep in range(epochs):
    random.shuffle(rows)
    for i in range(0, len(rows), batch):
        mel, mask = collate(rows[i:i + batch], a["crop_frames"])
        with torch.autocast(dev, dtype=scaler_dtype, enabled=scaler_dtype != torch.float32):
            rec = model(mel)
            loss = ((rec - mel).abs() * mask).sum() / (mask.sum() * mel.shape[1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % a["log_every"] == 0 or smoke:
            line = f"ep={ep} step={step} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s vram={torch.cuda.max_memory_allocated()/2**20 if dev=='cuda' else 0:.0f}MiB"
            print(line, flush=True); log.write(line + "\n"); log.flush()
    torch.save(model.state_dict(), os.path.join(run, "ae.pt"))

# validation L1
model.eval()
with torch.no_grad():
    vl, n = 0.0, 0
    for i in range(0, len(val), batch):
        mel, mask = collate(val[i:i + batch], 10 ** 9)
        vl += (((model(mel) - mel).abs() * mask).sum() / (mask.sum() * mel.shape[1])).item(); n += 1
    print(f"val_l1={vl/max(n,1):.4f}", flush=True)
    # encode all mels -> latents, collect stats
    os.makedirs(os.path.join(prep, "latents"), exist_ok=True)
    s1 = torch.zeros(c["latent"]["dim"], device=dev); s2 = torch.zeros_like(s1); cnt = 0
    for r in all_rows + val:
        m = torch.load(os.path.join(prep, "mels", r["id"] + ".pt")).float()[None].to(dev)
        z = model.encode(m)[0]
        torch.save(z.half().cpu(), os.path.join(prep, "latents", r["id"] + ".pt"))
        s1 += z.sum(1); s2 += (z ** 2).sum(1); cnt += z.shape[1]
    mean = s1 / cnt; std = (s2 / cnt - mean ** 2).sqrt()
    torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(prep, "latent_stats.pt"))
json.dump({"params": count_params(model), "val_l1": vl / max(n, 1), "steps": step,
           "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0,
           "seconds": time.time() - t0}, open(os.path.join(run, "summary.json"), "w"), indent=1)
print("done", flush=True)
