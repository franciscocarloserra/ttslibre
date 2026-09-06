"""LibriTTS-R subset -> mels + manifest + vocab + provenance. Audio is never committed."""
import glob, json, os, random, sys
import torch, torchaudio, soundfile as sf
from common import load_config, Mel, Tokenizer, P

c = load_config()
d = c["data"]
raw, out = P(d["raw_dir"]), P(d["prep_dir"])
os.makedirs(os.path.join(out, "mels"), exist_ok=True)
mel = Mel(d)
rows, texts = [], []
wavs = sorted(glob.glob(os.path.join(raw, "*", "*", "*.wav")))
if not wavs:
    sys.exit(f"no wavs under {raw}")
for i, w in enumerate(wavs):
    uid = os.path.basename(w)[:-4]
    txt = w[:-4] + ".normalized.txt"
    if not os.path.exists(txt):
        continue
    text = open(txt).read().strip()
    x, sr = sf.read(w, dtype="float32"); a = torch.from_numpy(x).T if x.ndim == 2 else torch.from_numpy(x)[None]
    if sr != d["sample_rate"]:
        a = torchaudio.functional.resample(a, sr, d["sample_rate"])
    sec = a.shape[1] / d["sample_rate"]
    if not d["min_seconds"] <= sec <= d["max_seconds"]:
        continue
    m = mel(a.mean(0, keepdim=True))[0]
    torch.save(m.half(), os.path.join(out, "mels", uid + ".pt"))
    rows.append({"id": uid, "speaker": uid.split("_")[0], "text": text, "frames": m.shape[1], "seconds": round(sec, 3),
                 "source": d["source"], "license": d["license"], "path": os.path.relpath(w, raw)})
    texts.append(text)
    if i % 200 == 0:
        print(f"{i}/{len(wavs)}", flush=True)
random.Random(d["seed"]).shuffle(rows)
nv = max(1, int(len(rows) * d["val_fraction"]))
with open(os.path.join(out, "val.jsonl"), "w") as f:
    for r in rows[:nv]: f.write(json.dumps(r) + "\n")
with open(os.path.join(out, "train.jsonl"), "w") as f:
    for r in rows[nv:]: f.write(json.dumps(r) + "\n")
base = json.load(open(P(d["base_vocab"]))) if d.get("base_vocab") else ["<pad>"]  # keep the English ids, append new chars
Tokenizer(base + sorted({ch for t in texts for ch in t} - set(base))).save(os.path.join(out, "vocab.json"))
# provenance manifest (no audio), git-ignored, ships with the dataset on Hugging Face
prov = P("../../datasets/manifests")
os.makedirs(prov, exist_ok=True)
with open(os.path.join(prov, d["manifest"]), "w") as f:
    for r in sorted(rows, key=lambda r: r["id"]): f.write(json.dumps(r) + "\n")
hours = sum(r["seconds"] for r in rows) / 3600
print(f"utts={len(rows)} val={nv} speakers={len({r['speaker'] for r in rows})} hours={hours:.2f} vocab={len(set(''.join(texts)))+1}")
