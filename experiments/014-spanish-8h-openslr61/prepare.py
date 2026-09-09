"""OpenSLR 61 (Argentinian Spanish, CC BY-SA 4.0) -> mels + manifest + vocab + provenance. Audio is never committed.
Vocab = English base vocab (data.base_vocab) + new characters found in the transcripts, appended so checkpoint ids stay valid."""
import json, os, random, sys
from multiprocessing import Pool
import torch, torchaudio, soundfile as sf
from common import load_config, Mel, Tokenizer, P

c = load_config(); d = c["data"]
raw, out = P(d["raw_root"]), P(d["prep_dir"])
os.makedirs(os.path.join(out, "mels"), exist_ok=True)
mel = Mel(d)


def one(item):
    uid, text = item
    torch.set_num_threads(1)
    w = os.path.join(raw, uid + ".wav")
    if not os.path.exists(w): return None
    x, sr = sf.read(w, dtype="float32"); a = torch.from_numpy(x).T if x.ndim == 2 else torch.from_numpy(x)[None]
    if sr != d["sample_rate"]: a = torchaudio.functional.resample(a, sr, d["sample_rate"])
    sec = a.shape[1] / d["sample_rate"]
    if not d["min_seconds"] <= sec <= d["max_seconds"]: return None
    m = mel(a.mean(0, keepdim=True))[0].half()
    torch.save(m, os.path.join(out, "mels", uid + ".pt"))
    return {"id": uid, "speaker": uid.split("_")[1], "text": text, "frames": m.shape[1], "seconds": round(sec, 3),
            "source": d["source"], "license": d["license"], "path": uid + ".wav"}


if __name__ == "__main__":
    items = [l.rstrip("\n").split("\t") for f in d["index_files"] for l in open(os.path.join(raw, f))]
    items = [(u, t.strip()) for u, t in items if t.strip()]
    if not items: sys.exit(f"no index rows under {raw}")
    rows = []
    with Pool(d["prep_workers"]) as pool:
        for i, r in enumerate(pool.imap_unordered(one, items, chunksize=16)):
            if r: rows.append(r)
            if i % 500 == 0: print(f"{i}/{len(items)}", flush=True)
    random.Random(d["seed"]).shuffle(rows)
    nv = max(1, int(len(rows) * d["val_fraction"]))
    with open(os.path.join(out, "val.jsonl"), "w") as f:
        for r in rows[:nv]: f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "train.jsonl"), "w") as f:
        for r in rows[nv:]: f.write(json.dumps(r) + "\n")
    base = json.load(open(P(d["base_vocab"])))
    Tokenizer(base + sorted({ch for r in rows for ch in r["text"]} - set(base))).save(os.path.join(out, "vocab.json"))
    prov = P(d["manifest_dir"]); os.makedirs(prov, exist_ok=True)
    with open(os.path.join(prov, d["manifest"]), "w") as f:
        for r in sorted(rows, key=lambda r: r["id"]): f.write(json.dumps(r) + "\n")
    hours = sum(r["seconds"] for r in rows) / 3600
    print(f"utts={len(rows)} val={nv} speakers={len({r['speaker'] for r in rows})} hours={hours:.2f} vocab={len(base)}+{len(set(''.join(r['text'] for r in rows)) - set(base))}")
