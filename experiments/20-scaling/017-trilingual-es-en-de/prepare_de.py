"""German source (LJSpeech-style layout: metadata.csv `id|text|...` + wavs/) -> mels + latents (existing AE) + manifest + vocab.
Vocab = data.base_vocab (Spanish prep vocab) + new characters found in the transcripts, appended so ids stay valid. Audio is never committed.
Layout knobs in config data.sources.de: metadata, wav_dir, delimiter, id_col, text_col, speaker."""
import json, os, random, sys
from multiprocessing import Pool
import torch, torchaudio, soundfile as sf
from common import load_config, LatentAE, Mel, Tokenizer, P

c = load_config(); d = c["data"]; src = d["sources"]["de"]
raw, out = P(src["raw_root"]), P(src["prep"])
for sub in ("mels", "latents"): os.makedirs(os.path.join(out, sub), exist_ok=True)
mel = Mel(d)


def one(item):
    uid, text = item
    torch.set_num_threads(1)
    w = os.path.join(raw, src["wav_dir"], uid + ".wav")
    if not os.path.exists(w): return None
    x, sr = sf.read(w, dtype="float32"); a = torch.from_numpy(x).T if x.ndim == 2 else torch.from_numpy(x)[None]
    if sr != d["sample_rate"]: a = torchaudio.functional.resample(a, sr, d["sample_rate"])
    sec = a.shape[1] / d["sample_rate"]
    if not d["min_seconds"] <= sec <= d["max_seconds"]: return None
    m = mel(a.mean(0, keepdim=True))[0].half()
    torch.save(m, os.path.join(out, "mels", uid + ".pt"))
    return {"id": uid, "speaker": src["speaker"], "text": text, "frames": m.shape[1], "seconds": round(sec, 3),
            "source": src["source"], "license": src["license"], "path": os.path.join(src["wav_dir"], uid + ".wav")}


if __name__ == "__main__":
    items = [l.rstrip("\n").split(src["delimiter"]) for l in open(os.path.join(raw, src["metadata"]), encoding="utf-8")]
    items = [(x[src["id_col"]], x[src["text_col"]].strip()) for x in items if len(x) > max(src["id_col"], src["text_col"]) and x[src["text_col"]].strip()]
    if not items: sys.exit(f"no metadata rows under {raw}")
    rows = []
    with Pool(d["prep_workers"]) as pool:
        for i, r in enumerate(pool.imap_unordered(one, items, chunksize=16)):
            if r: rows.append(r)
            if i % 500 == 0: print(f"{i}/{len(items)}", flush=True)
    random.Random(d["seed"]).shuffle(rows)
    nv = max(1, int(len(rows) * d["val_fraction"]))
    with open(os.path.join(out, "val.jsonl"), "w") as f:
        for r in rows[:nv]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "train.jsonl"), "w") as f:
        for r in rows[nv:]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    base = json.load(open(P(d["base_vocab"])))
    Tokenizer(base + sorted({ch for r in rows for ch in r["text"]} - set(base))).save(os.path.join(out, "vocab.json"))
    prov = P(d["manifest_dir"]); os.makedirs(prov, exist_ok=True)
    with open(os.path.join(prov, src["manifest"]), "w") as f:
        for r in sorted(rows, key=lambda r: r["id"]): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # latents with the shared AE (same stats as every other prep)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ae = LatentAE(c).to(dev).eval(); ae.load_state_dict(torch.load(P("runs/ae/ae.pt"), map_location=dev))
    with torch.no_grad():
        for i, r in enumerate(rows):
            m = torch.load(os.path.join(out, "mels", r["id"] + ".pt")).float()[None].to(dev)
            torch.save(ae.encode(m)[0].half().cpu(), os.path.join(out, "latents", r["id"] + ".pt"))
    import shutil; shutil.copy(P(d["latent_stats"]), os.path.join(out, "latent_stats.pt"))
    print(f"utts={len(rows)} val={nv} hours={sum(r['seconds'] for r in rows)/3600:.2f} vocab={len(base)}+{len(set(''.join(r['text'] for r in rows)) - set(base))}")
