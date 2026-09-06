"""LibriTTS-R subsets -> mels + manifests. Vocab and val split are fixed from an earlier prep (data.vocab_from, data.val_from)
so the checkpoint's character table and the held-out sample sentences stay identical. Clips with characters outside that vocab are dropped."""
import glob, json, os, sys
from multiprocessing import Pool
import torch, torchaudio, soundfile as sf
from common import load_config, Mel, Tokenizer, P

c = load_config(); d = c["data"]
root, out = P(d["raw_root"]), P(d["prep_dir"])
os.makedirs(os.path.join(out, "mels"), exist_ok=True)
vocab = set(json.load(open(P(d["vocab_from"]))))
val_ids = {json.loads(l)["id"] for l in open(P(d["val_from"]))}
mel = Mel(d)


def one(w):
    torch.set_num_threads(1)
    uid = os.path.basename(w)[:-4]; txt = w[:-4] + ".normalized.txt"
    if not os.path.exists(txt): return None
    text = open(txt).read().strip()
    if set(text) - vocab: return None
    x, sr = sf.read(w, dtype="float32"); a = torch.from_numpy(x).T if x.ndim == 2 else torch.from_numpy(x)[None]
    if sr != d["sample_rate"]: a = torchaudio.functional.resample(a, sr, d["sample_rate"])
    sec = a.shape[1] / d["sample_rate"]
    if not d["min_seconds"] <= sec <= d["max_seconds"]: return None
    m = mel(a.mean(0, keepdim=True))[0]
    torch.save(m.half(), os.path.join(out, "mels", uid + ".pt"))
    subset = os.path.relpath(w, root).split(os.sep)[0]
    return {"id": uid, "speaker": uid.split("_")[0], "text": text, "frames": m.shape[1], "seconds": round(sec, 3),
            "source": f"LibriTTS-R/{subset}", "license": "CC BY 4.0", "path": os.path.relpath(w, root)}


if __name__ == "__main__":
    wavs = sorted(w for s in d["subsets"] for w in glob.glob(os.path.join(root, s, "*", "*", "*.wav")))
    if not wavs: sys.exit(f"no wavs under {root}")
    rows = []
    with Pool(d["prep_workers"]) as pool:
        for i, r in enumerate(pool.imap_unordered(one, wavs, chunksize=16)):
            if r: rows.append(r)
            if i % 1000 == 0: print(f"{i}/{len(wavs)}", flush=True)
    rows.sort(key=lambda r: r["id"])
    val = [r for r in rows if r["id"] in val_ids]; train = [r for r in rows if r["id"] not in val_ids]
    with open(os.path.join(out, "val.jsonl"), "w") as f:
        for r in val: f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "train.jsonl"), "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    json.dump(json.load(open(P(d["vocab_from"]))), open(os.path.join(out, "vocab.json"), "w"))
    os.makedirs(P(d["manifest_dir"]), exist_ok=True)
    for s in d["subsets"]:
        with open(os.path.join(P(d["manifest_dir"]), f"libritts-r-{s}.jsonl"), "w") as f:
            for r in rows:
                if r["source"].endswith(s): f.write(json.dumps(r) + "\n")
    hours = sum(r["seconds"] for r in rows) / 3600
    print(f"wavs={len(wavs)} kept={len(rows)} train={len(train)} val={len(val)} speakers={len({r['speaker'] for r in rows})} hours={hours:.2f}")
