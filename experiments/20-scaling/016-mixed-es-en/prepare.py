"""Mixed Spanish+English prep from two existing preps (same AE, same latent stats; the Spanish vocab is the English vocab plus 13 chars, ids unchanged).
Per language: fewest speakers (most hours first) whose hours reach data.hours_per_lang. Latents/mels are symlinked, nothing is re-encoded.
train.jsonl and val.jsonl interleave the two languages (es, en, es, en, ...) so any prefix is balanced; rows carry "lang" and an absolute "path"; text is wrapped in language tags <es>...</es> / <en>...</en> (Supertonic style) and the tags are appended to the vocab."""
import json, os, random, shutil, collections
from common import load_config, P

c = load_config(); d = c["data"]; out = P(d["prep_dir"])
for sub in ("latents", "mels"): os.makedirs(os.path.join(out, sub), exist_ok=True)
picked = {}
for lang, src in d["sources"].items():
    prep = P(src["prep"]); rows = [json.loads(l) for f in ("train.jsonl", "val.jsonl") for l in open(os.path.join(prep, f))]
    hours = collections.Counter()
    for r in rows: hours[r["speaker"]] += r["seconds"] / 3600
    spk, acc = [], 0.0
    for s, h in hours.most_common():
        if acc >= d["hours_per_lang"]: break
        spk.append(s); acc += h
    sel = [r for r in rows if r["speaker"] in spk]
    for r in sel:
        r["lang"] = lang; r["text"] = f"<{lang}>{r['text']}</{lang}>"; r["path"] = os.path.abspath(os.path.join(P(src["raw_root"]), r["path"]))
        for sub in ("latents", "mels"):
            dst = os.path.join(out, sub, r["id"] + ".pt")
            if not os.path.lexists(dst): os.symlink(os.path.abspath(os.path.join(prep, sub, r["id"] + ".pt")), dst)
    random.Random(d["seed"]).shuffle(sel); picked[lang] = sel
    print(f"{lang}: {len(spk)} speakers {acc:.2f} h {len(sel)} clips ({', '.join(spk)})", flush=True)
h_min = min(sum(r["seconds"] for r in v) for v in picked.values())  # equal hours per language: trim the larger one
for l, v in picked.items():
    acc, k = 0.0, 0
    while k < len(v) and acc < h_min: acc += v[k]["seconds"]; k += 1
    picked[l] = v[:k]
n = min(len(v) for v in picked.values())
mixed = [r for pair in zip(*(picked[l][:n] for l in picked)) for r in pair] + [r for l in picked for r in picked[l][n:]]  # es, en, es, en, ... then the remainder
nv = max(2, int(len(mixed) * d["val_fraction"])) // 2 * 2
with open(os.path.join(out, "val.jsonl"), "w") as f:
    for r in mixed[:nv]: f.write(json.dumps(r) + "\n")
with open(os.path.join(out, "train.jsonl"), "w") as f:
    for r in mixed[nv:]: f.write(json.dumps(r) + "\n")
json.dump(json.load(open(P(d["vocab_from"]))) + [f"<{l}>" for l in d["sources"]] + [f"</{l}>" for l in d["sources"]], open(os.path.join(out, "vocab.json"), "w")); shutil.copy(P(d["latent_stats"]), os.path.join(out, "latent_stats.pt"))
print(f"train={len(mixed)-nv} val={nv} hours={sum(r['seconds'] for r in mixed)/3600:.2f} per lang " + ", ".join(f"{l} {sum(r['seconds'] for r in v)/3600:.2f} h {len(v)} clips" for l, v in picked.items()))
