"""Whisper WER over the whole val set for a checkpoint: eval.val_draws draws per sentence, speaker 1988 reference.
Usage: eval_val.py <name> [<path.pt>]   name in eval.val_checkpoints, or any name with an explicit path. Writes runs/evalval/<name>.jsonl and prints mean WER."""
import json, os, re, subprocess, sys, unicodedata, urllib.request
import soundfile as sf
from num2words import num2words
from common import load_config, P
from synth import Synth

c = load_config(); e, d = c["eval"], c["data"]
tokn = os.environ.get(e["whisper_token_env"]) or subprocess.run(["bash", "-c", "grep -o 'TTS_TOKEN:-[0-9a-f]*' ~/projects/know-how/local-tts/tts | cut -d- -f2"], capture_output=True, text=True).stdout.strip()


def strip(x): return "".join(ch for ch in unicodedata.normalize("NFD", x.replace("ñ", "n~")) if unicodedata.category(ch) != "Mn").replace("n~", "ñ")
def norm(x): return re.sub(r"[^a-zñ' ]", " ", strip(re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=c["ttl"].get("lang", "en")), x.lower()))).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); dd = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dd[0] = dd[0], i
        for j in range(1, len(h) + 1):
            cur = min(dd[j] + 1, dd[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, dd[j] = dd[j], cur
    return dd[len(h)] / max(len(r), 1)


def whisper(p):
    req = urllib.request.Request(e["whisper_url"], data=open(p, "rb").read(), headers={"Authorization": f"Bearer {tokn}"})
    return urllib.request.urlopen(req, timeout=120).read().decode()



if __name__ == "__main__":
    name = sys.argv[1]; path = sys.argv[2] if len(sys.argv) > 2 else e["val_checkpoints"][name]
    S = Synth(c, run=path); zref, rmask = S.style_from_wav(P(c["synth"]["ref_clip"]))
    val = [json.loads(l) for l in open(os.path.join(P(d["prep_dir"]), "val.jsonl"))]
    if e["val_max"]: val = val[: e["val_max"]]
    out = P(e["val_out"]); os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    ws = []
    with open(os.path.join(out, f"{name}.jsonl"), "w") as f:
        for i, r in enumerate(val):
            for k in range(e["val_draws"]):
                wav, _ = S(r["text"], zref, rmask, steps=c["ttl"]["sample_steps"], cfg=c["ttl"]["sample_cfg"])
                p = os.path.join(out, "wav", f"{name}_{r['id']}_{k}.wav"); sf.write(p, wav, d["sample_rate"])
                hyp = whisper(p); w = wer(r["text"], hyp); ws.append(w)
                f.write(json.dumps({"id": r["id"], "draw": k, "words": len(norm(r["text"])), "wer": round(w, 4), "text": r["text"], "whisper": hyp}) + "\n"); f.flush()
            if i % 10 == 0: print(f"{name} {i}/{len(val)} mean wer {sum(ws)/len(ws):.3f}", flush=True)
    ws.sort(); n = len(ws)
    summ = {"name": name, "checkpoint": path, "sentences": len(val), "draws": e["val_draws"], "mean_wer": sum(ws) / n, "median_wer": ws[n // 2], "frac_over_0.5": sum(w > 0.5 for w in ws) / n, "frac_zero": sum(w == 0 for w in ws) / n}
    json.dump(summ, open(os.path.join(out, f"{name}.summary.json"), "w"), indent=1); print(json.dumps(summ))
