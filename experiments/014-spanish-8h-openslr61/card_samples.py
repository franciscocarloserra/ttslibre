"""Samples for the Hugging Face model card: every card.voices x (sentences + card.val_ids), card.draws draws, Whisper WER.
Keeps the lowest-WER draw per (voice, sentence) if it is <= card.max_wer. Writes card.out_dir/{all.jsonl, kept.jsonl, kept/*.wav} and voicepacks ../voices/<voice>.pt.
Usage: card_samples.py [<checkpoint.pt>]  (default card.checkpoint)"""
import json, os, shutil, sys
import torch, soundfile as sf
from common import load_config, P
from synth import Synth
from eval_val import wer, whisper

c = load_config(); k, d = c["card"], c["data"]
path = sys.argv[1] if len(sys.argv) > 1 else k["checkpoint"]
S = Synth(c, run=path)
val = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(os.path.join(P(d["prep_dir"]), "val.jsonl"))}
sents = [("novel%d" % (i + 1), t.strip()) for i, t in enumerate(open(P(k["sentences"]))) if t.strip()] + [(i, val[i]) for i in k["val_ids"]]
out = P(k["out_dir"]); os.makedirs(os.path.join(out, "kept"), exist_ok=True); os.makedirs(os.path.join(out, "wav"), exist_ok=True)
voices = os.path.join(P(".."), "voices")
rows, kept = [], []
for v, clip in k["voices"].items():
    clip = os.path.join(P(d["raw_root"]), clip); zref, rmask = S.style_from_wav(clip)
    x, sr = sf.read(clip)
    torch.save({"zref": zref.cpu().half(), "source": clip, "seconds": round(len(x) / sr, 2), "exp": os.path.basename(os.getcwd())}, os.path.join(voices, v + ".pt"))
    for sid, text in sents:
        best = None
        for n in range(k["draws"]):
            wav, _ = S(text, zref, rmask, steps=k["steps"], cfg=k["cfg"])
            p = os.path.join(out, "wav", f"{v}_{sid}_{n}.wav"); sf.write(p, wav, d["sample_rate"])
            hyp = whisper(p); w = wer(text, hyp)
            r = {"voice": v, "sentence": sid, "draw": n, "text": text, "whisper": hyp, "wer": round(w, 3), "seconds": round(len(wav) / d["sample_rate"], 2), "wav": p}
            rows.append(r); best = r if best is None or w < best["wer"] else best
        if best["wer"] <= k["max_wer"]:
            dst = os.path.join(out, "kept", f"{v}_{sid}.wav"); shutil.copy(best["wav"], dst); kept.append(best | {"kept": dst})
        print(f"{v} {sid} best wer {best['wer']:.2f} {'kept' if best['wer'] <= k['max_wer'] else 'dropped'}", flush=True)
open(os.path.join(out, "all.jsonl"), "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
open(os.path.join(out, "kept.jsonl"), "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
print(f"kept {len(kept)}/{len(k['voices']) * len(sents)}  mean best wer {sum(r['wer'] for r in kept) / max(len(kept), 1):.3f}")
