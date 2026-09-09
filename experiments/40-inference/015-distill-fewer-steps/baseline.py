"""Stage 0: teacher WER grid over (steps, cfg). Writes runs/baseline/grid.jsonl (one line per draw) and grid.md (table).
Uses the 014 code (common.py, synth.py symlinked) and its Whisper WER definition (eval_val.py).
Usage: baseline.py [--section eval] [--run path.pt]   (section: config key with sentences/draws/grid/out/max_minutes; --run overrides the checkpoint;
Whisper failures are logged and the row gets wer null.)"""
import json, os, random, sys, time
import soundfile as sf
sys.path.insert(0, "../../20-scaling/014-spanish-8h-openslr61")
from eval_val import wer, whisper, norm
from common import load_config, P
from synth import Synth

cfg = json.load(open("config.json")); a = sys.argv[1:]
sec = a[a.index("--section") + 1] if "--section" in a else "baseline"; b = dict(cfg["baseline"]); b.update(cfg[sec])
ckpt = a[a.index("--run") + 1] if "--run" in a else (b.get("run") or cfg["teacher"])
c = json.load(open(cfg["base_config"])); c["synth"]["ref_clip"] = cfg["ref_clip"]
S = Synth(c, run=os.path.abspath(ckpt))  # common.py is symlinked here, so P() resolves 014's relative paths from this dir (runs/ae, datasets: same layout)
zref, rmask = S.style_from_wav(P(cfg["ref_clip"]))
val = [json.loads(l) for l in open(os.path.join(P(c["data"]["prep_dir"]), "val.jsonl"))]
random.Random(b["seed"]).shuffle(val); val = val[: b["sentences"]]
out = b["out"]; os.makedirs(os.path.join(out, "wav"), exist_ok=True)
t0 = time.time(); rows = []
with open(os.path.join(out, "grid.jsonl"), "w") as f:
    for steps, g in b["grid"]:
        ws, gen = [], 0.0
        for r in val:
            for k in range(b["draws"]):
                t1 = time.time(); wav, _ = S(r["text"], zref, rmask, steps=steps, cfg=g); gen += time.time() - t1
                p = os.path.join(out, "wav", f"s{steps}_c{g}_{r['id']}_{k}.wav"); sf.write(p, wav, c["data"]["sample_rate"])
                try: hyp = whisper(p); w = wer(r["text"], hyp); ws.append(w)
                except Exception as ex: hyp, w = f"whisper failed: {ex}", None; print(hyp, flush=True)
                f.write(json.dumps({"steps": steps, "cfg": g, "id": r["id"], "draw": k, "wer": w and round(w, 4), "text": r["text"], "whisper": hyp}) + "\n"); f.flush()
                if (time.time() - t0) / 60 > b["max_minutes"]: print("time budget hit", flush=True); break
        ws.sort(); n = max(len(ws), 1); ws = ws or [float("nan")]
        rows.append((steps, g, sum(ws) / n, ws[n // 2], sum(w > 0.5 for w in ws) / n, gen / n))
        print(f"steps {steps:2d} cfg {g}: mean wer {rows[-1][2]:.3f} median {rows[-1][3]:.3f} >0.5 {rows[-1][4]:.2f} gen {rows[-1][5]:.2f}s/draw", flush=True)
with open(os.path.join(out, "grid.md"), "w") as f:
    f.write(f"checkpoint {ckpt}, {len(val)} held-out sentences x {b['draws']} draws\n\n| steps | cfg | mean WER | median | frac > 0.5 | gen s/draw (GPU) |\n|---|---|---|---|---|---|\n")
    for r in rows: f.write(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.2f} | {r[5]:.2f} |\n")
print(open(os.path.join(out, "grid.md")).read())
