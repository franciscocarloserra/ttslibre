"""Voicepacks: a voice is the compressed AE latent of a reference clip (checkpoint-independent, the AE is shared by all experiments).
Usage (from experiments/):  voice.py make <clip.wav> <name> [--exp EXP]          -> voices/<name>.pt (latents, checkpoint-independent)
                            voice.py style <name> --run <ckpt.pt>               -> voices/<name>.style.pt (style tensors for that checkpoint, editable)
                            voice.py mix <a> <b> <t> <out> --run <ckpt.pt>      -> voices/<out>.style.pt = (1-t)*a + t*b (a, b: voice names, .pt or .style.pt)
                            voice.py test <name> --run <ckpt.pt> [--exp EXP]    -> voices/<name>.test.json (Whisper WER on the 007 sentences); name may be x.style
Requires an experiment >= 013 for style/mix (TTL.style)."""
import json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); VOICES = os.path.join(HERE, "voices")
argv = sys.argv[1:]
EXP = argv[argv.index("--exp") + 1] if "--exp" in argv else sorted(os.path.relpath(d, HERE) for d in glob.glob(os.path.join(HERE, "*", "[0-9][0-9][0-9]-*")) if os.path.isdir(d))[-1]  # group/NNN-slug
os.chdir(os.path.join(HERE, EXP)); sys.path.insert(0, os.getcwd())
import torch, soundfile as sf
from common import load_config, P
from synth import Synth

c = load_config(); os.makedirs(VOICES, exist_ok=True)
run = argv[argv.index("--run") + 1] if "--run" in argv else None
S = Synth(c, run=run)
if argv[0] == "make":
    clip, name = os.path.abspath(argv[1]), argv[2]
    zref, rmask = S.style_from_wav(clip)
    x, sr = sf.read(clip)
    torch.save({"zref": zref.cpu().half(), "source": clip, "seconds": round(len(x) / sr, 2), "exp": EXP}, os.path.join(VOICES, name + ".pt"))
    print(f"voices/{name}.pt  frames={zref.shape[2]}  from {clip}")
elif argv[0] in ("style", "mix"):
    def tensors(n):
        p = os.path.join(VOICES, n + ".pt")
        if n.endswith(".style"): return S.style_from_pack(p)
        return S.style_tensors(*S.style_from_voice(p))
    if argv[0] == "style":
        name, st = argv[1], tensors(argv[1])
    else:
        a, b, t, name = argv[1], argv[2], float(argv[3]), argv[4]; A, B = tensors(a), tensors(b)
        st = {k: (1 - t) * A[k] + t * B[k] for k in A}
    torch.save({k: v.cpu() for k, v in st.items()} | {"checkpoint": os.path.abspath(run), "made_from": argv[1:-1] if argv[0] == "mix" else [name]}, os.path.join(VOICES, name.replace(".style", "") + ".style.pt"))
    print(f"voices/{name.replace('.style', '')}.style.pt  style {tuple(st['style'].shape)}  dp {tuple(st['dp'].shape)}  checkpoint {run}")
elif argv[0] == "test":
    name = argv[1]; v = json.load(open(os.path.join(HERE, "30-voices", "007-zero-shot-voice", "config.json")))["refs"]
    zref, rmask = S.style(os.path.join(VOICES, name + ".pt"))
    import re as _re, subprocess, urllib.request
    from num2words import num2words
    e = c["eval"]; tokn = os.environ.get(e["whisper_token_env"]) or subprocess.run(["bash", "-c", "grep -o 'TTS_TOKEN:-[0-9a-f]*' ~/projects/know-how/local-tts/tts | cut -d- -f2"], capture_output=True, text=True).stdout.strip()
    norm = lambda x: _re.sub(r"[^a-z' ]", " ", _re.sub(r"\d+", lambda m: num2words(int(m.group())), x.lower())).split()
    def wer(ref, hyp):
        r, h = norm(ref), norm(hyp); dd = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, dd[0] = dd[0], i
            for j in range(1, len(h) + 1):
                cur = min(dd[j] + 1, dd[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, dd[j] = dd[j], cur
        return dd[len(h)] / max(len(r), 1)
    def whisper(data):
        return urllib.request.urlopen(urllib.request.Request(e["whisper_url"], data=data, headers={"Authorization": f"Bearer {tokn}"}), timeout=120).read().decode()
    out = {"voice": name, "checkpoint": os.path.abspath(run), "sentences": {}}; gen = []
    for sname, text in v["sentences"].items():
        ws = []
        for k in range(v["repeats"]):
            t0 = time.time(); wav, _ = S(text, zref, rmask, steps=v["steps"], cfg=v["cfg"]); gen.append(time.time() - t0)
            path = os.path.join(VOICES, f"{name}_{sname}_{k}.wav"); sf.write(path, wav, c["data"]["sample_rate"])
            heard = whisper(open(path, "rb").read()); w = wer(text, heard); ws.append(w)
            print(f"{sname} {k} wer {w:.2f}  whisper heard: {heard}", flush=True)
        out["sentences"][sname] = {"text": text, "wer": [round(w, 2) for w in ws], "mean": round(sum(ws) / len(ws), 3)}
    out["gen_seconds_mean"] = round(sum(gen) / len(gen), 3); out["mean_wer"] = round(sum(s["mean"] for s in out["sentences"].values()) / len(out["sentences"]), 3)
    json.dump(out, open(os.path.join(VOICES, name + ".test.json"), "w"), indent=1); print(f"mean wer {out['mean_wer']}")
