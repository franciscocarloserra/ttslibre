"""Whisper WER on fixed sentences for: ours, Kokoro-82M, Supertonic 3 (both via the local TTS server).
Also params, CPU RTF (8 threads, one sentence) and training peak VRAM for ours.
Usage: eval.py [--systems ours,kokoro,supertonic] [--run runs/ttl]"""
import json, os, re, subprocess, sys, time, urllib.request
import soundfile as sf, torch
from common import load_config, P, count_params

c = load_config(); e = c["eval"]
args = sys.argv[1:]
systems = (args[args.index("--systems") + 1] if "--systems" in args else "ours,kokoro,supertonic").split(",")
run = args[args.index("--run") + 1] if "--run" in args else "runs/ttl"
sents = [l.strip() for l in open(P(e["sentences"])) if l.strip()]
out = P(e["out_dir"]); os.makedirs(out, exist_ok=True)
tokens = {"whisper": os.environ.get(e["whisper_token_env"], ""), "tts": os.environ.get(e["tts_token_env"], "")}
if not tokens["whisper"]:  # same default token as the local servers' scripts
    tokens["whisper"] = tokens["tts"] = subprocess.run(["bash", "-c", "grep -o 'TTS_TOKEN:-[0-9a-f]*' ~/projects/know-how/local-tts/tts | cut -d- -f2"], capture_output=True, text=True).stdout.strip()


def norm(s):
    from num2words import num2words
    s = re.sub(r"\d+", lambda m: num2words(int(m.group())), s.lower())  # Whisper writes "19", texts say "nineteen"
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", s)).strip()


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def whisper(path):
    req = urllib.request.Request(e["whisper_url"], data=open(path, "rb").read(), headers={"Authorization": f"Bearer {tokens['whisper']}"})
    return urllib.request.urlopen(req, timeout=120).read().decode()


def tts_server(text, voice, path):
    req = urllib.request.Request(f"{e['tts_url']}?voice={voice}&fmt=wav", data=text.encode(), headers={"Authorization": f"Bearer {tokens['tts']}"})
    open(path, "wb").write(urllib.request.urlopen(req, timeout=300).read())


synths = {}
if "ours" in systems:
    from synth import Synth
    S = Synth(c, run=os.path.join(run, "ttl.pt"))
    zref, rmask = S.style_from_wav(P(c["synth"]["ref_clip"]))
    synths["ours"] = lambda text, path: sf.write(path, S(text, zref, rmask)[0], c["data"]["sample_rate"])
if "kokoro" in systems:
    synths["kokoro"] = lambda text, path: tts_server(text, e["kokoro_voice"], path)
if "supertonic" in systems:
    synths["supertonic"] = lambda text, path: tts_server(text, e["supertonic_voice"], path)

results = {}
for name, fn in synths.items():
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    errs, words, rows = 0, 0, []
    for i, s in enumerate(sents):
        p = os.path.join(d, f"{i:02d}.wav")
        fn(s, p)
        hyp = whisper(p)
        er, n = wer(s, hyp); errs += er; words += n
        rows.append({"i": i, "ref": s, "hyp": hyp, "err": er, "n": n})
        print(f"[{name}] {i} wer={er/n:.2f} | {hyp}", flush=True)
    results[name] = {"wer": errs / words, "errors": errs, "words": words}
    json.dump(rows, open(os.path.join(d, "transcripts.json"), "w"), indent=1)

if "ours" in systems:
    ae = json.load(open(P("runs/ae/summary.json"))); tl = json.load(open(P(os.path.join(run, "summary.json"))))
    torch.set_num_threads(c["synth"]["cpu_threads"])
    Sc = Synth(c, run=os.path.join(run, "ttl.pt"), device="cpu")
    zr, rm = Sc.style_from_wav(P(c["synth"]["ref_clip"]))
    Sc(sents[0], zr, rm)  # warm-up
    t0 = time.time(); _, dur = Sc(sents[0], zr, rm); el = time.time() - t0
    results["ours"].update({"params_ttl": tl["params"], "params_ae": ae["params"], "params_vocos": count_params(Sc.vocos),
                            "rtf_cpu": el / dur, "gen_s_cpu": el, "train_peak_vram_mib": max(ae["peak_vram_mib"], tl["peak_vram_mib"]), "steps": tl["steps"]})
    sd = P(e["samples_dir"]); os.makedirs(sd, exist_ok=True)
    for i in range(min(e["n_samples"], len(sents))):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(out, "ours", f"{i:02d}.wav"), "-c:a", "libvorbis", "-q:a", str(e["ogg_quality"]), os.path.join(sd, f"{i:02d}.ogg")])
json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
print(json.dumps(results, indent=1))
