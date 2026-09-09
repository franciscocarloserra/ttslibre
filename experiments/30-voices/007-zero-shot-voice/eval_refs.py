"""Zero-shot style test: one trained model, several reference clips, same sentences. Writes wavs + WER table.
Usage: eval_refs.py"""
import json, os, re, subprocess, urllib.request
import soundfile as sf
from num2words import num2words
from common import load_config, P
from synth import Synth

c = load_config(); r, e = c["refs"], c["eval"]
S = Synth(c, run=r["model"])
out = P(r["out"]); os.makedirs(out, exist_ok=True)
tokn = os.environ.get(e["whisper_token_env"]) or subprocess.run(["bash", "-c", "grep -o 'TTS_TOKEN:-[0-9a-f]*' ~/projects/know-how/local-tts/tts | cut -d- -f2"], capture_output=True, text=True).stdout.strip()
norm = lambda x: re.sub(r"[^a-z' ]", " ", re.sub(r"\d+", lambda m: num2words(int(m.group())), x.lower())).split()


def wer(ref, hyp):
    a, b = norm(ref), norm(hyp); dd = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dd[0] = dd[0], i
        for j in range(1, len(b) + 1):
            cur = min(dd[j] + 1, dd[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev, dd[j] = dd[j], cur
    return dd[len(b)] / len(a)


def whisper(path):
    req = urllib.request.Request(e["whisper_url"], data=open(path, "rb").read(), headers={"Authorization": f"Bearer {tokn}"})
    return urllib.request.urlopen(req, timeout=120).read().decode()


table, log = {}, open(os.path.join(out, "results.log"), "w")
for spk, path in r["refs"].items():
    zref, rmask = S.style_from_wav(P(path))
    for name, text in r["sentences"].items():
        ws = []
        for k in range(r["repeats"]):
            wav, _ = S(text, zref, rmask, steps=r["steps"], cfg=r["cfg"])
            f = os.path.join(out, f"{spk}_{name}_{k}.wav"); sf.write(f, wav, c["data"]["sample_rate"])
            hyp = whisper(f); w = wer(text, hyp); ws.append(w)
            line = f"ref {spk}  {name} #{k}  wer {w:.2f}  whisper: {hyp}"; print(line, flush=True); log.write(line + "\n")
        table[(spk, name)] = sum(ws) / len(ws)
names = list(r["sentences"])
md = "| ref | " + " | ".join(names) + " | mean |\n|---|" + "---|" * (len(names) + 1) + "\n"
for spk in r["refs"]:
    row = [table[(spk, n)] for n in names]
    md += f"| {spk} | " + " | ".join(f"{w:.2f}" for w in row) + f" | {sum(row)/len(row):.2f} |\n"
print(md); open(os.path.join(out, "table.md"), "w").write(md)
