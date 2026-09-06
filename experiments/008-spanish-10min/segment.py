"""Split one long recording into clips at silences and transcribe each with Whisper.
Writes <raw>/<speaker>/1/<speaker>_NNNN.wav + .normalized.txt (LibriTTS-R layout, so prepare.py works as is).
Usage: segment.py"""
import json, os, subprocess, urllib.request
import soundfile as sf, numpy as np
from common import load_config, P

c = load_config(); s, d, e = c["segment"], c["data"], c["eval"]
raw = P(d["raw_dir"]); out = os.path.join(raw, s["speaker"], "1"); os.makedirs(out, exist_ok=True)
x, sr = sf.read(os.path.join(raw, s["source"]), dtype="float32")
tokn = os.environ.get(e["whisper_token_env"]) or subprocess.run(["bash", "-c", "grep -o 'TTS_TOKEN:-[0-9a-f]*' ~/projects/know-how/local-tts/tts | cut -d- -f2"], capture_output=True, text=True).stdout.strip()
# silences from ffmpeg silencedetect
r = subprocess.run(["ffmpeg", "-i", os.path.join(raw, s["source"]), "-af", f"silencedetect=noise={s['noise_db']}dB:d={s['min_silence']}", "-f", "null", "-"], capture_output=True, text=True).stderr
ss = [float(l.split("silence_start: ")[1].split()[0]) for l in r.splitlines() if "silence_start" in l]
se = [float(l.split("silence_end: ")[1].split()[0]) for l in r.splitlines() if "silence_end" in l]
cuts = sorted((a + b) / 2 for a, b in zip(ss, se))  # cut in the middle of each silence
bounds, start = [], 0.0
for cut in cuts + [len(x) / sr]:
    if cut - start >= s["min_seconds"]:
        bounds.append((start, cut)); start = cut
n = 0
for a, b in bounds:
    if b - a > s["max_seconds"] or n >= s["max_clips"]: continue
    clip = x[int(a * sr):int(b * sr)]
    path = os.path.join(out, f"{s['speaker']}_{n:04d}.wav"); sf.write(path, clip, sr)
    req = urllib.request.Request(e["whisper_url"], data=open(path, "rb").read(), headers={"Authorization": f"Bearer {tokn}"})
    text = urllib.request.urlopen(req, timeout=120).read().decode().strip()
    open(path[:-4] + ".normalized.txt", "w").write(text)
    print(f"{n:04d} {b-a:5.2f}s  {text}", flush=True); n += 1
print(f"clips={n} seconds={sum(b-a for a,b in bounds[:n]):.0f}")
