"""Interactive panel: pick a checkpoint, type text, listen. Runs on CPU by default (config: panel.*).
Usage: panel.py   -> http://localhost:<panel.port>"""
import glob, io, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch, soundfile as sf
from common import load_config, P
from synth import Synth

c = load_config(); p = c["panel"]
torch.set_num_threads(p["cpu_threads"])
cache = {}  # ckpt path -> (mtime, Synth)


def get_synth(run):
    mt = os.path.getmtime(run)
    if run not in cache or cache[run][0] != mt:  # reload when the training run overwrote the checkpoint
        cache[run] = (mt, Synth(c, run=run, device=p["device"]))
    return cache[run][1]


HTML = """<!doctype html><meta charset=utf-8><title>ttslibre panel</title>
<style>body{background:#111;color:#ddd;font:15px system-ui;max-width:720px;margin:2em auto;padding:0 1em}
input,select,textarea,button{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:.5em;font:inherit;width:100%%;box-sizing:border-box;margin:.3em 0}
button{background:#2a6;color:#000;cursor:pointer}details{margin:.5em 0;color:#999}pre{color:#8c8;white-space:pre-wrap}</style>
<h3>ttslibre panel</h3>
<select id=run>%s</select>
<textarea id=text rows=3>%s</textarea>
<details><summary>advanced</summary>
ref clip <input id=ref value="%s">
steps <input id=steps value=%d> cfg <input id=cfg value=%s> duration scale <input id=dur value=%s></details>
<button onclick=go()>generate</button>
<audio id=au controls style="width:100%%;margin-top:1em"></audio><pre id=out></pre>
<script>
async function go(){const b=document.querySelector('button');b.disabled=true;out.textContent='generating...';
const r=await fetch('/synth',{method:'POST',body:JSON.stringify({run:run.value,text:text.value,ref:ref.value,steps:+steps.value,cfg:+cfg.value,dur:+dur.value})});
if(!r.ok){out.textContent=await r.text();b.disabled=false;return}
au.src=URL.createObjectURL(await r.blob());au.play();out.textContent=decodeURIComponent(r.headers.get('x-info'));b.disabled=false}
</script>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        runs = sorted(glob.glob(P(p["runs_glob"])), key=os.path.getmtime, reverse=True)
        opts = "".join(f"<option value='{r}'>{os.path.relpath(r, P('..'))} ({time.strftime('%H:%M', time.localtime(os.path.getmtime(r)))})</option>" for r in runs)
        s = c["synth"]
        body = (HTML % (opts, c["ttl"]["sample_text"], P(s["ref_clip"]), s["steps"], s["cfg"], s["duration_scale"])).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            S = get_synth(q["run"])
            zref, rmask = S.style_from_wav(q["ref"])
            t0 = time.time(); wav, dur = S(q["text"], zref, rmask, steps=q["steps"], cfg=q["cfg"], duration_scale=q["dur"]); el = time.time() - t0
            buf = io.BytesIO(); sf.write(buf, wav, c["data"]["sample_rate"], format="WAV")
            info = json.dumps({"seconds": round(dur, 2), "gen_s": round(el, 2), "rtf": round(el / max(dur, 1e-6), 2), "device": S.dev})
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("X-Info", info); self.end_headers(); self.wfile.write(buf.getvalue())
        except Exception as ex:
            self.send_response(500); self.end_headers(); self.wfile.write(str(ex).encode())

    def log_message(self, *a): pass


print(f"http://localhost:{p['port']}", flush=True)
ThreadingHTTPServer(("0.0.0.0", p["port"]), H).serve_forever()
