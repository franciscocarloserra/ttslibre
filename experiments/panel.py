"""Experiments panel: WER chart + sample rounds of any run, and interactive synthesis from any checkpoint.
Lives in experiments/; imports code and config from one experiment dir (--exp NAME, default: newest).
Usage: panel.py [--exp 004-word-generalization]   -> http://localhost:<panel.port>"""
import glob, io, json, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
EXP = argv[argv.index("--exp") + 1] if "--exp" in argv else sorted(d for d in os.listdir(HERE) if re.match(r"\d{3}-", d))[-1]
os.chdir(os.path.join(HERE, EXP)); sys.path.insert(0, os.getcwd())
import torch, soundfile as sf
from common import load_config, P
from synth import Synth

c = load_config(); p = c["panel"]
LOG_RE = [re.compile(r"SAMPLE (\w+) step=(\d+) wer=([\d.]+) \| (.*)"), re.compile(r"(\S+) \((\d+)\)  (\w+) wer ([\d.]+)  whisper heard: (.*)"),
          re.compile(r"SAMPLE step=(\d+) wer=([\d.]+) \| (.*)")]
LABEL = {"train": "training sentence (should be memorized)", "knownwords": "known words, new order", "heldout1": "never-seen words", "heldout2": "never-seen words"}


def sentences(run_dir):
    """name -> {text, original wav path or None}, reconstructed from the run's effective config and the prep split."""
    cfg = os.path.join(run_dir, "config.effective.json")
    cc = json.load(open(cfg)) if os.path.exists(cfg) else c
    t, d = cc["ttl"], cc["data"]; prep = P(d["prep_dir"])
    train = [json.loads(l) for l in open(os.path.join(prep, "train.jsonl"))]; val = [json.loads(l) for l in open(os.path.join(prep, "val.jsonl"))]
    if d.get("speaker"): train = [r for r in train if r["speaker"] == d["speaker"]]; val = [r for r in val if r["speaker"] == d["speaker"]] or train[:4]
    names = [("train", t["sample_text"])] + [tuple(x) for x in t.get("sample_extra", [])] + [(f"heldout{i+1}", r["text"]) for i, r in enumerate(val[: t.get("sample_heldout_n", 0)])]
    out = {}
    for n, text in names:
        orig = [r for r in train + val if r["text"] == text]
        out[n] = {"text": text, "label": LABEL.get(n, n), "original": os.path.join(P(d["raw_dir"]), orig[0]["path"]) if orig else None}
    return out


def rounds(run_dir):
    """Parse progress.log -> [{step, name, wer, heard}] (all log formats used so far)."""
    out = []
    for line in open(os.path.join(run_dir, "progress.log")):
        for i, rx in enumerate(LOG_RE):
            m = rx.match(line)
            if m:
                g = m.groups()
                out.append({"name": "train", "step": int(g[0]), "wer": float(g[1]), "heard": g[2], "elapsed": ""} if i == 2 else
                           {"name": g[2], "step": int(g[1]), "wer": float(g[3]), "heard": g[4], "elapsed": g[0]} if i == 1 else
                           {"name": g[0], "step": int(g[1]), "wer": float(g[2]), "heard": g[3], "elapsed": ""})
                break
    return {"rounds": out, "sentences": sentences(run_dir)}
torch.set_num_threads(p["cpu_threads"])
cache = {}  # ckpt path -> (mtime, Synth)


def get_synth(run):
    mt = os.path.getmtime(run)
    if run not in cache or cache[run][0] != mt:  # reload when the training run overwrote the checkpoint
        cache[run] = (mt, Synth(c, run=run, device=p["device"]))
    return cache[run][1]


HTML = """<!doctype html><meta charset=utf-8><title>ttslibre panel</title>
<style>body{background:#111;color:#ddd;font:15px system-ui;margin:0;padding:1em}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:2em}
input,select,textarea,button{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:.5em;font:inherit;width:100%%;box-sizing:border-box;margin:.3em 0}
button{background:#2a6;color:#000;cursor:pointer}button.play{width:5.5em;padding:.2em .4em;margin:0;background:#333;color:#ddd}details{margin:.5em 0;color:#999}pre{color:#8c8;white-space:pre-wrap}
table{width:100%%;font-size:13px;border-collapse:collapse}td{padding:2px 4px;vertical-align:middle}svg{background:#181818;border-radius:4px;display:block;width:100%%}</style>
<div class=cols><div>
<h3>run</h3><select id=view onchange="load()">%s</select>
<svg id=chart viewBox="0 0 700 220"></svg>
<h3>generate</h3>
voice <select id=refsel onchange="ref.value=this.value"></select>
<input id=text value="%s" onkeydown="if(event.key=='Enter')go()">
<button onclick="go()">generate</button>
<div id=gen style="margin-top:.5em"></div><pre id=out></pre>
<details><summary>advanced</summary>
ref clip <input id=ref value="%s" size=60>
steps <input id=steps value="%d"> cfg <input id=cfg value="%s"> duration scale <input id=dur value="%s"></details>
</div><div id=rounds></div></div>
<script>
let cur=null;const AU=new Audio();AU.onended=()=>{if(cur){cur.textContent='▶ '+cur.dataset.d;cur=null}};
function pl(b){if(cur===b){AU.pause();AU.currentTime=0;b.textContent='▶ '+b.dataset.d;cur=null;return}if(cur){cur.textContent='▶ '+cur.dataset.d}cur=b;AU.src=b.dataset.src;AU.play();b.textContent='■ '+b.dataset.d}
function btn(src){return `<button class=play data-src="${src}" data-d="…" onclick="pl(this)">▶ …</button>`}
const DUR={};function durs(){for(const b of document.querySelectorAll('button.play[data-d="…"]')){const k=b.dataset.src;if(DUR[k]){b.dataset.d=DUR[k];b.textContent='▶ '+DUR[k];continue}if(DUR[k]===null)continue;DUR[k]=null;const a=new Audio();a.preload='metadata';a.onloadedmetadata=()=>{DUR[k]=a.duration.toFixed(1)+'s';durs()};a.src=k}}
const COL={train:'#6c6',knownwords:'#fc6',heldout1:'#f66',heldout2:'#c6f'};
const el=(t,a)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e};
async function load(){const j=await (await fetch('/rounds?run='+encodeURIComponent(view.value))).json();const d=j.rounds,S=j.sentences;
const W=700,H=220,L=40,B=24,mx=Math.max(1,...d.map(r=>r.step)),my=Math.max(1,...d.map(r=>r.wer));
const X=s=>L+(W-L-10)*s/mx,Y=w=>H-B-(H-B-10)*w/my;const c=document.getElementById('chart');c.textContent='';
for(const v of [0,0.5,1,1.5,2]) if(v<=my){c.append(el('line',{x1:L,x2:W,y1:Y(v),y2:Y(v),stroke:'#333'}));const t=el('text',{x:2,y:Y(v)+4,fill:'#888','font-size':11});t.textContent=v;c.append(t)}
const names=[...new Set(d.map(r=>r.name))];
names.forEach((n,i)=>{c.append(el('polyline',{points:d.filter(r=>r.name==n).map(r=>X(r.step)+','+Y(r.wer)).join(' '),fill:'none',stroke:COL[n]||'#aaa','stroke-width':2}));
const t=el('text',{x:L+10+i*130,y:14,fill:COL[n]||'#aaa','font-size':12});t.textContent=n+' (wer)';c.append(t)});
const t=el('text',{x:W-70,y:H-6,fill:'#888','font-size':11});t.textContent='step '+mx;c.append(t);
const steps=[...new Set(d.map(r=>r.step))].sort((a,b)=>b-a);const N=steps.length;let h='<h3>rounds, newest first</h3>';
const legend=Object.entries(S).map(([n,x])=>`<div style="color:${COL[n]||'#aaa'}"><b>${n}</b> = ${x.label}: <i>${x.text}</i>${x.original?` ${btn(`/wav?run=x&f=x&orig=${encodeURIComponent(x.original)}`)} (original recording)`:''}</div>`).join('');
h+='<details open><summary>the sentences</summary>'+legend+'</details>';
steps.forEach((s,i)=>{const rs=d.filter(r=>r.step==s);h+=`<div style="border:1px solid #333;border-radius:6px;padding:.5em;margin:.6em 0"><div style="color:#aaa;margin-bottom:.3em">round ${N-i} of ${N} &middot; step ${s}${rs[0].elapsed?' &middot; '+rs[0].elapsed+' into the run':''}</div><table>`;
for(const r of rs) h+=`<tr><td style="color:${COL[r.name]||'#aaa'};white-space:nowrap;width:9em"><b>${r.name}</b><br>wer ${r.wer.toFixed(2)}</td><td style="width:6em">${btn(`/wav?run=${encodeURIComponent(view.value)}&f=${r.name}_step_${String(s).padStart(6,'0')}.wav`)}</td><td style="color:#999">input: <span style="color:#ddd">${(S[r.name]||{}).text||''}</span><br>whisper: <span style="color:#ddd">${r.heard}</span></td></tr>`;h+='</table></div>'});
const el=document.getElementById('rounds');if(el.dataset.h!==h){el.innerHTML=h;el.dataset.h=h;durs()}}
load();setInterval(load,15000);
(async()=>{const R=await (await fetch('/refs')).json();refsel.innerHTML=Object.entries(R).map(([k,v])=>`<option value="${v}">${k}</option>`).join('');ref.value=refsel.value})();
async function go(){const b=document.querySelector('button');b.disabled=true;out.textContent='generating...';
const r=await fetch('/synth',{method:'POST',body:JSON.stringify({run:view.value+'/ttl.pt',text:text.value,ref:ref.value,steps:+steps.value,cfg:+cfg.value,dur:+dur.value})});
if(!r.ok){out.textContent=await r.text();b.disabled=false;return}
const blob=await r.blob();const u=URL.createObjectURL(blob);gen.innerHTML=btn(u);durs();pl(gen.firstChild);out.textContent=decodeURIComponent(r.headers.get('x-info'))+'\\nwer: checking...';b.disabled=false;
const w=await (await fetch('/wer',{method:'POST',headers:{'X-Text':encodeURIComponent(text.value)},body:blob})).json();out.textContent=out.textContent.replace('wer: checking...',w.error?'wer: failed '+w.error:`wer ${w.wer.toFixed(2)}\\nwhisper: ${w.heard}`)}
</script>"""


import urllib.request, urllib.parse, subprocess
from num2words import num2words
_e = c["eval"]
_tok = os.environ.get(_e["whisper_token_env"]) or subprocess.run(["bash", "-c", "grep -o 'TTS_TOKEN:-[0-9a-f]*' ~/projects/know-how/local-tts/tts | cut -d- -f2"], capture_output=True, text=True).stdout.strip()
_norm = lambda x: re.sub(r"[^a-z' ]", " ", re.sub(r"\d+", lambda m: num2words(int(m.group())), x.lower())).split()


def wer(ref, hyp):
    a, b = _norm(ref), _norm(hyp); dd = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dd[0] = dd[0], i
        for j in range(1, len(b) + 1):
            cur = min(dd[j] + 1, dd[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev, dd[j] = dd[j], cur
    return dd[len(b)] / max(len(a), 1)


def whisper(data):
    req = urllib.request.Request(_e["whisper_url"], data=data, headers={"Authorization": f"Bearer {_tok}"})
    return urllib.request.urlopen(req, timeout=120).read().decode()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/refs":  # voices: reference clips listed in 007's config
            r = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "007-style-refs", "config.json")))["refs"]["refs"]
            body = json.dumps({k: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "007-style-refs", v)) for k, v in r.items()}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/rounds?"):
            body = json.dumps(rounds(self.path.split("run=")[1].replace("%2F", "/"))).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/wav?"):
            q = dict(kv.split("=") for kv in self.path[5:].split("&")); f = os.path.join(q["run"].replace("%2F", "/"), "samples", q["f"])
            if "orig" in q: f = q["orig"].replace("%2F", "/")
            if not os.path.exists(f): f = os.path.join(q["run"].replace("%2F", "/"), "samples", q["f"].split("_", 1)[1])  # 002 layout: step_XXXXXX.wav
            if not os.path.exists(f): self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.end_headers(); self.wfile.write(open(f, "rb").read()); return
        runs = sorted([r for r in glob.glob(P(p["runs_glob"])) if "todelete" not in r], key=os.path.getmtime, reverse=True)
        views = "".join(f"<option value='{os.path.dirname(r)}'>{os.path.relpath(r, P('..')).rsplit('/', 1)[0]}</option>" for r in runs)
        opts = "".join(f"<option value='{r}'>{os.path.relpath(r, P('..'))} ({time.strftime('%H:%M', time.localtime(os.path.getmtime(r)))})</option>" for r in runs)
        s = c["synth"]
        body = (HTML % (views, c["ttl"]["sample_text"], P(s["ref_clip"]), s["steps"], s["cfg"], s["duration_scale"])).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if self.path == "/wer":  # body = wav bytes, X-Text = expected sentence -> {wer, heard}
            data = self.rfile.read(int(self.headers["Content-Length"])); text = urllib.parse.unquote(self.headers["X-Text"])
            try:
                heard = whisper(data); body = json.dumps({"wer": round(wer(text, heard), 2), "heard": heard})
            except Exception as ex:
                body = json.dumps({"error": str(ex)})
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body.encode()); return
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
