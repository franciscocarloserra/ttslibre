"""Experiments panel: WER chart + sample rounds of any run, and interactive synthesis from any checkpoint.
Lives in experiments/; imports code and config from one experiment dir (--exp NAME, default: newest).
Usage: panel.py [--exp 004-words-or-sentences]   -> http://localhost:<panel.port>"""
import glob, io, json, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
EXP = argv[argv.index("--exp") + 1] if "--exp" in argv else sorted(os.path.relpath(d, HERE) for d in glob.glob(os.path.join(HERE, "*", "[0-9][0-9][0-9]-*")) if os.path.isdir(d))[-1]  # group/NNN-slug
os.chdir(os.path.join(HERE, EXP)); sys.path.insert(0, os.getcwd())
import torch, soundfile as sf
from common import load_config, P
from synth import Synth

c = load_config(); p = c["panel"]
LOG_RE = [re.compile(r"SAMPLE (\w+) step=(\d+) wer=([\d.]+) \| (.*)"), re.compile(r"(\S+) \((\d+)\)  (\w+) wer ([\d.]+)  whisper heard: (.*)"),
          re.compile(r"SAMPLE step=(\d+) wer=([\d.]+) \| (.*)")]
LABEL = {"train": "training sentence (should be memorized)", "knownwords": "known words, new order", "heldout1": "never-seen words", "heldout2": "never-seen words"}


def _P(rel):
    """P() with fallback one level up: config.effective.json written before the 2026-09-09 regrouping has dataset paths one level short."""
    p = P(rel); return p if os.path.exists(p) else P(os.path.join("..", rel))


def sentences(run_dir):
    """name -> {text, original wav path or None}, reconstructed from the run's effective config and the prep split."""
    cfg = os.path.join(run_dir, "config.effective.json")
    cc = json.load(open(cfg)) if os.path.exists(cfg) else c
    t, d = cc["ttl"], cc["data"]; prep = _P(d["prep_dir"])
    train = [json.loads(l) for l in open(os.path.join(prep, "train.jsonl"))]; val = [json.loads(l) for l in open(os.path.join(prep, "val.jsonl"))]
    if d.get("speaker"): train = [r for r in train if r["speaker"] == d["speaker"]]; val = [r for r in val if r["speaker"] == d["speaker"]] or train[:4]
    names = [("train", t["sample_text"])] + [tuple(x) for x in t.get("sample_extra", [])] + [(f"heldout{i+1}", r["text"]) for i, r in enumerate(val[: t.get("sample_heldout_n", 0)])]
    out = {}
    for n, text in names:
        orig = [r for r in train + val if r["text"] == text]
        out[n] = {"text": text, "label": LABEL.get(n, n), "original": os.path.join(_P(d.get("raw_dir") or d["raw_root"]), orig[0]["path"]) if orig else None}
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
import threading
cache = {}  # ckpt path -> [mtime, Synth, last_used]
_lock = threading.Lock()


def get_synth(run):
    mt = os.path.getmtime(run)
    with _lock:
        if run not in cache or cache[run][0] != mt:  # reload when the training run overwrote the checkpoint
            cache[run] = [mt, Synth(c, run=run, device=p["device"]), 0]
        cache[run][2] = time.time()
        return cache[run][1]


def _evict():  # free VRAM: drop models idle for more than panel.hot_seconds (GPU is shared with training)
    while True:
        time.sleep(10)
        with _lock:
            for k in [k for k, v in cache.items() if time.time() - v[2] > p["hot_seconds"]]: del cache[k]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
threading.Thread(target=_evict, daemon=True).start()


HTML = """<!doctype html><meta charset=utf-8><title>ttslibre panel</title>
<style>body{background:#111;color:#ddd;font:15px system-ui;margin:0;padding:1em}
html,body{width:100%%}.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:2em;width:100%%}
input,select,textarea,button{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:.5em;font:inherit;width:100%%;box-sizing:border-box;margin:.3em 0}
button{background:#2a6;color:#000;cursor:pointer}button.play{width:5.5em;padding:.2em .4em;margin:0;background:#333;color:#ddd}details{margin:.5em 0;color:#999}pre{color:#8c8;white-space:pre-wrap}
.bw{color:#f66;text-decoration:underline}.dl{color:#f66;text-decoration:line-through}.bc{background:#a22;color:#fff;border-radius:2px}.dim{color:#777}
table{width:100%%;font-size:13px;border-collapse:collapse}td{padding:2px 4px;vertical-align:middle}svg{background:#181818;border-radius:4px;display:block;width:100%%}</style>
<div class=cols><div>
<h3>run</h3><select id=view onchange="load();ckpts()">%s</select> checkpoint <select id=ckpt></select>
<svg id=chart viewBox="0 0 700 220"></svg>
<h3>generate</h3>
voice <select id=refsel onchange="ref.value=this.value"></select>
<textarea id=text rows=3 onkeydown="if(event.key=='Enter'&&!event.shiftKey){event.preventDefault();go()}">%s</textarea>
<button onclick="go()">generate</button>
<div id=gen style="margin-top:.5em"></div>
<canvas id=wv width=1200 height=76 style="width:100%%;height:76px;background:#07080a;border-radius:4px;display:block;margin-top:.4em"></canvas>
<canvas id=sg width=1200 height=128 style="width:100%%;height:150px;background:#07080a;border-radius:4px;display:block;margin-top:.3em;image-rendering:pixelated"></canvas>
<div id=words style="margin:.4em 0;line-height:1.8"></div><pre id=out></pre>
<details><summary>advanced</summary>
ref clip <input id=ref value="%s" size=60>
steps <input id=steps value="%d"> cfg <input id=cfg value="%s"> duration scale <input id=dur value="%s"> lang <input id=lang value="%s" size=3 title="language tag for 016+ checkpoints (es/en); empty = config synth.lang"></details>
<details><summary>compare two models</summary>
A <select id=ma>%s</select> steps <input id=sa value="%d"> cfg <input id=ca_cfg value="%s"><br>
B <select id=mb>%s</select> steps <input id=sb value="%d"> cfg <input id=cb_cfg value="%s">
<button id=cmpb onclick="cmp()">generate with both</button>
<div id=cmpout></div></details>
</div><div><div id=rounds></div><h3>training runs, longest first</h3><div id=ov></div></div></div>
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
const EL={};for(const r of d)if(r.elapsed)EL[r.step]=r.elapsed;const ks=Object.keys(EL).map(Number).sort((a,b)=>a-b);
for(let i=1;i<=5;i++){const s=Math.round(mx*i/5);const k=ks.length?ks.reduce((p,q)=>Math.abs(q-s)<Math.abs(p-s)?q:p):s;c.append(el('line',{x1:X(s),x2:X(s),y1:H-B,y2:H-B+4,stroke:'#666'}));const t=el('text',{x:X(s),y:H-6,fill:'#888','font-size':11,'text-anchor':'end'});const hrs=e=>{const m=/(\d+)h(\d+)m|(\d+)m(\d+)s/.exec(e||'');return m?(m[1]?+m[1]+m[2]/60:m[3]/60).toFixed(1)+' h':''};t.textContent=ks.length?hrs(EL[k]):'step '+s;c.append(t)}
const steps=[...new Set(d.map(r=>r.step))].sort((a,b)=>b-a);const N=steps.length;let h='<h3>rounds, newest first</h3>';
const legend=Object.entries(S).map(([n,x])=>`<div style="color:${COL[n]||'#aaa'}"><b>${n}</b> = ${x.label}: <i>${x.text}</i>${x.original?` ${btn(`/wav?run=x&f=x&orig=${encodeURIComponent(x.original)}`)} (original recording)`:''}</div>`).join('');
h+='<details open><summary>the sentences</summary>'+legend+'</details>';
steps.forEach((s,i)=>{const rs=d.filter(r=>r.step==s);h+=`<div style="border:1px solid #333;border-radius:6px;padding:.5em;margin:.6em 0"><div style="color:#aaa;margin-bottom:.3em">round ${N-i} of ${N} &middot; step ${s}${rs[0].elapsed?' &middot; '+rs[0].elapsed+' into the run':''}</div><table>`;
for(const r of rs) h+=`<tr><td style="color:${COL[r.name]||'#aaa'};white-space:nowrap;width:9em"><b>${r.name}</b><br>wer ${r.wer.toFixed(2)}</td><td style="width:6em">${btn(`/wav?run=${encodeURIComponent(view.value)}&f=${r.name}_step_${String(s).padStart(6,'0')}.wav`)}</td><td style="color:#999">input: <span style="color:#ddd">${(S[r.name]||{}).text||''}</span><br>whisper: <span style="color:#ddd">${r.heard}</span></td></tr>`;h+='</table></div>'});
const rd=document.getElementById('rounds');if(rd.dataset.h!==h){rd.innerHTML=h;rd.dataset.h=h;durs()}}
async function ckpts(){const L=await (await fetch('/ckpts?run='+encodeURIComponent(view.value))).json();const cur=ckpt.value;ckpt.innerHTML=L.map(x=>`<option value='${x.path}'>${x.name} (${x.time})</option>`).join('');if([...ckpt.options].some(o=>o.value===cur))ckpt.value=cur}
let OVN=%d;async function ov(){const R=await (await fetch('/overview')).json();const rows=R.slice(0,OVN);let h='';
for(const r of rows)h+=`<div style="border:1px solid #333;border-radius:6px;padding:.5em;margin:.4em 0"><div><b>${r.wall}</b> <span style="color:#aaa">${r.exp}/${r.run}</span></div><div style="color:#ddd;margin:.2em 0">${r.question}</div><div style="color:#888;font-size:13px">${r.data}${r.hours!=null?` · ${r.hours} h · ${r.clips} clips · ${r.speakers} spk`:''} · init ${r.init}</div></div>`;
if(R.length>OVN)h+=`<button class=play style="width:auto" onclick="OVN+=%d;ov()">+%d more (${R.length-OVN} left)</button>`;ov_el.innerHTML=h}
const ov_el=document.getElementById('ov');ov();
load();ckpts();setInterval(load,15000);setInterval(ckpts,60000);
(async()=>{const R=await (await fetch('/refs')).json();refsel.innerHTML=Object.entries(R).map(([k,v])=>`<option value="${v}">${k}</option>`).join('');ref.value=refsel.value})();
let VZ=null;
function drawWave(v,bad){const c=wv.getContext('2d'),W=wv.width,H=wv.height;c.fillStyle='#07080a';c.fillRect(0,0,W,H);
for(const [s,e] of bad||[]){c.fillStyle='rgba(255,60,60,.28)';c.fillRect(W*s/v.duration,0,Math.max(2,W*(e-s)/v.duration),H)}
c.fillStyle='#7cf2b0';const n=v.wave.length;for(let i=0;i<n;i++){const [lo,hi]=v.wave[i];c.fillRect(i*W/n,H/2-hi*H/2,Math.max(1,W/n),Math.max(1,(hi-lo)*H/2))}}
function drawSpec(v){const s=Uint8Array.from(v.spec.match(/../g).map(h=>parseInt(h,16)));const [F,M]=v.shape;sg.width=F;sg.height=M;const g=sg.getContext('2d'),im=g.createImageData(F,M);
for(let f=0;f<F;f++)for(let m=0;m<M;m++){const x=s[f*M+m]/255,o=4*((M-1-m)*F+f);im.data[o]=255*Math.min(1,x*1.6);im.data[o+1]=255*Math.pow(x,1.5);im.data[o+2]=90+165*Math.max(0,1-x*2.2)*(x>0.05?1:0.3);im.data[o+3]=255}g.putImageData(im,0,0)}
async function showViz(blob){VZ=await (await fetch('/viz',{method:'POST',body:blob})).json();drawWave(VZ);drawSpec(VZ)}
function showWords(w){const esc=t=>t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ref=w.ref.map(r=>`<span class="${r.op=='eq'?'':r.op=='del'?'dl':'bw'}">${esc(r.w)}</span>`).join(' ');
const hyp=w.hyp.map(h=>h.op=='eq'?esc(h.w):[...h.w].map((ch,i)=>h.bad[i]?`<span class=bc>${esc(ch)}</span>`:esc(ch)).join('')).join(' ');
words.innerHTML=`<div><span class=dim>expected:</span> ${ref}</div><div><span class=dim>heard:</span> ${hyp}${w.has_ts?'':' <span class=dim>(no timestamps: restart whisper server)</span>'}</div>`;
if(VZ&&w.has_ts){const bad=w.hyp.filter(h=>h.op!='eq'&&h.start!=null).map(h=>[h.start,h.end]);
// deletions: mark the gap after the previous aligned hyp word
const hs=w.hyp.filter(h=>h.start!=null);w.ref.forEach((r,i)=>{if(r.op!='del')return;const prev=hs[Math.min(i,hs.length)-1];if(prev)bad.push([prev.end,prev.end+0.15])});drawWave(VZ,bad)}}
async function one(run,slot,st,cf){const r=await fetch('/synth',{method:'POST',body:JSON.stringify({run:run,text:text.value,ref:ref.value,steps:st,cfg:cf,dur:+dur.value,lang:lang.value})});
if(!r.ok){slot.innerHTML='<pre>'+await r.text()+'</pre>';return}const blob=await r.blob();slot.innerHTML=btn(URL.createObjectURL(blob))+' <pre style="display:inline">'+decodeURIComponent(r.headers.get('x-info'))+'</pre>';durs();
const w=await (await fetch('/wer',{method:'POST',headers:{'X-Text':encodeURIComponent(text.value)},body:blob})).json();slot.innerHTML+=`<pre>${w.error?'wer failed '+w.error:`wer ${w.wer.toFixed(2)} · whisper: ${w.heard}`}</pre>`}
async function cmp(){cmpb.disabled=true;cmpout.innerHTML=`<div><b>A</b> ${ma.options[ma.selectedIndex].text}<div id=ca>generating...</div></div><div><b>B</b> ${mb.options[mb.selectedIndex].text}<div id=cb>generating...</div></div>`;
await one(ma.value,ca,+sa.value,+ca_cfg.value);await one(mb.value,cb,+sb.value,+cb_cfg.value);cmpb.disabled=false}
async function go(){const b=document.querySelector('button');b.disabled=true;out.textContent='generating...';
const r=await fetch('/synth',{method:'POST',body:JSON.stringify({run:ckpt.value,text:text.value,ref:ref.value,steps:+steps.value,cfg:+cfg.value,dur:+dur.value,lang:lang.value})});
if(!r.ok){out.textContent=await r.text();b.disabled=false;return}
const blob=await r.blob();const u=URL.createObjectURL(blob);gen.innerHTML=btn(u);durs();pl(gen.firstChild);out.textContent=decodeURIComponent(r.headers.get('x-info'))+'\\nwer: checking...';b.disabled=false;words.innerHTML='';showViz(blob);
const w=await (await fetch('/wer',{method:'POST',headers:{'X-Text':encodeURIComponent(text.value)},body:blob})).json();out.textContent=out.textContent.replace('wer: checking...',w.error?'wer: failed '+w.error:`wer ${w.wer.toFixed(2)}\\nwhisper: ${w.heard}`);if(!w.error)showWords(w)}
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


import numpy as np, difflib


def mel_db(w, sr, sp):
    """numpy mel spectrogram in dB relative to the clip max, frames x mels (same as supertonic-voicelab panel)."""
    n, h = sp["n_fft"], sp["hop"]; win = np.hanning(n).astype(np.float32); pad = np.pad(w, n // 2)
    frames = np.lib.stride_tricks.sliding_window_view(pad, n)[::h] * win; S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = lambda f: 2595 * np.log10(1 + f / 700); imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(0), mel(sp["fmax_hz"]), sp["n_mels"] + 2)); freqs = np.fft.rfftfreq(n, 1 / sr)
    lo, cc, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    fb = np.maximum(0, np.minimum((freqs - lo) / (cc - lo), (hi - freqs) / (hi - cc))).astype(np.float32)
    m = 10 * np.log10(np.maximum(S @ fb.T, 1e-10)); return m - m.max()


def viz(data):
    """wav bytes -> {duration, wave: [[lo,hi]...] per pixel column (-1..1), spec: hex uint8 frames x mels}."""
    sp = p["viz"]; w, sr = sf.read(io.BytesIO(data), dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    n = sp["wave_px"]; cols = np.array_split(w, n) if len(w) >= n else [w]
    wave = [[round(float(x.min()), 3), round(float(x.max()), 3)] for x in cols]
    m = mel_db(w, sr, sp); u8 = np.clip((m + 80) / 80 * 255, 0, 255).astype(np.uint8)  # 80 dB range
    return {"duration": len(w) / sr, "wave": wave, "spec": u8.tobytes().hex(), "shape": list(u8.shape)}


def align(ref, hyp):
    """Levenshtein backtrace on normalized tokens -> list of (op, i, j): op in eq/sub/del/ins; i idx into ref, j into hyp."""
    a, b = _norm(ref), _norm(hyp); D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): D[i][0] = i
    for j in range(len(b) + 1): D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1): D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], len(a), len(b)
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]): ops.append(("eq" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1)); i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1: ops.append(("del", i - 1, None)); i -= 1
        else: ops.append(("ins", None, j - 1)); j -= 1
    return a, b, ops[::-1]


def diff_chars(x, y):
    """chars of y that differ from x (substituted word pair) -> list of bools per char of y."""
    bad = [True] * len(y)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, x, y).get_opcodes():
        if tag == "equal":
            for k in range(j1, j2): bad[k] = False
    return bad


def wer_report(text, data):
    """{wer, heard, has_ts, ref:[{w,op}], hyp:[{w,op,start,end,bad:[bool per char]}]} using ?words=1 when the whisper server supports it."""
    try:
        raw = whisper(data, words=True); j = json.loads(raw); heard, words = j["text"], j["words"]; has_ts = True
    except Exception:
        heard, words, has_ts = whisper(data), [], False
    a, b, ops = align(text, heard)
    # map normalized hyp tokens back onto timestamped words (both derive from the same string, in order)
    tw = [t for w in words for t in _norm(w["w"]) for _ in [0]]  # normalized tokens in word order
    spans = [(w["start"], w["end"]) for w in words for _ in _norm(w["w"])]
    ref = [{"w": t, "op": "eq"} for t in a]; hyp = [{"w": t, "op": "eq", "bad": [False] * len(t)} for t in b]
    if has_ts and len(spans) == len(b):
        for k, sp in enumerate(spans): hyp[k]["start"], hyp[k]["end"] = sp
    for op, i, jj in ops:
        if op == "sub": ref[i]["op"] = "sub"; hyp[jj]["op"] = "sub"; hyp[jj]["bad"] = diff_chars(a[i], b[jj])
        elif op == "del": ref[i]["op"] = "del"
        elif op == "ins": hyp[jj]["op"] = "ins"; hyp[jj]["bad"] = [True] * len(b[jj])
    return {"wer": round(wer(text, heard), 2), "heard": heard, "has_ts": has_ts, "ref": ref, "hyp": hyp}


def whisper(data, words=False):
    req = urllib.request.Request(_e["whisper_url"] + (p["viz"]["whisper_words_query"] if words else ""), data=data, headers={"Authorization": f"Bearer {_tok}"})
    return urllib.request.urlopen(req, timeout=120).read().decode()


_ov_cache = {}


def _hours(prep):
    """(hours, clips, speakers) of a prep dir's train.jsonl, cached by mtime."""
    f = os.path.join(prep, "train.jsonl")
    if not os.path.exists(f): return None
    mt = os.path.getmtime(f)
    if _ov_cache.get(f, (None,))[0] != mt:
        rows = [json.loads(l) for l in open(f)]
        _ov_cache[f] = (mt, round(sum(r.get("seconds", 0) for r in rows) / 3600, 1), len(rows), len({r.get("speaker") for r in rows}))
    return _ov_cache[f][1:]


def _elapsed(log):
    """last 'XhYYm' / 'XXmYYs' prefix in progress.log -> seconds, string."""
    last = ""
    for line in open(log, errors="replace"):
        m = re.match(r"(\d+)h(\d+)m|(\d+)m(\d+)s", line)
        if m: last = m
    if not last: return 0, ""
    g = last.groups(); sec = int(g[0]) * 3600 + int(g[1]) * 60 if g[0] else int(g[2]) * 60 + int(g[3])
    return sec, last.group(0)


def overview():
    """One row per training run (progress.log present): experiment, run, wall time, data, init, question. Sorted by wall time, longest first."""
    rows, seen = [], set()
    for log in glob.glob(os.path.join(HERE, "*", "[0-9][0-9][0-9]-*", "runs", "*", "progress.log")):
        rd = os.path.dirname(log); real = os.path.realpath(rd)
        if real in seen or "todelete" in rd or "smoke" in rd or os.path.islink(rd): continue  # symlinked runs (015 teacher -> 014) are listed once, at their owner
        seen.add(real); exp = os.path.relpath(os.path.dirname(os.path.dirname(rd)), HERE)
        cfgf = os.path.join(rd, "config.effective.json"); cc = json.load(open(cfgf)) if os.path.exists(cfgf) else {}
        d = cc.get("data", {}); prep = d.get("prep_dir", ""); h = None
        for base in (os.path.join(HERE, exp), os.path.join(HERE, exp, "..")):  # configs written before the 2026-09-09 regrouping are one level short
            h = h or (prep and _hours(os.path.join(base, prep)))
        init = (cc.get("ttl", {}) or {}).get("init_from") or "scratch"
        q = ""
        rf = os.path.join(HERE, exp, "README.md")
        if os.path.exists(rf):
            m = re.search(r"\*\*Short\.\*\*\s*(.+)", open(rf).read()) or re.search(r"\*\*(?:Question|Goal)\.\*\*\s*(.+)", open(rf).read()); q = m.group(1) if m else ""  # one plain line per experiment README
        sec, el = _elapsed(log)
        rows.append({"exp": exp, "run": os.path.basename(rd), "wall_s": sec, "wall": el, "data": os.path.basename(prep.rstrip("/")) if prep else "", "hours": h and h[0], "clips": h and h[1], "speakers": h and h[2], "init": (lambda m: f"{m.group(1)}/{m.group(2)}" if m else init)(re.search(r"(\d{3})-[^/]*/runs/([^/]+)", init)), "question": q})
    return sorted(rows, key=lambda r: -r["wall_s"])


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/overview":
            body = json.dumps(overview()).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body); return
        if self.path == "/refs":  # voices: 007 reference clips + voices/*.pt
            r = json.load(open(os.path.join(HERE, "30-voices", "007-zero-shot-voice", "config.json")))["refs"]["refs"]
            R = {k: os.path.abspath(os.path.join(HERE, "30-voices", "007-zero-shot-voice", v)) for k, v in r.items()}
            R.update({os.path.basename(f)[:-3]: f for f in sorted(glob.glob(os.path.join(HERE, "voices", "*.pt")))})  # voicepacks made by voice.py
            body = json.dumps(R).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/ckpts?"):
            rd = self.path.split("run=")[1].replace("%2F", "/")
            fs = sorted([f for f in glob.glob(os.path.join(rd, "*.pt")) if re.match(r"(ttl|best)(_.*)?\.pt$", os.path.basename(f))], key=os.path.getmtime, reverse=True)
            body = json.dumps([{"path": f, "name": os.path.basename(f), "time": time.strftime("%H:%M", time.localtime(os.path.getmtime(f)))} for f in fs]).encode()
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
        runs = sorted([r for r in glob.glob(P(p["runs_glob"]), recursive=True) if "todelete" not in r and "smoke" not in r and re.match(r"(ttl|best)(_.*)?\.pt$", os.path.basename(r))], key=os.path.getmtime, reverse=True)
        views = "".join(f"<option value='{v}'>{os.path.relpath(v, P('../..'))}</option>" for v in dict.fromkeys(os.path.dirname(r) for r in runs))
        s = c["synth"]; files = "".join(f"<option value='{r}'>{os.path.relpath(r, P('../..'))}</option>" for r in runs)
        body = (HTML % (views, c["ttl"]["sample_text"], P(s["ref_clip"]), s["steps"], s["cfg"], s["duration_scale"], s.get("lang", ""), files, s["steps"], s["cfg"], files, s["steps"], s["cfg"], p["overview_rows"], p["overview_rows"], p["overview_rows"])).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if self.path == "/wer":  # body = wav bytes, X-Text = expected sentence -> {wer, heard}
            data = self.rfile.read(int(self.headers["Content-Length"])); text = urllib.parse.unquote(self.headers["X-Text"])
            try:
                body = json.dumps(wer_report(text, data))
            except Exception as ex:
                body = json.dumps({"error": str(ex)})
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body.encode()); return
        if self.path == "/viz":  # body = wav bytes -> waveform min/max columns + mel spectrogram
            body = json.dumps(viz(self.rfile.read(int(self.headers["Content-Length"])))).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body); return
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            S = get_synth(q["run"])
            zref, rmask = S.style(q["ref"]) if hasattr(S, "style") else S.style_from_wav(q["ref"])
            t0 = time.time(); wav, dur = S(q["text"], zref, rmask, steps=q["steps"], cfg=q["cfg"], duration_scale=q["dur"], lang=q.get("lang") or None); el = time.time() - t0
            buf = io.BytesIO(); sf.write(buf, wav, c["data"]["sample_rate"], format="WAV")
            info = json.dumps({"ckpt": os.path.relpath(q["run"], P("../..")), "steps": q["steps"], "cfg": q["cfg"], "lang": q.get("lang") or S.c["synth"].get("lang", ""), "seconds": round(dur, 2), "gen_s": round(el, 2), "rtf": round(el / max(dur, 1e-6), 2), "device": S.dev})
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("X-Info", info); self.end_headers(); self.wfile.write(buf.getvalue())
        except Exception as ex:
            self.send_response(500); self.end_headers(); self.wfile.write(str(ex).encode())

    def log_message(self, *a): pass


print(f"http://localhost:{p['port']}", flush=True)
ThreadingHTTPServer((p.get("host", "127.0.0.1"), p["port"]), H).serve_forever()
