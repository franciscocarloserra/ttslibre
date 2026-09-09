"""Stage 2: text-to-latent (flow matching, context-sharing batch expansion) + duration predictor.
Samples one training sentence and N held-out sentences every sample_every steps (Whisper WER per sentence in TensorBoard).
Stops at ttl.steps, ttl.max_minutes, or when held-out WER stays under ttl.stop_wer.
Usage: train.py [--run name] [--set key=value ...] [--smoke] [--resume]"""
import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ.setdefault("GRPC_VERBOSITY", "ERROR")  # tensorboard imports TF: hide its cuFFT/cuDNN factory warnings
import json, os, random, sys, time
import torch
from common import load_config, TTL, Tokenizer, compress, lengths_to_mask, P, count_params

c = load_config()
argv = sys.argv[1:]
smoke, resume = "--smoke" in argv, "--resume" in argv
for kv in [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]:  # --set ttl.lr=1e-3
    key, val = kv.split("=", 1); node = c
    for k in key.split(".")[:-1]: node = node[k]
    node[key.split(".")[-1]] = json.loads(val) if val[:1] in "0123456789-[{tfn\"" else val
t, d, K, L = c["ttl"], c["data"], c["latent"]["compress"], c["latent"]["dim"]
prep = P(d["prep_dir"])
run = P(os.path.join("runs", argv[argv.index("--run") + 1] if "--run" in argv else ("ttl_smoke" if smoke else "ttl")))
os.makedirs(run, exist_ok=True)
json.dump(c, open(os.path.join(run, "config.effective.json"), "w"), indent=2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
fps = d["sample_rate"] / d["hop"] / K  # compressed frames per second
tok = Tokenizer.load(os.path.join(prep, "vocab.json"))
stats = torch.load(os.path.join(prep, "latent_stats.pt"))
mean, std = stats["mean"].to(dev)[None, :, None], stats["std"].to(dev)[None, :, None]
rows = [json.loads(l) for l in open(os.path.join(prep, "train.jsonl"))]
val = [json.loads(l) for l in open(os.path.join(prep, "val.jsonl"))]
if d["speaker"]:  # overfit knobs: one speaker and/or a cap on utterances
    rows = [r for r in rows if r["speaker"] == d["speaker"]]; val = [r for r in val if r["speaker"] == d["speaker"]] or rows[:4]
if d.get("max_clip_seconds"):
    rows = [r for r in rows if r["seconds"] < d["max_clip_seconds"]]
if d["max_utts"]:
    rows = rows[: d["max_utts"]]
steps, batch, max_sec = (c["smoke"]["ttl_steps"], c["smoke"]["batch"], c["smoke"]["max_seconds"]) if smoke else (t["steps"], t["batch"], t["max_seconds"])
Ke = t["batch_expand"]
model = TTL(c, len(tok.vocab)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=t["lr"])
amp = torch.bfloat16 if t["amp"] and dev == "cuda" else torch.float32
step = 0
if t.get("init_from") and not resume:  # warm start from an aligned checkpoint (weights only); vocab may have grown
    import unicodedata
    sd = torch.load(P(t["init_from"]), map_location=dev)["model"]; own = model.state_dict()
    base_of = lambda ch: {"ñ": "n"}.get(ch) or unicodedata.normalize("NFD", ch)[0]  # new char rows start from their unaccented letter
    for k, v in sd.items():
        if v.shape != own[k].shape:
            own[k][:v.shape[0]] = v
            for i, ch in enumerate(tok.vocab[v.shape[0]:], v.shape[0]):
                b = base_of(ch); own[k][i] = v[tok.vocab.index(b)] if b in tok.vocab[:v.shape[0]] else own[k][i]
            sd[k] = own[k]
    model.load_state_dict(sd)
if resume and os.path.exists(os.path.join(run, "ttl.pt")):
    ck = torch.load(os.path.join(run, "ttl.pt"), map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
print(f"[train] model {count_params(model)/1e6:.2f}M params (duration predictor {count_params(model.dp)/1e6:.2f}M)  clips {len(rows)}  batch {batch}x{Ke}  budget {steps} steps / {t['max_minutes']} min  {dev}" + (f"  resumed at step {step}" if resume and step else ""), flush=True)


_lat = {}
def load_latent(r):  # latents are cached in RAM after first read (whole prep is ~150 MB): per-clip torch.load left the GPU at 15-27 %
    if r["id"] not in _lat: _lat[r["id"]] = torch.load(os.path.join(prep, "latents", r["id"] + ".pt")).float()
    z = _lat[r["id"]][None].to(dev)
    return compress((z - mean) / std, K)[0]  # (K*L, Tc)


def make_batch(rs, train=True):
    """Returns ids, tmask, z1 (B,C,T), mask (B,1,T), zref (B,C,S), rmask, loss_mask (B,1,T), total_frames."""
    zs = [load_latent(r) for r in rs]
    ids = [torch.tensor(tok.encode(r["text"])) for r in rs]
    total = torch.tensor([z.shape[1] for z in zs], device=dev, dtype=torch.float32)
    maxT = int(max_sec * fps)
    B, C = len(zs), zs[0].shape[0]
    T = min(maxT, max(z.shape[1] for z in zs))
    z1 = torch.zeros(B, C, T, device=dev); lens = torch.zeros(B, dtype=torch.long, device=dev)
    lm = torch.ones(B, 1, T, device=dev)
    S = int(min(t["ref_max_seconds"], max_sec) * fps)
    zref = torch.zeros(B, C, S, device=dev); rlens = torch.zeros(B, dtype=torch.long, device=dev)
    for i, z in enumerate(zs):
        n = z.shape[1]
        if n > T:  # crop long utterances to max_sec (text still full: alignment must tolerate it)
            s = random.randint(0, n - T) if train else 0
            z = z[:, s:s + T]; n = T
        z1[i, :, :n] = z; lens[i] = n
        rl = int(min(random.uniform(t["ref_min_seconds"], t["ref_max_seconds"]) * fps, t["ref_max_fraction"] * n, S))
        rl = max(rl, 1)
        rs0 = random.randint(0, n - rl)
        zref[i, :, :rl] = z[:, rs0:rs0 + rl]; rlens[i] = rl
        lm[i, :, rs0:rs0 + rl] = 0  # mask reference region out of the loss
    tl = torch.tensor([len(x) for x in ids], device=dev)
    idb = torch.zeros(B, int(tl.max()), dtype=torch.long, device=dev)
    for i, x in enumerate(ids): idb[i, :len(x)] = x
    return idb, lengths_to_mask(tl), z1, lengths_to_mask(lens, T)[:, None].float(), zref, lengths_to_mask(rlens, S), lm, total


def fm_loss(z1, mask, lm, text, tmask, style):
    B = z1.shape[0]
    tt = torch.rand(B, device=dev)
    z0 = torch.randn_like(z1)
    zt = (1 - (1 - t["sigma_min"]) * tt)[:, None, None] * z0 + tt[:, None, None] * z1
    target = z1 - (1 - t["sigma_min"]) * z0
    v = model.velocity(zt * mask, mask, tt, text, tmask, style)
    m = mask * lm
    return ((v - target).abs() * m).sum() / (m.sum() * z1.shape[1])


def train_step(rs):
    ids, tmask, z1, mask, zref, rmask, lm, total = make_batch(rs)
    B = z1.shape[0]
    with torch.autocast(dev, dtype=amp, enabled=amp != torch.float32):
        text, style = model.encode(ids, tmask, zref, rmask)
        drop = torch.rand(B, device=dev) < t["p_uncond"]
        ut, us = model.uncond(B, text.shape[1], dev, text.dtype)
        text = torch.where(drop[:, None, None], ut, text); style = torch.where(drop[:, None, None], us, style)
        # context-sharing batch expansion: same conditions, Ke noise/time draws
        rep = lambda x: x.repeat_interleave(Ke, 0)
        loss_fm = fm_loss(rep(z1), rep(mask), rep(lm), rep(text), rep(tmask), rep(style))
        pred = model.dp(ids, tmask, zref, rmask)
        loss_dp = (pred - total.log()).abs().mean()
        loss = loss_fm + c["dp"]["loss_weight"] * loss_dp
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
    opt.step()
    return loss_fm.item(), loss_dp.item(), gn.item()


@torch.no_grad()
def validate():
    model.eval(); fm, dp, n = 0.0, 0.0, 0
    for i in range(0, len(val), batch):
        ids, tmask, z1, mask, zref, rmask, lm, total = make_batch(val[i:i + batch], train=False)
        with torch.autocast(dev, dtype=amp, enabled=amp != torch.float32):
            text, style = model.encode(ids, tmask, zref, rmask)
            fm += fm_loss(z1, mask, lm, text, tmask, style).item()
            dp += (model.dp(ids, tmask, zref, rmask) - total.log()).abs().mean().item(); n += 1
    model.train()
    return fm / n, dp / n


def save():
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "vocab": tok.vocab, "elapsed": time.time() - t0}, os.path.join(run, "ttl.pt"))


from torch.utils.tensorboard import SummaryWriter
tb = SummaryWriter(run, flush_secs=5)
sampler = None


def wer(ref, hyp):
    import re
    from num2words import num2words
    import unicodedata
    strip = lambda x: "".join(ch for ch in unicodedata.normalize("NFD", x.replace("ñ", "n~")) if unicodedata.category(ch) != "Mn").replace("n~", "ñ")
    norm = lambda x: re.sub(r"[^a-zñ' ]", " ", strip(re.sub(r"</?[a-z]{2}>", "", re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=t.get("lang", "en")), x.lower())))).split()
    r, h = norm(ref), norm(hyp)
    dd = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dd[0] = dd[0], i
        for j in range(1, len(h) + 1):
            cur = min(dd[j] + 1, dd[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, dd[j] = dd[j], cur
    return dd[len(h)] / max(len(r), 1)


def whisper(path):
    import urllib.request, subprocess
    e = c["eval"]
    tokn = os.environ.get(e["whisper_token_env"]) or subprocess.run(["bash", "-c", "grep -o 'TTS_TOKEN:-[0-9a-f]*' ~/projects/know-how/local-tts/tts | cut -d- -f2"], capture_output=True, text=True).stdout.strip()
    req = urllib.request.Request(e["whisper_url"], data=open(path, "rb").read(), headers={"Authorization": f"Bearer {tokn}"})
    return urllib.request.urlopen(req, timeout=120).read().decode()


# sample set: "train" = the fixed training sentence, "heldout<i>" = val sentences of the speaker never seen in training
# held-out probes per language: heldout_es1.., heldout_en1.. (first sample_heldout_per_lang val rows of each lang); the stop criterion averages all of them
samples = [("train", t["sample_text"])] + [tuple(x) for x in t.get("sample_extra", [])]
for lang in sorted({r.get("lang", "") for r in val}):
    samples += [(f"heldout_{lang}{i+1}", r["text"]) for i, r in enumerate([r for r in val if r.get("lang", "") == lang][: t["sample_heldout_per_lang"]])]
heldout_hist = []
last_wer = {}  # latest WER per sample sentence, shown on every log line


_gpu = {"v": None}
def _gpu_sample():
    """nvidia-smi sampled in a background thread (a synchronous sample pauses the training loop and measures an idle GPU: 016 logged 17-34% while live util was 90-100%)."""
    import subprocess
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-l", "1"], capture_output=True, text=True, timeout=c["gpu"]["util_sample_seconds"] + 1).stdout
    except subprocess.TimeoutExpired as ex: o = ex.stdout.decode() if ex.stdout else ""
    v = [int(x) for x in o.split() if x.isdigit()]; _gpu["v"] = sum(v) / len(v) if v else -1

def gpu_util():
    """GPU utilization % measured during the previous log window (gpu.util_sample_seconds one-per-second samples, background thread); None until the first sample lands."""
    import threading
    r = _gpu["v"]; _gpu["v"] = None
    threading.Thread(target=_gpu_sample, daemon=True).start()
    return r


def hms(sec):
    sec = int(sec); return f"{sec//3600}h{sec%3600//60:02d}m" if sec >= 3600 else f"{sec//60}m{sec%60:02d}s"


TTY = sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"
C = {"dim": "\033[2m", "bold": "\033[1m", "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m", "cyan": "\033[36m", "0": "\033[0m"}
def col(txt, name): return f"{C[name]}{txt}{C['0']}" if TTY else str(txt)
_hist = {}
def trend(key, v, fmt, lower_is_better=True):
    """Value colored by its own trend: vs the mean of its last log.trend_window printed values, green if better by more than log.trend_tol (relative), red if worse, plain otherwise."""
    h = _hist.setdefault(key, []); o = sum(h) / len(h) if h else None; h.append(v); del h[:-c["log"]["trend_window"]]
    if o is None or o == 0 or abs(v - o) / abs(o) < c["log"]["trend_tol"]: return fmt.format(v)
    return col(fmt.format(v), "green" if (v < o) == lower_is_better else "red")
def emit(stage, msg, plain=None):
    """[stage] elapsed (step)  msg -> terminal (colored) and progress.log (plain). One line format for run.sh, preflight.sh and train.py."""
    head = f"{hms(time.time()-t0)} ({step})"
    print(f"{col(f'[{stage}]', 'dim')} {head}  {msg}", flush=True)
    log.write(f"[{stage}] {head}  {plain if plain is not None else msg}\n"); log.flush()
best = [9e9, 0]  # best held-out mean WER and its step
def wer_line(color):
    """Per language: held-out mean (novel), then train and best so far. Colored by trend and by the wer_good/wer_bad bands."""
    f = (lambda k, w: trend(k, w, "{:.2f}")) if color else (lambda k, w: f"{w:.2f}")
    langs = sorted({n.split("_")[1][:2] for n in last_wer if "_" in n}) or [""]; parts = []
    for l in langs:
        hs = [w for n, w in last_wer.items() if n.startswith("heldout") and (not l or n.split("_")[1].startswith(l))]
        nv = [w for n, w in last_wer.items() if n.startswith("novel") and (not l or n.endswith(l))]
        if hs or nv: parts.append((l or "all") + " " + (f(f"h{l}", sum(hs) / len(hs)) if hs else "") + (f" (novel {f(f'n{l}', nv[0])})" if nv else ""))
    return "  ".join(parts) + f"  train {f('train', last_wer.get('train', 1.0))}  best {best[0]:.2f} @ {best[1]}"


@torch.no_grad()
def sample(step):
    """Synthesize every sample sentence with the current weights; log audio, transcript and WER to TensorBoard and disk."""
    global sampler
    import soundfile as sf
    from synth import Synth
    if sampler is None:
        sampler = Synth.__new__(Synth); sampler.c, sampler.dev, sampler.K, sampler.fps = c, dev, K, fps
        sampler.tok, sampler.mean, sampler.std = tok, mean, std
        from common import LatentAE, Mel
        sampler.ae = LatentAE(c).to(dev).eval(); sampler.ae.load_state_dict(torch.load(P("runs/ae/ae.pt"), map_location=dev))
        sampler.mel = Mel(d).to(dev)
        from vocos import Vocos
        sampler.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).eval()
        sampler.zref, sampler.rmask = sampler.style_from_wav(P(c["synth"]["ref_clip"]))
        for name, text in samples:  # once per run: text and original recording of each sample sentence
            tb.add_text(f"{name}/text", text, 0)
            orig = [r for r in rows + val if r["text"] == text]
            if orig:
                w, sr = sf.read(os.path.join(P(d["raw_root"]), orig[0]["path"]), dtype="float32")  # mixed prep: path is absolute, join returns it
                tb.add_audio(f"{name}/original", torch.from_numpy(w if w.ndim == 1 else w.mean(1))[None], 0, sample_rate=sr)
    sampler.model = model; model.eval()
    os.makedirs(os.path.join(run, "samples"), exist_ok=True)
    ws = {}
    for name, text in samples:
        wav, _ = sampler(text, sampler.zref, sampler.rmask, steps=t["sample_steps"], cfg=t["sample_cfg"])
        tb.add_audio(f"{name}/generated", torch.from_numpy(wav)[None], step, sample_rate=d["sample_rate"])
        path = os.path.join(run, "samples", f"{name}_step_{step:06d}.wav")
        sf.write(path, wav, d["sample_rate"])
        try:  # Whisper is the validator
            hyp = whisper(path); w = wer(text, hyp); ws[name] = w; last_wer[name] = w
            tb.add_scalar(f"wer/{name}", w, step); tb.add_text(f"{name}/whisper", f"wer={w:.2f} | {hyp}", step)
            log.write(f"[sample] {hms(time.time()-t0)} ({step})  {name} wer {w:.2f}  heard: {hyp}\n"); log.flush()  # per-probe transcripts: progress.log and the panel only
        except Exception as ex:
            emit("sample", f"{name} whisper failed: {ex}")
    model.train()
    hw = [w for n, w in ws.items() if n.startswith("heldout")]
    if hw:
        m = sum(hw) / len(hw); heldout_hist.append(m); tb.add_scalar("wer/heldout_mean", m, step)
        if m < min(heldout_hist[:-1], default=9e9):  # best checkpoint by held-out WER
            torch.save({"model": model.state_dict(), "step": step, "vocab": tok.vocab, "heldout_wer": m}, os.path.join(run, "best.pt")); best[:] = [m, step]
    emit("wer", wer_line(True), wer_line(False))


log = open(os.path.join(run, "progress.log"), "a")
t0, order, kept = time.time(), [], 0
ips_ref = (time.time(), step)  # (time, step) at the last printed train line, for it/s
if resume and step: t0 -= ck.get("elapsed", 0)  # budget clock continues across restarts
if step == 0: sample(0)  # baseline sample: what the untrained (or warm-started) model says
while step < steps:
    if not order:
        order = rows[:]; random.shuffle(order)
    rs, order = order[:batch], order[batch:]
    lr = t["lr"] * min(1.0, (step + 1) / t["warmup"]) * (0.5 ** (step // t["lr_halve_every"]))
    for g in opt.param_groups: g["lr"] = lr
    lf, ld, gn = train_step(rs)
    step += 1
    if lf != lf or ld != ld:  # NaN guard: stop, keep the last good checkpoint on disk
        emit("stop", col("FAILED: loss is NaN", "red"), "FAILED: loss is NaN"); break
    if step % t["log_every"] == 0 or smoke:
        gu = gpu_util() if step % (t["log_every"] * c["gpu"]["util_every_logs"]) == 0 else None
        if gu is not None: tb.add_scalar("train/gpu_util", gu, step)
        if step % (t["log_every"] * c["log"]["print_every_logs"]) == 0 or smoke:  # terminal: every print_every_logs log windows; progress.log: every log_every steps
            ips = (step - ips_ref[1]) / max(time.time() - ips_ref[0], 1e-6); ips_ref = (time.time(), step)
            left = max(0, min(t["max_minutes"] * 60 - (time.time() - t0), (steps - step) / max(ips, 1e-6)))
            plain = f"loss {lf:.3f}/{ld:.3f}  lr {lr:.1e}  grad {gn:.2f}  {ips:.1f} it/s  eta {hms(left)}" + (f"  gpu {gu:.0f}%" if gu is not None else "")
            gcol = lambda u: col(f"{u:.0f}%", "green" if abs(u - c["gpu"]["util_target_pct"]) <= c["gpu"]["util_tolerance_pct"] else "red")
            emit("train", f"loss {trend('lf', lf, '{:.3f}')}/{trend('ld', ld, '{:.3f}')}  lr {lr:.1e}  grad {trend('gn', gn, '{:.2f}')}  {trend('ips', ips, '{:.1f}', False)} it/s  eta {hms(left)}" + (f"  gpu {gcol(gu)}" if gu is not None else ""), plain)
        else:
            log.write(f"[train] {hms(time.time()-t0)} ({step})  loss {lf:.3f}/{ld:.3f}  lr {lr:.1e}  grad {gn:.2f}" + (f"  gpu {gu:.0f}%" if gu is not None else "") + "\n"); log.flush()
        tb.add_scalar("train/fm", lf, step); tb.add_scalar("train/dp", ld, step); tb.add_scalar("train/lr", lr, step)
    if step % t["val_every"] == 0 or step == steps:
        vf, vd = validate()
        emit("val", f"loss {trend('vf', vf, '{:.3f}')}/{trend('vd', vd, '{:.3f}')}", f"loss {vf:.3f}/{vd:.3f}")
        tb.add_scalar("val/fm", vf, step); tb.add_scalar("val/dp", vd, step)
    if step % t["sample_every"] == 0 or step == steps:
        sample(step)
    flag = os.path.join(run, "save_now")  # touched by the panel ("checkpoint now"): snapshot the current weights at the next log step
    if os.path.exists(flag):
        os.remove(flag); torch.save({"model": model.state_dict(), "step": step, "vocab": tok.vocab, "elapsed": time.time() - t0}, os.path.join(run, f"ttl_{hms(time.time()-t0)}_step{step}.pt")); emit("ckpt", f"snapshot on demand ttl_{hms(time.time()-t0)}_step{step}.pt")
    if step % t["ckpt_every"] == 0 or step == steps:
        save()
        # partial checkpoints: ttl.keep_checkpoints snapshots spread over the time budget (weights only)
        q = int((time.time() - t0) / (t["max_minutes"] * 60 / t["keep_checkpoints"]))
        if q > kept and q <= t["keep_checkpoints"]:
            kept = q; torch.save({"model": model.state_dict(), "step": step, "vocab": tok.vocab, "elapsed": time.time() - t0}, os.path.join(run, f"ttl_{hms(time.time()-t0)}_step{step}.pt")); emit("ckpt", f"snapshot {q}/{t['keep_checkpoints']} ttl_{hms(time.time()-t0)}_step{step}.pt")
    stop = None
    if time.time() - t0 > t["max_minutes"] * 60: stop = f"max_minutes={t['max_minutes']}"
    W = t["stop_wer_window"]
    if len(heldout_hist) >= W and sum(heldout_hist[-W:]) / W < t["stop_wer"]: stop = f"heldout WER mean of last {W} < {t['stop_wer']}"
    if time.time() - t0 > t["fail_after_minutes"] * 60 and last_wer.get("train", 1.0) > t["fail_wer"]: stop = f"FAILED: train WER {last_wer.get('train', 1.0):.2f} > {t['fail_wer']} after {t['fail_after_minutes']} min"
    if stop:
        emit("stop", col(stop, "red" if stop.startswith("FAILED") else "cyan"), stop); save(); break
vf, vd = validate()
json.dump({"params": count_params(model), "dp_params": count_params(model.dp), "steps": step, "val_fm": vf, "val_dp_logl1": vd,
           "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0, "seconds": time.time() - t0,
           "batch": batch, "batch_expand": Ke}, open(os.path.join(run, "summary.json"), "w"), indent=1)
emit("done", f"val loss audio {vf:.3f} duration {vd:.3f}  summary.json written")
