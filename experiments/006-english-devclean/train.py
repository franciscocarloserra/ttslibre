"""Stage 2: text-to-latent (flow matching, context-sharing batch expansion) + duration predictor.
Samples one training sentence and N held-out sentences every sample_every steps (Whisper WER per sentence in TensorBoard).
Stops at ttl.steps, ttl.max_minutes, or when held-out WER stays under ttl.stop_wer.
Usage: train.py [--run name] [--set key=value ...] [--smoke] [--resume]"""
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
if t.get("init_from") and not resume:  # warm start from an aligned checkpoint (weights only)
    model.load_state_dict(torch.load(P(t["init_from"]), map_location=dev)["model"])
if resume and os.path.exists(os.path.join(run, "ttl.pt")):
    ck = torch.load(os.path.join(run, "ttl.pt"), map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
print(f"ttl params={count_params(model)/1e6:.2f}M (dp {count_params(model.dp)/1e6:.2f}M) train={len(rows)} steps={steps} batch={batch}x{Ke} dev={dev} fps={fps:.2f}", flush=True)


def load_latent(r):
    z = torch.load(os.path.join(prep, "latents", r["id"] + ".pt")).float()[None].to(dev)
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
    opt.step()
    return loss_fm.item(), loss_dp.item()


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
    norm = lambda x: re.sub(r"[^a-z' ]", " ", re.sub(r"\d+", lambda m: num2words(int(m.group())), x.lower())).split()
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
samples = [("train", t["sample_text"])] + [tuple(x) for x in t.get("sample_extra", [])] + [(f"heldout{i+1}", r["text"]) for i, r in enumerate(val[: t["sample_heldout_n"]])]
heldout_hist = []
last_wer = {}  # latest WER per sample sentence, shown on every log line


def hms(sec):
    sec = int(sec); return f"{sec//3600}h{sec%3600//60:02d}m" if sec >= 3600 else f"{sec//60}m{sec%60:02d}s"


def wer_str():
    return " ".join(f"{n} {w:.2f}" for n, w in last_wer.items()) if last_wer else "-"


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
                w, sr = sf.read(os.path.join(P(d["raw_dir"]), orig[0]["path"]), dtype="float32")
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
            line = f"{hms(time.time()-t0)} ({step})  {name} wer {w:.2f}  whisper heard: {hyp}"
        except Exception as ex:
            line = f"{hms(time.time()-t0)} ({step})  {name} whisper failed: {ex}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
    model.train()
    hw = [w for n, w in ws.items() if n.startswith("heldout")]
    if hw:
        m = sum(hw) / len(hw); heldout_hist.append(m); tb.add_scalar("wer/heldout_mean", m, step)
        if m < min(heldout_hist[:-1], default=9e9):  # best checkpoint by held-out WER
            torch.save({"model": model.state_dict(), "step": step, "vocab": tok.vocab, "heldout_wer": m}, os.path.join(run, "best.pt"))


log = open(os.path.join(run, "progress.log"), "a")
t0, order = time.time(), []
if resume and step: t0 -= ck.get("elapsed", 0)  # budget clock continues across restarts
while step < steps:
    if not order:
        order = rows[:]; random.shuffle(order)
    rs, order = order[:batch], order[batch:]
    lr = t["lr"] * min(1.0, (step + 1) / t["warmup"]) * (0.5 ** (step // t["lr_halve_every"]))
    for g in opt.param_groups: g["lr"] = lr
    lf, ld = train_step(rs)
    step += 1
    if lf != lf or ld != ld:  # NaN guard: stop, keep the last good checkpoint on disk
        line = f"{hms(time.time()-t0)} ({step})  STOP: FAILED: loss is NaN"; print(line, flush=True); log.write(line + "\n"); log.flush(); break
    if step % t["log_every"] == 0 or smoke:
        line = f"{hms(time.time()-t0)} ({step})  wer {wer_str()}  audio loss {lf:.3f}  duration loss {ld:.3f}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        tb.add_scalar("train/fm", lf, step); tb.add_scalar("train/dp", ld, step); tb.add_scalar("train/lr", lr, step)
    if step % t["val_every"] == 0 or step == steps:
        vf, vd = validate()
        line = f"{hms(time.time()-t0)} ({step})  validation: audio loss {vf:.3f}  duration loss {vd:.3f}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        tb.add_scalar("val/fm", vf, step); tb.add_scalar("val/dp", vd, step)
    if step % t["sample_every"] == 0 or step == steps:
        sample(step)
    if step % t["ckpt_every"] == 0 or step == steps:
        save()
    stop = None
    if time.time() - t0 > t["max_minutes"] * 60: stop = f"max_minutes={t['max_minutes']}"
    W = t["stop_wer_window"]
    if len(heldout_hist) >= W and sum(heldout_hist[-W:]) / W < t["stop_wer"]: stop = f"heldout WER mean of last {W} < {t['stop_wer']}"
    if time.time() - t0 > t["fail_after_minutes"] * 60 and last_wer.get("train", 1.0) > t["fail_wer"]: stop = f"FAILED: train WER {last_wer.get('train', 1.0):.2f} > {t['fail_wer']} after {t['fail_after_minutes']} min"
    if stop:
        line = f"{hms(time.time()-t0)} ({step})  STOP: {stop}"; print(line, flush=True); log.write(line + "\n"); log.flush()
        save(); break
vf, vd = validate()
json.dump({"params": count_params(model), "dp_params": count_params(model.dp), "steps": step, "val_fm": vf, "val_dp_logl1": vd,
           "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0, "seconds": time.time() - t0,
           "batch": batch, "batch_expand": Ke}, open(os.path.join(run, "summary.json"), "w"), indent=1)
print("done", flush=True)
