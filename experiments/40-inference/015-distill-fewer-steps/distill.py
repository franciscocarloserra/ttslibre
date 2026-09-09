"""Stage 1: progressive distillation of the Euler sampler (Salimans & Ho 2022 on flow matching).
Each round halves the step count: teacher takes 2 Euler steps of size 1/N (with cfg), student one step of 2/N with cfg 1.
Loss = masked MSE on the velocity implied by the teacher's 2-step move. Student initialized from its teacher; the previous
round's student becomes the next teacher (cfg 1.0). Probes: sample_heldout_n held-out sentences x draws at probe_steps / cfg 1,
WER via Whisper (eval.whisper_url; unreachable Whisper is logged, not fatal). Success stop in the last round when the probe
mean WER is within success.max_wer_delta of the teacher's baseline (runs/baseline/grid.jsonl at teacher_steps/teacher_cfg)
for success.window consecutive probes. Writes runs/distill/r<N>to<M>/{student.pt,best.pt,progress.log,probes/}.
Usage: distill.py [--smoke] [--resume-rounds]   (smoke: distill.smoke knobs; resume: skip rounds whose student.pt exists)"""
import copy, json, os, random, sys, time
import torch, soundfile as sf
sys.path.insert(0, "../../20-scaling/014-spanish-8h-openslr61")
from eval_val import wer, whisper
from common import load_config, TTL, Tokenizer, compress, lengths_to_mask, P
from synth import Synth

cfg = json.load(open("config.json")); D = cfg["distill"]; smoke = "--smoke" in sys.argv
if smoke: D.update(cfg["distill"]["smoke"])
c = json.load(open(cfg["base_config"])); c["synth"]["ref_clip"] = cfg["ref_clip"]
t, d, K = c["ttl"], c["data"], c["latent"]["compress"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.bfloat16 if t["amp"] and dev == "cuda" else torch.float32
prep = P(d["prep_dir"]); fps = d["sample_rate"] / d["hop"] / K
tok = Tokenizer.load(os.path.join(prep, "vocab.json"))
st = torch.load(os.path.join(prep, "latent_stats.pt")); mean, std = st["mean"].to(dev)[None, :, None], st["std"].to(dev)[None, :, None]
rows = [json.loads(l) for l in open(os.path.join(prep, "train.jsonl"))]
val = [json.loads(l) for l in open(os.path.join(prep, "val.jsonl"))]
b = cfg["baseline"]; random.Random(b["seed"]).shuffle(val); probe_set = val[: b["sentences"]][: D["sample_heldout_n"]]
S = Synth(c, run=os.path.abspath(cfg["teacher"]))  # sampler; .model swapped to the student for probes
zref, rmask = S.style_from_wav(P(cfg["ref_clip"]))
teacher = S.model; sc = D["success"]


def teacher_ref_wer():
    """Baseline mean WER of the original teacher on the probe sentences at success.teacher_steps/teacher_cfg (None if stage 0 is missing)."""
    p = os.path.join(b["out"], "grid.jsonl")
    if not os.path.exists(p): return None
    ids = {r["id"] for r in probe_set}
    ws = [r["wer"] for r in map(json.loads, open(p)) if r["steps"] == sc["teacher_steps"] and r["cfg"] == sc["teacher_cfg"] and r["id"] in ids and r["wer"] is not None]
    return sum(ws) / len(ws) if ws else None


ref_wer = teacher_ref_wer()


class Data:
    """All training latents (normalized, compressed) and token ids padded into GPU tensors once; batches are gathers, no per-sample Python."""
    def __init__(self, rows):
        zs = [compress((torch.load(os.path.join(prep, "latents", r["id"] + ".pt")).float()[None].to(dev) - mean) / std, K)[0] for r in rows]
        self.lens = torch.tensor([z.shape[1] for z in zs], device=dev); self.C = zs[0].shape[0]
        self.Z = torch.zeros(len(zs), self.C, int(self.lens.max()), device=dev)
        for i, z in enumerate(zs): self.Z[i, :, :z.shape[1]] = z
        ids = [torch.tensor(tok.encode(r["text"])) for r in rows]
        self.tl = torch.tensor([len(x) for x in ids], device=dev); self.ids = torch.zeros(len(ids), int(self.tl.max()), dtype=torch.long, device=dev)
        for i, x in enumerate(ids): self.ids[i, :len(x)] = x
        self.T = int(t["max_seconds"] * fps); self.S = int(min(t["ref_max_seconds"], t["max_seconds"]) * fps)

    def batch(self, idx):
        """ids, tmask, z1 (B,C,T), mask (B,1,T), zref (B,C,S), rmask. Random crop to T and a random reference window inside the crop."""
        B = len(idx); lens = self.lens[idx]; n = lens.clamp(max=self.T); T = int(n.max())
        start = (torch.rand(B, device=dev) * (lens - n + 1).float()).long()
        pos = start[:, None] + torch.arange(T, device=dev)[None]
        z1 = torch.gather(self.Z[idx], 2, pos.clamp(max=self.Z.shape[2] - 1)[:, None].expand(-1, self.C, -1))
        mask = (torch.arange(T, device=dev)[None] < n[:, None])[:, None].float(); z1 = z1 * mask
        rl = torch.minimum((torch.rand(B, device=dev) * (t["ref_max_seconds"] - t["ref_min_seconds"]) + t["ref_min_seconds"]) * fps, t["ref_max_fraction"] * n.float()).clamp(min=1, max=self.S).long()
        r0 = (torch.rand(B, device=dev) * (n - rl + 1).float()).long()
        rpos = r0[:, None] + torch.arange(self.S, device=dev)[None]
        zref = torch.gather(z1, 2, rpos.clamp(max=T - 1)[:, None].expand(-1, self.C, -1)) * (torch.arange(self.S, device=dev)[None] < rl[:, None])[:, None].float()
        tl = self.tl[idx]; L = int(tl.max())
        return self.ids[idx, :L], lengths_to_mask(tl, L), z1, mask, zref, lengths_to_mask(rl, self.S)


def euler(model, z, mask, tt, text, tmask, style, g, dt):
    v = model.velocity(z * mask, mask, tt, text, tmask, style)
    if g != 1.0:
        ut, us = model.uncond(z.shape[0], text.shape[1], dev, text.dtype)
        vu = model.velocity(z * mask, mask, tt, ut, tmask, us); v = vu + g * (v - vu)
    return z + dt * v


def hms(sec): sec = int(sec); return f"{sec//3600}h{sec%3600//60:02d}m" if sec >= 3600 else f"{sec//60}m{sec%60:02d}s"


def run_round(teacher, N, M, tcfg, out):
    os.makedirs(os.path.join(out, "probes"), exist_ok=True); log = open(os.path.join(out, "progress.log"), "a")
    student = copy.deepcopy(teacher).train()
    for p in student.dp.parameters(): p.requires_grad_(False)  # durations are not distilled
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=D["lr"])
    teacher.eval(); step, t0, kept, hist, best = 0, time.time(), 0, [], 9e9; order = torch.zeros(0, dtype=torch.long, device=dev)
    Mstep = max(1, M)

    def say(s):
        line = f"{hms(time.time()-t0)} ({step})  {s}"; print(line, flush=True); log.write(line + "\n"); log.flush()

    def probe():
        nonlocal best
        student.eval(); S.model = student; ws = []
        for r in probe_set:
            for k in range(b["draws"]):
                wav, _ = S(r["text"], zref, rmask, steps=D["probe_steps"], cfg=sc["student_cfg"])
                p = os.path.join(out, "probes", f"step{step:06d}_{r['id']}_{k}.wav"); sf.write(p, wav, d["sample_rate"])
                try: hyp = whisper(p); w = wer(r["text"], hyp); ws.append(w); say(f"probe {r['id']} draw {k} wer {w:.2f}  whisper heard: {hyp}")
                except Exception as ex: say(f"probe {r['id']} draw {k} whisper failed: {ex}")
        student.train(); S.model = teacher
        if not ws: return None
        m = sum(ws) / len(ws); hist.append(m); say(f"PROBE mean wer {m:.3f} ({D['probe_steps']} steps, cfg {sc['student_cfg']}); teacher ref {ref_wer}")
        if m < best: best = m; torch.save({"model": student.state_dict(), "step": step, "vocab": tok.vocab, "probe_wer": m, "steps": M}, os.path.join(out, "best.pt"))
        return m

    global data
    if data is None:
        t1 = time.time(); data = Data(rows); say(f"preloaded {len(rows)} latents+tokens to {dev} in {time.time()-t1:.0f}s ({data.Z.numel()*4/2**20:.0f} MB)")
    def train_step(idx):
        ids, tmask, z1, mask, zr, rm = data.batch(idx); B = z1.shape[0]
        i = torch.randint(0, M, (B,), device=dev); tt = i.float() * (2.0 / N)  # coarse grid t0 = 2i/N
        z0 = torch.randn_like(z1); zt = ((1 - tt)[:, None, None] * z0 + tt[:, None, None] * z1) * mask
        with torch.autocast(dev, dtype=amp, enabled=amp != torch.float32):
            with torch.no_grad():
                text, style = teacher.encode(ids, tmask, zr, rm)
                za = euler(teacher, zt, mask, tt, text, tmask, style, tcfg, 1.0 / N)
                zb = euler(teacher, za, mask, tt + 1.0 / N, text, tmask, style, tcfg, 1.0 / N)
                target = (zb - zt) / (2.0 / N)
            stext, sstyle = student.encode(ids, tmask, zr, rm)
            v = student.velocity(zt, mask, tt, stext, tmask, sstyle)
            loss = (((v - target) ** 2) * mask).sum() / (mask.sum() * z1.shape[1])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), t["grad_clip"]); opt.step()
        return loss

    def calibrate():
        """Pick the batch: largest of calib_batches that fits under calib_vram_fraction of VRAM with the best samples/s (one real step each)."""
        total = torch.cuda.get_device_properties(0).total_memory if dev == "cuda" else 0; best_b, best_r = None, 0
        for bsz in D["calib_batches"]:
            try:
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                train_step(torch.randperm(len(rows), device=dev)[:bsz]); torch.cuda.synchronize(); t1 = time.time()
                for _ in range(D["calib_steps"]): train_step(torch.randperm(len(rows), device=dev)[:bsz])
                torch.cuda.synchronize(); rate = bsz * D["calib_steps"] / (time.time() - t1); mem = torch.cuda.max_memory_allocated()
            except torch.OutOfMemoryError:
                say(f"calib batch {bsz}: OOM"); torch.cuda.empty_cache(); break
            say(f"calib batch {bsz}: {rate:.0f} samples/s, peak {mem/2**20:.0f} MiB of {total/2**20:.0f}")
            if mem > D["calib_vram_fraction"] * total: break
            if rate > best_r * (1 + D["calib_min_gain"]): best_b, best_r = bsz, rate
        return best_b

    if D["batch"] == "auto":
        D["batch"] = calibrate() or D["calib_batches"][0]; student.load_state_dict(teacher.state_dict()); opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=D["lr"])
    say(f"round {N}->{M} teacher_cfg {tcfg} train={len(rows)} batch={D['batch']} lr={D['lr']} budget={D['max_minutes_per_round']}min")
    probe()
    while True:
        if len(order) < D["batch"]: order = torch.randperm(len(rows), device=dev)
        idx, order = order[:D["batch"]], order[D["batch"]:]
        loss = train_step(idx); step += 1
        if loss != loss: say("STOP: FAILED: loss is NaN"); break
        if step % D["log_every"] == 0: say(f"loss {loss.item():.4f}  best probe {best:.3f}")
        stop = None
        if step % D["probe_every"] == 0:
            m = probe(); W = sc["window"]
            if ref_wer is not None and M == sc["student_steps"] and len(hist) >= W and all(h <= ref_wer + sc["max_wer_delta"] for h in hist[-W:]):
                stop = f"SUCCESS: last {W} probes <= teacher {ref_wer:.3f} + {sc['max_wer_delta']}"
        q = int((time.time() - t0) / (D["max_minutes_per_round"] * 60 / D["keep_checkpoints"]))
        if q > kept and q <= D["keep_checkpoints"]:
            kept = q; torch.save({"model": student.state_dict(), "step": step, "vocab": tok.vocab, "steps": M}, os.path.join(out, f"student_{hms(time.time()-t0)}_step{step}.pt"))
        if time.time() - t0 > D["max_minutes_per_round"] * 60: stop = f"max_minutes_per_round={D['max_minutes_per_round']}"
        if D.get("max_steps") and step >= D["max_steps"]: stop = f"max_steps={D['max_steps']}"
        if stop: say(f"STOP: {stop}"); break
    torch.save({"model": student.state_dict(), "step": step, "vocab": tok.vocab, "steps": M, "elapsed": time.time() - t0}, os.path.join(out, "student.pt"))
    json.dump({"round": [N, M], "steps": step, "seconds": time.time() - t0, "best_probe_wer": best if best < 9e9 else None, "teacher_ref_wer": ref_wer, "probe_hist": hist,
               "batch": D["batch"], "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0, "stop": stop}, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return student.eval(), stop and stop.startswith("SUCCESS")


data = None; resume = "--resume-rounds" in sys.argv
for j, (N, M) in enumerate(D["rounds"]):
    out = os.path.join(D["out"], f"r{N}to{M}")
    if resume and os.path.exists(os.path.join(out, "student.pt")):
        ck = torch.load(os.path.join(out, "student.pt"), map_location=dev); teacher.load_state_dict(ck["model"]); print(f"round {N}->{M} skipped, teacher <- {out}/student.pt", flush=True); continue
    teacher, ok = run_round(teacher, N, M, D["teacher_cfg"] if j == 0 else 1.0, out)
    print(f"round {N}->{M} done, success={bool(ok)}", flush=True)
    if ok: break
print("done", flush=True)
