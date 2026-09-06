"""Synthesize: text + reference wav -> wav. Euler flow-matching sampler with CFG, latent AE decode, Vocos.
Usage: synth.py "text" out.wav [--ref ref.wav] [--cpu] [--run runs/ttl]"""
import json, os, sys, time
import torch, soundfile as sf
from common import load_config, TTL, LatentAE, Tokenizer, Mel, compress, decompress, lengths_to_mask, P


class Synth:
    def __init__(self, c, run=None, ae_run=None, device=None):
        self.c, d, s = c, c["data"], c["synth"]
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        prep = P(d["prep_dir"])
        self.K, self.fps = c["latent"]["compress"], d["sample_rate"] / d["hop"] / c["latent"]["compress"]
        ck = torch.load(P(run or "runs/ttl/ttl.pt"), map_location=self.dev)
        self.tok = Tokenizer(ck["vocab"])
        self.model = TTL(c, len(self.tok.vocab)).to(self.dev).eval()
        self.model.load_state_dict(ck["model"])
        self.ae = LatentAE(c).to(self.dev).eval()
        self.ae.load_state_dict(torch.load(P(ae_run or "runs/ae/ae.pt"), map_location=self.dev))
        st = torch.load(os.path.join(prep, "latent_stats.pt"))
        self.mean, self.std = st["mean"].to(self.dev)[None, :, None], st["std"].to(self.dev)[None, :, None]
        self.mel = Mel(d).to(self.dev)
        from vocos import Vocos
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.dev).eval()

    @torch.no_grad()
    def style_from_wav(self, path):
        x, sr = sf.read(path, dtype="float32")
        x = torch.from_numpy(x if x.ndim == 1 else x.mean(1))[None].to(self.dev)
        if sr != self.c["data"]["sample_rate"]:
            import torchaudio
            x = torchaudio.functional.resample(x, sr, self.c["data"]["sample_rate"])
        z = (self.ae.encode(self.mel(x)) - self.mean) / self.std
        zref = compress(z, self.K)
        return zref, lengths_to_mask(torch.tensor([zref.shape[2]], device=self.dev))

    @torch.no_grad()
    def __call__(self, text, zref, rmask, steps=None, cfg=None, duration_scale=None):
        s = self.c["synth"]
        steps, cfg = steps or s["steps"], s["cfg"] if cfg is None else cfg
        ids = torch.tensor([self.tok.encode(text)], device=self.dev)
        tmask = torch.ones_like(ids, dtype=torch.bool)
        textemb, style = self.model.encode(ids, tmask, zref, rmask)
        n = int(round(self.model.dp(ids, tmask, zref, rmask).exp().item() * (duration_scale or s["duration_scale"])))
        n = max(n, 1)
        mask = torch.ones(1, 1, n, device=self.dev)
        ut, us = self.model.uncond(1, ids.shape[1], self.dev, textemb.dtype)
        z = torch.randn(1, self.model.in_dim, n, device=self.dev)
        for i in range(steps):
            t0, t1 = i / steps, (i + 1) / steps
            tt = torch.full((1,), t0, device=self.dev)
            v = self.model.velocity(z, mask, tt, textemb, tmask, style)
            if cfg != 1.0:
                vu = self.model.velocity(z, mask, tt, ut, tmask, us)
                v = vu + cfg * (v - vu)
            z = z + (t1 - t0) * v
        lat = decompress(z, self.K) * self.std + self.mean
        mel = self.ae.decode(lat)
        wav = self.vocos.decode(mel)
        return wav[0].cpu().numpy(), n / self.fps


if __name__ == "__main__":
    c = load_config()
    args = sys.argv[1:]
    cpu = "--cpu" in args
    ref = args[args.index("--ref") + 1] if "--ref" in args else P(c["synth"]["ref_clip"])
    run = args[args.index("--run") + 1] if "--run" in args else None
    text, out = args[0], args[1]
    if cpu:
        torch.set_num_threads(c["synth"]["cpu_threads"])
    S = Synth(c, run=run, device="cpu" if cpu else None)
    zref, rmask = S.style_from_wav(ref)
    t0 = time.time(); wav, dur = S(text, zref, rmask); el = time.time() - t0
    sf.write(out, wav, c["data"]["sample_rate"])
    print(json.dumps({"seconds": round(len(wav) / c["data"]["sample_rate"], 2), "gen_s": round(el, 3), "rtf": round(el / max(dur, 1e-6), 3), "device": S.dev}))
