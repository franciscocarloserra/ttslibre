"""Synthesize: text + reference wav -> wav. Euler flow-matching sampler with CFG, latent AE decode, Vocos.
Usage: synth.py "text" out.wav [--ref ref.wav | --ref voices/name.pt | --ref voices/name.style.pt] [--cpu] [--run runs/ttl]"""
import json, os, re, sys, time
import torch, soundfile as sf
from common import load_config, TTL, LatentAE, Tokenizer, Mel, compress, decompress, lengths_to_mask, P

EN = set("the and of to a in is it that for on with as at by this from or but not are be was were have has had will would can could should i you he she they we me my your our their if when what which who how why do does did done been being also there here just now only more most other some any all each every no yes very over under between about into than then so while because before after".split())
ES = set("el la los las de y que en un una uno es son por con para del al lo mi tu su sus este esta estos estas eso esto ese esa pero si no muy o como más ya hay ser estar tiene tengo tienes tenemos está están fue fueron era eran haber hacer cuando donde quien cual porque aunque entonces también sobre entre ante sin hasta desde hacia mientras despues antes asi solo todo toda todos todas cada otro otra nada nadie alguien algun alguna bien mal mejor peor me te se nos le les yo vos vosotros nosotros ellos ellas usted ustedes".split())


class Synth:
    def __init__(self, c, run=None, ae_run=None, device=None):
        self.c, d, s = c, c["data"], c["synth"]
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        prep = P(d["prep_dir"])
        self.K, self.fps = c["latent"]["compress"], d["sample_rate"] / d["hop"] / c["latent"]["compress"]
        ck = torch.load(P(run or "runs/ttl/ttl.pt"), map_location=self.dev)
        self.tok = Tokenizer(ck["vocab"])
        # default language tag: synth.lang of the experiment that owns the checkpoint (the panel may load checkpoints of other experiments)
        exp = os.path.abspath(P(run or "runs/ttl/ttl.pt")).split("/runs/")[0]; ec = os.path.join(exp, "config.json")
        self.lang = (json.load(open(ec))["synth"].get("lang", "") if os.path.exists(ec) else "") or s.get("lang", "")
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

    def style_from_voice(self, path):
        """Voicepack made by voice.py: saved compressed reference latents."""
        zref = torch.load(path, map_location=self.dev)["zref"].float()
        return zref, lengths_to_mask(torch.tensor([zref.shape[2]], device=self.dev))

    @torch.no_grad()
    def style_tensors(self, zref, rmask):
        """Voice as tensors for this checkpoint (see TTL.style). Editable: interpolate, average, add directions."""
        return {k: v.float() for k, v in self.model.style(zref, rmask).items()}

    def style_from_pack(self, path):
        """Style pack made by voice.py style/mix: tensors, no reference encoder at synthesis."""
        st = torch.load(path, map_location=self.dev)
        return {k: st[k].to(self.dev).float() for k in ("style", "dp")}

    def style(self, ref):
        """Any of: reference wav, voicepack (latents, checkpoint-independent), style pack (tensors, checkpoint-specific).
        Returns (zref, rmask) or a style dict; both are accepted by __call__."""
        if ref.endswith(".style.pt"): return self.style_from_pack(ref), None
        return self.style_from_voice(ref) if ref.endswith(".pt") else self.style_from_wav(ref)

    def detect(self, text):
        """es/en from text (same rule as know-how/local-tts tts_server.detect_lang): Spanish chars, apostrophe, stopword counts; English when undecided."""
        t = text.lower()
        if re.search(r"[ñáéíóúü¿¡]", t): return "es"
        if "'" in t or "\u2019" in t: return "en"
        w = re.findall(r"[a-z]+", t); en = sum(x in EN for x in w); es = sum(x in ES for x in w)
        return "es" if es > en else "en"  # tie or no clue: English (most users)

    @torch.no_grad()
    def tag(self, text, lang=None):
        """Wrap text in <lang>...</lang> when the checkpoint vocab has language tags (016+) and the text is not tagged yet; lang: given, else detected, else synth.lang."""
        lang = lang or self.detect(text) or getattr(self, "lang", "") or self.c["synth"].get("lang", "")
        return f"<{lang}>{text}</{lang}>" if lang and f"<{lang}>" in self.tok.idx and not text.lstrip().startswith("<") else text

    def __call__(self, text, zref, rmask, steps=None, cfg=None, duration_scale=None, lang=None):
        s = self.c["synth"]
        steps, cfg = steps or s["steps"], s["cfg"] if cfg is None else cfg
        text = self.tag(text, lang)
        ids = torch.tensor([self.tok.encode(text)], device=self.dev)
        tmask = torch.ones_like(ids, dtype=torch.bool)
        st = zref if isinstance(zref, dict) else self.style_tensors(zref, rmask)  # zref may be a style dict (style pack)
        textemb, style = self.model.encode(ids, tmask, style=st["style"])
        n = int(round(self.model.dp(ids, tmask, style=st["dp"]).exp().item() * (duration_scale or s["duration_scale"])))
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
    zref, rmask = S.style(ref)
    t0 = time.time(); wav, dur = S(text, zref, rmask); el = time.time() - t0
    sf.write(out, wav, c["data"]["sample_rate"])
    print(json.dumps({"seconds": round(len(wav) / c["data"]["sample_rate"], 2), "gen_s": round(el, 3), "rtf": round(el / max(dur, 1e-6), 3), "device": S.dev}))
