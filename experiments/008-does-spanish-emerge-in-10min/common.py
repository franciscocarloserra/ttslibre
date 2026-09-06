"""Shared pieces: config, mel, tokenizer, blocks, models. Architecture follows SupertonicTTS
(arXiv 2503.23108, Appendix A) and the Supertonic 3 tts.json layout: latent AE, text-to-latent
flow matching with character input and cross-attention alignment, utterance-level duration predictor."""
import json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    with open(path or os.path.join(HERE, "config.json")) as f:
        return json.load(f)


def P(rel):
    return rel if os.path.isabs(rel) else os.path.join(HERE, rel)


# ---------------------------------------------------------------- audio
class Mel(nn.Module):
    """Matches Vocos mel-24khz features: power=1, log with clamp."""
    def __init__(self, d):
        super().__init__()
        self.m = torchaudio.transforms.MelSpectrogram(sample_rate=d["sample_rate"], n_fft=d["n_fft"],
            hop_length=d["hop"], n_mels=d["n_mels"], f_min=d["mel_fmin"], f_max=d["mel_fmax"], center=True, power=1)
        self.clamp = d["log_clamp"]

    def forward(self, wav):  # (B,N) -> (B,n_mels,T)
        return torch.log(torch.clamp(self.m(wav), min=self.clamp))


# ---------------------------------------------------------------- text
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab  # list of chars, index 0 = pad/unknown
        self.idx = {c: i for i, c in enumerate(vocab)}

    @classmethod
    def build(cls, texts):
        chars = sorted({c for t in texts for c in t})
        return cls(["<pad>"] + chars)

    def encode(self, text):
        return [self.idx.get(c, 0) for c in text]

    def save(self, path):
        json.dump(self.vocab, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


# ---------------------------------------------------------------- blocks
class ConvNeXt(nn.Module):
    def __init__(self, dim, inter, k, dilation=1, causal=False):
        super().__init__()
        self.causal, self.pad = causal, (k - 1) * dilation
        self.dw = nn.Conv1d(dim, dim, k, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        self.pw1, self.pw2 = nn.Linear(dim, inter), nn.Linear(inter, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-2))

    def forward(self, x, mask=None):  # x (B,C,T), mask (B,1,T)
        r = x
        x = F.pad(x, (self.pad, 0) if self.causal else (self.pad // 2, self.pad - self.pad // 2))
        x = self.dw(x).transpose(1, 2)
        x = self.pw2(F.gelu(self.pw1(self.norm(x)))) * self.gamma
        x = r + x.transpose(1, 2)
        return x * mask if mask is not None else x


class ConvStack(nn.Module):
    def __init__(self, dim, inter, k, dilations, causal=False):
        super().__init__()
        self.blocks = nn.ModuleList(ConvNeXt(dim, inter, k, d, causal) for d in dilations)

    def forward(self, x, mask=None):
        for b in self.blocks:
            x = b(x, mask)
        return x


def rope(x, base=10000):  # x (B,H,T,D) rotate pairs
    B, H, T, D = x.shape
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    freqs = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = pos[:, None] * freqs[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1)
    return out.flatten(-2).to(x.dtype)


class Attention(nn.Module):
    """Multi-head attention. q from x, k/v from ctx (or separate key/value tensors). Optional rotary on self-attn."""
    def __init__(self, dim, heads, ctx_dim=None, rotary=False):
        super().__init__()
        ctx_dim = ctx_dim or dim
        self.h, self.rotary = heads, rotary
        self.q, self.k, self.v, self.o = nn.Linear(dim, dim), nn.Linear(ctx_dim, dim), nn.Linear(ctx_dim, dim), nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key, value=None, kmask=None):  # x (B,T,C); key (B,S,Ck); kmask (B,S) bool True=valid
        value = key if value is None else value
        B, T, _ = x.shape
        q = self.q(self.norm(x)).view(B, T, self.h, -1).transpose(1, 2)
        k = self.k(key).view(B, key.shape[1], self.h, -1).transpose(1, 2)
        v = self.v(value).view(B, value.shape[1], self.h, -1).transpose(1, 2)
        if self.rotary:
            q, k = rope(q), rope(k)
        am = kmask[:, None, None, :] if kmask is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return x + self.o(y.transpose(1, 2).reshape(B, T, -1))


class FFN(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.norm, self.f1, self.f2 = nn.LayerNorm(dim), nn.Linear(dim, inter), nn.Linear(inter, dim)

    def forward(self, x):
        return x + self.f2(F.gelu(self.f1(self.norm(x))))


class SelfAttnEncoder(nn.Module):
    def __init__(self, dim, inter, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, rotary=True), FFN(dim, inter)]) for _ in range(layers)])

    def forward(self, x, mask):  # x (B,T,C), mask (B,T) bool
        for a, f in self.layers:
            x = f(a(x, x, kmask=mask))
        return x * mask[..., None]


# ---------------------------------------------------------------- latent autoencoder (mel <-> latent)
class LatentAE(nn.Module):
    """Vocos-style ConvNeXt encoder to a low-dim latent, causal ConvNeXt decoder back to mel.
    Waveform synthesis is delegated to an external mel vocoder (Vocos) in this prototype."""
    def __init__(self, c):
        super().__init__()
        a, L, nm = c["ae"], c["latent"]["dim"], c["data"]["n_mels"]
        H, I, k = a["hidden"], a["intermediate"], a["kernel"]
        self.enc_in = nn.Conv1d(nm, H, k, padding=k // 2)
        self.enc = ConvStack(H, I, k, [1] * a["enc_layers"])
        self.enc_out = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, L))
        self.dec_in = nn.Conv1d(L, H, k, padding=k // 2)
        self.dec = ConvStack(H, I, k, [1, 2, 4] * (a["dec_layers"] // 3) + [1] * (a["dec_layers"] % 3), causal=True)
        self.dec_out = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, I), nn.PReLU(), nn.Linear(I, nm))

    def encode(self, mel):  # (B,nm,T) -> (B,L,T)
        x = self.enc(self.enc_in(mel))
        return self.enc_out(x.transpose(1, 2)).transpose(1, 2)

    def decode(self, z):  # (B,L,T) -> (B,nm,T)
        x = self.dec(self.dec_in(z))
        return self.dec_out(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, mel):
        return self.decode(self.encode(mel))


def compress(z, K):  # (B,L,T) -> (B,K*L,T/K), pads T to a multiple of K
    B, L, T = z.shape
    pad = (-T) % K
    z = F.pad(z, (0, pad))
    return z.view(B, L, -1, K).permute(0, 3, 1, 2).reshape(B, K * L, -1)


def decompress(zc, K):  # inverse of compress
    B, KL, Tc = zc.shape
    L = KL // K
    return zc.view(B, K, L, Tc).permute(0, 2, 3, 1).reshape(B, L, Tc * K)


# ---------------------------------------------------------------- text-to-latent
class RefEncoder(nn.Module):
    """Compressed reference latents -> n_style style vectors (timbre-token style, NANSY++)."""
    def __init__(self, in_dim, H, I, k, layers, n_style, heads=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, H)
        self.conv = ConvStack(H, I, k, [1] * layers)
        self.queries = nn.Parameter(torch.randn(n_style, H) * 0.02)
        self.attn1, self.attn2 = Attention(H, heads), Attention(H, heads)

    def forward(self, zref, mask):  # zref (B,C,S), mask (B,S) bool -> (B,n_style,H)
        x = self.conv(self.proj(zref.transpose(1, 2)).transpose(1, 2), mask[:, None]).transpose(1, 2)
        q = self.queries[None].expand(x.shape[0], -1, -1)
        s = self.attn1(q, x, kmask=mask)
        return self.attn2(s, x, kmask=mask)


class TextEncoder(nn.Module):
    def __init__(self, vocab, t, n_style):
        super().__init__()
        H, k = t["text_hidden"], t["kernel"]
        self.emb = nn.Embedding(vocab, t["char_dim"], padding_idx=0)
        self.proj = nn.Linear(t["char_dim"], H) if t["char_dim"] != H else nn.Identity()
        self.conv = ConvStack(H, t["text_intermediate"], k, t["text_conv_dilations"][: t["text_conv_layers"]])
        self.attn = SelfAttnEncoder(H, t["text_intermediate"], t["text_attn_heads"], t["text_attn_layers"])
        self.ref_keys = nn.Parameter(torch.randn(n_style, H) * 0.02)  # shared "reference key" (paper A.2.2)
        self.x1, self.x2 = Attention(H, 2, ctx_dim=t["ref_hidden"]), Attention(H, 2, ctx_dim=t["ref_hidden"])

    def forward(self, ids, tmask, style):  # ids (B,T), tmask (B,T) bool, style (B,n_style,Hr) -> (B,T,H)
        x = self.proj(self.emb(ids)).transpose(1, 2)
        x = self.conv(x, tmask[:, None]).transpose(1, 2)
        x = self.attn(x, tmask)
        x = self.x1(x, self.ref_keys[None].expand(x.shape[0], -1, -1).to(style.dtype), style)
        x = self.x2(x, style, style)
        return x * tmask[..., None]


def time_embedding(t, dim):  # Grad-TTS style sinusoidal, t (B,) in [0,1]
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    a = t[:, None].float() * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class VFBlock(nn.Module):
    def __init__(self, t, text_dim, ref_dim):
        super().__init__()
        H, I, k = t["vf_hidden"], t["vf_intermediate"], t["kernel"]
        self.time = nn.Linear(t["time_dim"], H)
        self.dil = ConvStack(H, I, k, t["vf_dilations"])
        self.text = Attention(H, t["vf_heads"], ctx_dim=text_dim)
        self.ref = Attention(H, t["vf_heads"], ctx_dim=ref_dim)
        self.plain = ConvStack(H, I, k, [1] * t["vf_plain_layers"])

    def forward(self, x, mask, temb, text, tmask, ref_keys, style):  # x (B,H,T)
        x = (x + self.time(temb)[:, :, None]) * mask
        x = self.dil(x, mask).transpose(1, 2)
        x = self.text(x, text, kmask=tmask)
        x = self.ref(x, ref_keys, style)
        return self.plain(x.transpose(1, 2), mask)


class VectorField(nn.Module):
    def __init__(self, c, in_dim, text_dim, ref_dim):
        super().__init__()
        t = c["ttl"]
        H, I, k, self.time_dim = t["vf_hidden"], t["vf_intermediate"], t["kernel"], t["time_dim"]
        self.proj_in = nn.Linear(in_dim, H)
        self.tmlp = nn.Sequential(nn.Linear(t["time_dim"], t["time_dim"] * 4), nn.SiLU(), nn.Linear(t["time_dim"] * 4, t["time_dim"]))
        self.blocks = nn.ModuleList(VFBlock(t, text_dim, ref_dim) for _ in range(t["vf_blocks"]))
        self.last = ConvStack(H, I, k, [1] * t["vf_last_layers"])
        self.proj_out = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, in_dim))

    def forward(self, zt, mask, t, text, tmask, ref_keys, style):  # zt (B,C,T), mask (B,1,T)
        temb = self.tmlp(time_embedding(t, self.time_dim).to(zt.dtype))
        x = self.proj_in(zt.transpose(1, 2)).transpose(1, 2) * mask
        for b in self.blocks:
            x = b(x, mask, temb, text, tmask, ref_keys, style)
        x = self.last(x, mask)
        return self.proj_out(x.transpose(1, 2)).transpose(1, 2) * mask


class DurationPredictor(nn.Module):
    """Utterance-level: text + reference -> log(total compressed frames)."""
    def __init__(self, c, vocab, in_dim):
        super().__init__()
        d, k = c["dp"], c["ttl"]["kernel"]
        H = d["char_dim"]
        self.emb = nn.Embedding(vocab, H, padding_idx=0)
        self.conv = ConvStack(H, d["intermediate"], k, [1] * d["conv_layers"])
        self.utt = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.attn = SelfAttnEncoder(H, d["intermediate"], d["attn_heads"], d["attn_layers"])
        self.text_out = nn.Linear(H, H)
        self.ref = RefEncoder(in_dim, H, d["intermediate"], k, d["ref_layers"], d["n_style"])
        self.ref_out = nn.Linear(d["n_style"] * H, H)
        self.head = nn.Sequential(nn.Linear(2 * H, d["hidden"]), nn.PReLU(), nn.Linear(d["hidden"], 1))

    def forward(self, ids, tmask, zref, rmask):
        x = self.conv(self.emb(ids).transpose(1, 2), tmask[:, None]).transpose(1, 2)
        x = torch.cat([self.utt.expand(x.shape[0], -1, -1), x], 1)
        m = torch.cat([torch.ones_like(tmask[:, :1]), tmask], 1)
        te = self.text_out(self.attn(x, m)[:, 0])
        re = self.ref_out(self.ref(zref, rmask).flatten(1))
        return self.head(torch.cat([te, re], -1)).squeeze(-1)


class TTL(nn.Module):
    """Text-to-latent: ref encoder + text encoder + vector field (+ duration predictor, trained jointly)."""
    def __init__(self, c, vocab):
        super().__init__()
        t, L, K = c["ttl"], c["latent"]["dim"], c["latent"]["compress"]
        self.in_dim, self.K = L * K, K
        self.ref = RefEncoder(self.in_dim, t["ref_hidden"], t["ref_intermediate"], t["kernel"], t["ref_conv_layers"], t["n_style"])
        self.text = TextEncoder(vocab, t, t["n_style"])
        self.vf = VectorField(c, self.in_dim, t["text_hidden"], t["ref_hidden"])
        self.dp = DurationPredictor(c, vocab, self.in_dim)
        self.uncond_style = nn.Parameter(torch.zeros(t["n_style"], t["ref_hidden"]))
        self.uncond_text = nn.Parameter(torch.zeros(1, t["text_hidden"]))
        self.sigma_min = t["sigma_min"]

    def encode(self, ids, tmask, zref, rmask):
        style = self.ref(zref, rmask)
        text = self.text(ids, tmask, style)
        return text, style

    def uncond(self, B, T, device, dtype):
        text = self.uncond_text[None].expand(B, T, -1).to(dtype)
        style = self.uncond_style[None].expand(B, -1, -1).to(dtype)
        return text, style

    def velocity(self, zt, mask, t, text, tmask, style):
        keys = self.text.ref_keys[None].expand(zt.shape[0], -1, -1).to(style.dtype)
        return self.vf(zt, mask, t, text, tmask, keys, style)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def lengths_to_mask(lengths, T=None):
    T = T or int(lengths.max())
    return torch.arange(T, device=lengths.device)[None] < lengths[:, None]
