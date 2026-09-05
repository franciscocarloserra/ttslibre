"""ONNX export of the four learned components (same split as Supertonic 3's onnx/):
style_encoder, text_encoder, vector_estimator, duration_predictor, plus latent_decoder.
Vocos (external, MIT) is not exported here. Usage: export.py [--run runs/ttl] [--out runs/onnx]"""
import os, sys
import torch, torch.nn as nn, onnx
from common import load_config, TTL, LatentAE, P

c = load_config()
args = sys.argv[1:]
run = args[args.index("--run") + 1] if "--run" in args else "runs/ttl"
out = P(args[args.index("--out") + 1] if "--out" in args else "runs/onnx")
os.makedirs(out, exist_ok=True)
ck = torch.load(P(os.path.join(run, "ttl.pt")), map_location="cpu")
m = TTL(c, len(ck["vocab"])).eval(); m.load_state_dict(ck["model"])
ae = LatentAE(c).eval(); ae.load_state_dict(torch.load(P("runs/ae/ae.pt"), map_location="cpu"))
C, H = m.in_dim, c["ttl"]["ref_hidden"]
for p_ in list(m.parameters()) + list(ae.parameters()):
    p_.requires_grad_(False)


class W(nn.Module):
    def __init__(self):
        super().__init__(); self.m, self.ae = m, ae


class StyleEnc(W):
    def forward(self, zref, rmask): return m.ref(zref, rmask > 0)

class TextEnc(W):
    def forward(self, ids, tmask, style): return m.text(ids, tmask > 0, style)

class VF(W):
    def forward(self, zt, mask, t, text, tmask, style): return m.velocity(zt, mask, t, text, tmask > 0, style)

class DP(W):
    def forward(self, ids, tmask, zref, rmask): return m.dp(ids, tmask > 0, zref, rmask > 0).exp()

class Dec(W):
    def forward(self, z): return ae.decode(z)


ids = torch.randint(1, len(ck["vocab"]), (1, 20)); tm = torch.ones(1, 20, dtype=torch.int64)
zref = torch.randn(1, C, 30); rm = torch.ones(1, 30, dtype=torch.int64)
style = torch.randn(1, c["ttl"]["n_style"], H); text = torch.randn(1, 20, c["ttl"]["text_hidden"])
zt = torch.randn(1, C, 40); mask = torch.ones(1, 1, 40); t = torch.tensor([0.5])
specs = [
    ("style_encoder", StyleEnc(), (zref, rm), ["zref", "rmask"], ["style"], {"zref": {2: "S"}, "rmask": {1: "S"}}),
    ("text_encoder", TextEnc(), (ids, tm, style), ["text_ids", "text_mask", "style"], ["text_emb"], {"text_ids": {1: "T"}, "text_mask": {1: "T"}, "text_emb": {1: "T"}}),
    ("vector_estimator", VF(), (zt, mask, t, text, tm, style), ["noisy_latent", "latent_mask", "t", "text_emb", "text_mask", "style"], ["velocity"],
     {"noisy_latent": {2: "L"}, "latent_mask": {2: "L"}, "text_emb": {1: "T"}, "text_mask": {1: "T"}, "velocity": {2: "L"}}),
    ("duration_predictor", DP(), (ids, tm, zref, rm), ["text_ids", "text_mask", "zref", "rmask"], ["frames"], {"text_ids": {1: "T"}, "text_mask": {1: "T"}, "zref": {2: "S"}, "rmask": {1: "S"}}),
    ("latent_decoder", Dec(), (torch.randn(1, c["latent"]["dim"], 120),), ["latent"], ["mel"], {"latent": {2: "N"}, "mel": {2: "N"}}),
]
for name, mod, inp, inames, onames, dyn in specs:
    path = os.path.join(out, name + ".onnx")
    with torch.no_grad():
        torch.onnx.export(mod, inp, path, input_names=inames, output_names=onames, dynamic_axes=dyn, opset_version=17, dynamo=False)
    onnx.checker.check_model(path)
    ref = mod(*inp)
    import onnxruntime as ort, numpy as np
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o = s.run(None, {n: x.numpy() for n, x in zip(inames, inp)})[0]
    print(f"{name}: {os.path.getsize(path)/2**20:.1f} MiB max_abs_diff={np.abs(o - ref.numpy()).max():.2e}", flush=True)
