"""Small GPT-2-style decoder: Bitsy Vulkan vs PyTorch CUDA numerics + speed.

Per-head projections and a block-causal mask let attention use only the 2D ops
already implemented. Weights are created from one NumPy RNG stream and copied
into both frameworks, so logits/gradients are directly comparable.

Dataclass config, no argparse.
"""

from __future__ import annotations

import json, statistics, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "benchmarks")]
from common.clock import gpu_clock_mhz  # noqa: E402
import bitsy as bt  # noqa: E402
from bitsy import nn, optim  # noqa: E402
BTensor = bt.autograd.Tensor

@dataclass(frozen=True, slots=True)
class GPTConfig:
  vocab:int = 1024
  d_model:int = 192
  n_layer:int = 4
  n_head:int = 4
  d_ff:int = 768
  block:int = 32
  batch:int = 2
  seq:int = 8
  lr:float = 1e-3
  iters:int = 10
  warmup:int = 2
  result_path:Path = ROOT / "benchmarks" / "results" / "gpt2_tiny.json"

CONFIG = GPTConfig()

def make_weights(cfg:GPTConfig):
  rng = np.random.default_rng(1234)
  w = {}
  w["embed"] = (rng.standard_normal((cfg.vocab, cfg.d_model)) * 0.02).astype(np.float32)
  w["head"] = (rng.standard_normal((cfg.d_model, cfg.vocab)) * 0.02).astype(np.float32)
  for l in range(cfg.n_layer):
    p = f"L{l}."
    for h in range(cfg.n_head):
      w[p + f"q{h}"] = (rng.standard_normal((cfg.d_model, cfg.d_model // cfg.n_head)) * 0.02).astype(np.float32)
      w[p + f"k{h}"] = (rng.standard_normal((cfg.d_model, cfg.d_model // cfg.n_head)) * 0.02).astype(np.float32)
      w[p + f"v{h}"] = (rng.standard_normal((cfg.d_model, cfg.d_model // cfg.n_head)) * 0.02).astype(np.float32)
      w[p + f"o{h}"] = (rng.standard_normal((cfg.d_model // cfg.n_head, cfg.d_model)) * 0.02).astype(np.float32)
    w[p + "c1w"] = (rng.standard_normal((cfg.d_model, cfg.d_ff)) * 0.02).astype(np.float32)
    w[p + "c2w"] = (rng.standard_normal((cfg.d_ff, cfg.d_model)) * 0.02).astype(np.float32)
  return w

def build_bitsy(cfg:GPTConfig, w, dev):
  emb = nn.Embedding(cfg.vocab, cfg.d_model, dev)
  emb.W = nn.Parameter(BTensor.from_numpy(w["embed"], dev))
  layers = []
  for l in range(cfg.n_layer):
    p = f"L{l}."
    ln1 = nn.LayerNorm(cfg.d_model, device=dev)
    ln2 = nn.LayerNorm(cfg.d_model, device=dev)
    q = [nn.Parameter(BTensor.from_numpy(w[p + f"q{h}"], dev)) for h in range(cfg.n_head)]
    k = [nn.Parameter(BTensor.from_numpy(w[p + f"k{h}"], dev)) for h in range(cfg.n_head)]
    v = [nn.Parameter(BTensor.from_numpy(w[p + f"v{h}"], dev)) for h in range(cfg.n_head)]
    o = [nn.Parameter(BTensor.from_numpy(w[p + f"o{h}"], dev)) for h in range(cfg.n_head)]
    c1 = nn.Linear(cfg.d_model, cfg.d_ff, np.random.default_rng(0), dev)
    c2 = nn.Linear(cfg.d_ff, cfg.d_model, np.random.default_rng(0), dev)
    c1.W = nn.Parameter(BTensor.from_numpy(w[p + "c1w"], dev))
    c2.W = nn.Parameter(BTensor.from_numpy(w[p + "c2w"], dev))
    layers.append((ln1, ln2, q, k, v, o, c1, c2))
  lnf = nn.LayerNorm(cfg.d_model, device=dev)
  head = nn.Linear(cfg.d_model, cfg.vocab, np.random.default_rng(0), dev)
  head.W = nn.Parameter(BTensor.from_numpy(w["head"], dev))
  return emb, layers, lnf, head, dev

def b_forward(cfg, model, idx_np):
  emb, layers, lnf, head, dev = model
  x = emb(idx_np)
  B, T, C = x.shape
  flat_n = B * T
  mask = np.zeros((flat_n, flat_n), np.float32)
  for i in range(flat_n):
    bi, ti = divmod(i, cfg.seq)
    for j in range(flat_n):
      bj, tj = divmod(j, cfg.seq)
      if bi != bj or tj > ti:
        mask[i, j] = -1e9
  mask_t = BTensor.from_numpy(mask, dev)
  for ln1, ln2, q, k, v, o, c1, c2 in layers:
    a = ln1(x)
    af = a.reshape(flat_n, C)
    y = None
    for h in range(cfg.n_head):
      qh, kh, vh = af @ q[h], af @ k[h], af @ v[h]
      scores = qh @ kh.transpose()
      att = nn.softmax(scores + mask_t)
      part = att @ vh @ o[h]
      y = part if y is None else y + part
    x = x + y.reshape(B, T, C)
    b = ln2(x).reshape(flat_n, C)
    x = x + c2(nn.gelu(c1(b))).reshape(B, T, C)
  x = lnf(x)
  xf = x.reshape(flat_n, C)
  return head(xf).reshape(B, T, cfg.vocab)

def build_torch(cfg:GPTConfig, w, device):
  class Attn(torch.nn.Module):
    def __init__(self):
      super().__init__()
      head_dim = cfg.d_model // cfg.n_head
      self.q = torch.nn.ModuleList([torch.nn.Linear(cfg.d_model, head_dim, bias=False) for _ in range(cfg.n_head)])
      self.k = torch.nn.ModuleList([torch.nn.Linear(cfg.d_model, head_dim, bias=False) for _ in range(cfg.n_head)])
      self.v = torch.nn.ModuleList([torch.nn.Linear(cfg.d_model, head_dim, bias=False) for _ in range(cfg.n_head)])
      self.o = torch.nn.ModuleList([torch.nn.Linear(head_dim, cfg.d_model, bias=False) for _ in range(cfg.n_head)])
    def forward(self, x, mask):
      flat = x.reshape(-1, cfg.d_model)
      y = None
      for h in range(cfg.n_head):
        qh, kh, vh = self.q[h](flat), self.k[h](flat), self.v[h](flat)
        scores = qh @ kh.T
        probs = torch.softmax(scores + mask, dim=-1)
        part = self.o[h](probs @ vh)
        y = part if y is None else y + part
      return y.reshape(x.shape)
  class Block(torch.nn.Module):
    def __init__(self, p):
      super().__init__()
      self.ln1 = torch.nn.LayerNorm(cfg.d_model, eps=1e-5)
      self.ln2 = torch.nn.LayerNorm(cfg.d_model, eps=1e-5)
      self.attn = Attn()
      self.c1 = torch.nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
      self.c2 = torch.nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
      with torch.no_grad():
        for h in range(cfg.n_head):
          self.attn.q[h].weight.copy_(torch.from_numpy(w[p + f"q{h}"].T.copy()))
          self.attn.k[h].weight.copy_(torch.from_numpy(w[p + f"k{h}"].T.copy()))
          self.attn.v[h].weight.copy_(torch.from_numpy(w[p + f"v{h}"].T.copy()))
          self.attn.o[h].weight.copy_(torch.from_numpy(w[p + f"o{h}"].T.copy()))
        self.c1.weight.copy_(torch.from_numpy(w[p + "c1w"].T.copy()))
        self.c2.weight.copy_(torch.from_numpy(w[p + "c2w"].T.copy()))
    def forward(self, x, mask):
      a = x + self.attn(self.ln1(x), mask)
      return a + self.c2(torch.nn.functional.gelu(self.c1(self.ln2(a))))
  class GPT(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.embed = torch.nn.Embedding(cfg.vocab, cfg.d_model)
      self.blocks = torch.nn.ModuleList([Block(f"L{l}.") for l in range(cfg.n_layer)])
      self.lnf = torch.nn.LayerNorm(cfg.d_model, eps=1e-5)
      self.head = torch.nn.Linear(cfg.d_model, cfg.vocab, bias=False)
      with torch.no_grad():
        self.embed.weight.copy_(torch.from_numpy(w["embed"]))
        self.head.weight.copy_(torch.from_numpy(w["head"].T.copy()))
    def forward(self, idx):
      mask = torch.zeros((idx.shape[0] * cfg.seq, idx.shape[0] * cfg.seq), device=idx.device)
      n = idx.shape[0] * cfg.seq
      for i in range(n):
        bi, ti = divmod(i, cfg.seq)
        for j in range(n):
          bj, tj = divmod(j, cfg.seq)
          if bi != bj or tj > ti: mask[i, j] = -1e9
      x = self.embed(idx)
      for blk in self.blocks: x = blk(x, mask)
      return self.head(self.lnf(x).reshape(-1, cfg.d_model)).reshape(x.shape[0], x.shape[1], cfg.vocab)
  return GPT().to(device)

def run(config:GPTConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  cfg = config
  w = make_weights(cfg)
  n_params = sum(v.size for v in w.values())
  rng = np.random.default_rng(9)
  idx_np = rng.integers(0, cfg.vocab, (cfg.batch, cfg.seq), dtype=np.int64)
  dev = "vulkan"
  bm = build_bitsy(cfg, w, dev)
  tm = build_torch(cfg, w, torch.device("cuda"))
  bopt = optim.AdamW([p for p in bitsy_params(bm)], lr=cfg.lr)
  topt = torch.optim.AdamW(tm.parameters(), lr=cfg.lr)
  # Bitsy forward/loss
  blogits = b_forward(cfg, bm, idx_np)
  idx_flat = idx_np.reshape(-1)
  onehot = np.zeros((idx_flat.shape[0], cfg.vocab), np.float32)
  onehot[np.arange(idx_flat.shape[0]), idx_flat] = 1.0
  bloss = bt.nn.cross_entropy(blogits.reshape(cfg.batch * cfg.seq, cfg.vocab), BTensor.from_numpy(onehot, dev))
  bloss.backward(); bt.flush(); bt.wait()
  # Torch forward/loss
  tidx = torch.from_numpy(idx_np).cuda()
  tlogits = tm(tidx)
  tloss = torch.nn.functional.cross_entropy(tlogits.reshape(-1, cfg.vocab), tidx.reshape(-1))
  tloss.backward(); torch.cuda.synchronize()
  b_gdiff = {
    "embed": float(np.abs(bm[0].W.grad.numpy() - tm.embed.weight.grad.detach().cpu().numpy()).max()),
    "head": float(np.abs(bm[3].W.grad.numpy() - tm.head.weight.grad.detach().cpu().numpy().T).max()),
    "q0": float(np.abs(bm[1][0][2][0].grad.numpy() - tm.blocks[0].attn.q[0].weight.grad.detach().cpu().numpy().T).max()),
  }
  bopt.step(); topt.step(); bt.flush(); bt.wait(); torch.cuda.synchronize()
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "params": n_params, "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    "bitsy_loss": float(bloss.numpy().ravel()[0]), "torch_loss": float(tloss.item()),
    "logits_max_abs_diff": float(np.abs(blogits.numpy() - tlogits.detach().cpu().numpy()).max()),
  }
  report["passed_loss"] = abs(report["bitsy_loss"] - report["torch_loss"]) < 1e-3
  report["embed_grad_max_abs_diff"] = b_gdiff["embed"]
  report["head_grad_max_abs_diff"] = b_gdiff["head"]
  report["q0_grad_max_abs_diff"] = b_gdiff["q0"]
  report["passed_grads"] = all(v < 1e-3 for v in b_gdiff.values())
  report["passed"] = report["passed_loss"] and report["passed_grads"]
  # Timing: warmup, correct sync, median + spread, live GPU clock.
  bs, ts = [], []
  for _ in range(cfg.warmup): bt.flush(); bt.wait()
  clock0 = gpu_clock_mhz()
  for _ in range(cfg.iters):
    st = time.perf_counter(); b_forward(cfg, bm, idx_np); bt.flush(); bt.wait(); bs.append((time.perf_counter()-st)*1e3)
  for _ in range(cfg.iters):
    st = time.perf_counter(); tm(tidx); torch.cuda.synchronize(); ts.append((time.perf_counter()-st)*1e3)
  report["gpu_clock_mhz"] = (clock0 + gpu_clock_mhz()) * 0.5
  report["bitsy_fwd_ms"] = float(statistics.median(bs))
  report["torch_fwd_ms"] = float(statistics.median(ts))
  report["bitsy_fwd_spread_ms"] = [float(min(bs)), float(max(bs))]
  report["torch_fwd_spread_ms"] = [float(min(ts)), float(max(ts))]
  report["torch_over_bitsy"] = report["torch_fwd_ms"] / report["bitsy_fwd_ms"]
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"GPT-2 params {n_params}; loss {report['bitsy_loss']:.5f} vs {report['torch_loss']:.5f}; "
        f"logits diff {report['logits_max_abs_diff']:.3e}")
  print(f"grads: embed {report['embed_grad_max_abs_diff']:.3e} head "
        f"{report['head_grad_max_abs_diff']:.3e} q0 {report['q0_grad_max_abs_diff']:.3e} "
        f"passed={report['passed_grads']}")
  print(f"forward ms bitsy {report['bitsy_fwd_ms']:.2f} torch {report['torch_fwd_ms']:.2f} "
        f"ratio {report['torch_over_bitsy']:.2f}")
  return 0 if report["passed"] else 1

def bitsy_params(model):
  emb, layers, lnf, head, dev = model
  out = [emb.W, lnf.W, lnf.b, head.W, head.b]
  for ln1, ln2, q, k, v, o, c1, c2 in layers:
    out += [ln1.W, ln1.b, ln2.W, ln2.b, c1.W, c1.b, c2.W, c2.b]
    out += q + k + v + o
  return out

if __name__ == "__main__": raise SystemExit(run())
