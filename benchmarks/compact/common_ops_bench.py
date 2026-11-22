"""Common-operations speed check: bitsy_v2 (compact C+Vulkan) vs PyTorch CUDA.

Honest methodology: warmup, median + spread, live GPU clock logged. Each case is
end-to-end (operation + host readback) so both frameworks pay the same download.
A multi-step async loop is also reported (the fair framework-overhead metric).

Dataclass config, no argparse.
"""
from __future__ import annotations

import json, statistics, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import bitsy_v2 as bv
from bitsy_v2 import functional as F


def gpu_clock():
  try:
    o = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.current.graphics",
      "--format=csv,noheader"], text=True, timeout=5)
    return float(o.strip().splitlines()[0].split()[0])
  except Exception:
    return 0.0


@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path = ROOT / "build-rtvk"
  device:int = bv.Engine.VULKAN
  n:int = 1 << 20
  warmup:int = 5
  iters:int = 20
  steps:int = 10
  result_path:Path = ROOT / "benchmarks" / "results" / "common_ops.json"


def bench(fn, w, it):
  for _ in range(w): fn()
  ts = []
  for _ in range(it):
    s = time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)),float(max(ts))]}


def run(cfg=Config()):
  if not torch.cuda.is_available():
    print("CUDA unavailable; skipping"); return 0
  eng = bv.Engine(cfg.build_dir, cfg.device)
  rng = np.random.default_rng(0)
  x = rng.standard_normal(cfg.n).astype(np.float32)
  y = (rng.standard_normal(cfg.n) * 0.5 + 0.5).astype(np.float32)  # positive
  X, Y = bv.Tensor.from_numpy(eng.rt, x), bv.Tensor.from_numpy(eng.rt, y)
  tx, ty = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

  # (bitsy builder, torch builder) -> both return an object with .numpy()/cpu()
  cases = {
    "add":       (lambda: (X + Y),          lambda: tx + ty),
    "mul":       (lambda: (X * Y),          lambda: tx * ty),
    "sub":       (lambda: (X - Y),          lambda: tx - ty),
    "div":       (lambda: (X / Y),          lambda: tx / ty),
    "relu":      (lambda: X.relu(),         lambda: torch.relu(tx)),
    "sigmoid":   (lambda: X.sigmoid(),      lambda: torch.sigmoid(tx)),
    "tanh":      (lambda: X.tanh(),         lambda: torch.tanh(tx)),
    "exp":       (lambda: Y.exp(),          lambda: torch.exp(ty)),
    "log":       (lambda: Y.log(),          lambda: torch.log(ty)),
    "sqrt":      (lambda: Y.sqrt(),         lambda: torch.sqrt(ty)),
    "abs":       (lambda: X.abs(),          lambda: torch.abs(tx)),
    "gelu":      (lambda: X.gelu(),         lambda: torch.nn.functional.gelu(tx)),
    "silu":      (lambda: X.silu(),         lambda: torch.nn.functional.silu(tx)),
    "sum":       (lambda: X.sum(axis=0),    lambda: tx.sum()),
    "mean":      (lambda: X.mean(axis=0),   lambda: tx.mean()),
    "max":       (lambda: X.max(axis=0),    lambda: tx.max()),
    "fused_chain":(lambda: ((X*X)+X).relu(),lambda: torch.relu(tx*tx+tx)),
  }

  clk0 = gpu_clock()
  # Global warmup: the readback ring and GPU clocks need to settle before any
  # per-op measurement, otherwise the first op measured looks ~2x slower.
  for _ in range(cfg.warmup):
    for bf, tf in cases.values():
      bf().numpy(); tf().cpu().numpy()
  rows = []
  for name, (bf, tf) in cases.items():
    # correctness first
    got = bf().numpy().copy()
    want = tf().cpu().numpy()
    diff = float(np.abs(got - want).max())
    bt = bench(lambda: bf().numpy(), cfg.warmup, cfg.iters)
    tt = bench(lambda: tf().cpu().numpy(), cfg.warmup, cfg.iters)
    rows.append({"op":name, "diff":diff, "bitsy":bt, "torch":tt,
                 "torch_vs_bitsy":bt["median_ms"]/tt["median_ms"]})
    g, b, t = rows[-1]["torch_vs_bitsy"], bt["median_ms"], tt["median_ms"]
    print(f"{name:14s} bitsy {b:8.3f} ms  torch {t:8.3f} ms  gap {g:7.2f}x  diff {diff:.1e}")

  # multi-step async loop (fair framework overhead) — reuse captured chain
  eng.capture_begin(); ycap = ((X*X)+X).relu(); ycap.realize(); eng.capture_end()
  def bitsy_loop():
    for _ in range(cfg.steps): eng.replay_submit_async()
    ycap.numpy()
  def torch_loop():
    o = None
    for _ in range(cfg.steps): o = torch.relu(tx*tx+tx)
    return o.cpu().numpy()
  bl = bench(bitsy_loop, cfg.warmup, cfg.iters)
  tl = bench(torch_loop, cfg.warmup, cfg.iters)
  clk1 = gpu_clock()

  rep = {"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
                   "clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "ops":rows,
    "loop":{"bitsy":bl,"torch":tl,
            "bitsy_per_step":bl["median_ms"]/cfg.steps,
            "torch_per_step":tl["median_ms"]/cfg.steps,
            "gap":(bl["median_ms"]/cfg.steps)/(tl["median_ms"]/cfg.steps)},
    "note":"end-to-end (op + readback), median/warmup/spread/clock. gap>1 = bitsy slower."}
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"\nloop {cfg.steps} steps: bitsy {rep['loop']['bitsy_per_step']:.4f} ms/step "
        f"torch {rep['loop']['torch_per_step']:.4f} ms/step gap {rep['loop']['gap']:.2f}x")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0

if __name__ == "__main__": raise SystemExit(run())
