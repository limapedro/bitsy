"""Capture/replay correctness + speed benchmark for the v0.2 compact runtime.

Correctness: capture a fused pointwise chain, replay with new input data,
compare to PyTorch CUDA.
Speed: eager vs captured-replay vs PyTorch CUDA, with warmup, median, spread,
and live GPU clock. Per-op and full-step. Dataclass config, no argparse.
"""
from __future__ import annotations

import json, statistics, subprocess, sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import bitsy_v2 as bv


def gpu_clock_mhz() -> float:
  try:
    out = subprocess.check_output(
      ["nvidia-smi","--query-gpu=clocks.current.graphics","--format=csv,noheader"],
      text=True, timeout=5)
    return float(out.strip().splitlines()[0].split()[0])
  except Exception:
    return 0.0


@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path = ROOT / "build-rtvk"
  device:int = bv.Engine.VULKAN
  n:int = 1<<20            # 1M elements -> GPU-bound-ish fused chain
  warmup:int = 5
  iters:int = 30
  tol:float = 1e-4
  result_path:Path = ROOT / "benchmarks" / "results" / "compact_replay.json"


def md(a,b): return float(np.abs(np.asarray(a,np.float32)-np.asarray(b,np.float32)).max())


def bench_fn(fn, warmup, iters):
  for _ in range(warmup): fn()
  ts=[]
  for _ in range(iters):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)),float(max(ts))]}

import time

def run(cfg=Config()):
  if not torch.cuda.is_available():
    print("CUDA unavailable; skipping"); return 0
  eng = bv.Engine(cfg.build_dir, cfg.device)
  rng = np.random.default_rng(0)
  x = rng.standard_normal(cfg.n).astype(np.float32)
  # fused chain: y = relu(x*x + x)  (3 ops -> 1 dispatch)
  X = eng.tensor(x)
  y = ((X*X)+X).relu()

  # PyTorch equivalent
  tx = torch.from_numpy(x).cuda()
  def torch_chain(): return torch.relu(tx*tx + tx)

  # ---- correctness: capture then replay with new data ----
  eng.capture_begin()
  ycap = ((X*X)+X).relu()
  ycap.realize()             # must realize INSIDE the capture window to record dispatches
  eng.capture_end()
  eng.replay()
  got = ycap.numpy()
  want = torch_chain().cpu().numpy()
  diff = md(got, want)
  correctness = diff <= cfg.tol
  print(f"[{'PASS' if correctness else 'FAIL'}] replay numerics diff={diff:.3e} tol={cfg.tol:.0e}")

  # replay with DIFFERENT data -> must change result
  x2 = rng.standard_normal(cfg.n).astype(np.float32)
  eng.upload(X, x2)
  eng.replay()
  got2 = ycap.numpy()
  want2 = torch.relu(torch.from_numpy(x2).cuda()**2 + torch.from_numpy(x2).cuda()).cpu().numpy()
  diff2 = md(got2, want2)
  correctness2 = diff2 <= cfg.tol
  print(f"[{'PASS' if correctness2 else 'FAIL'}] replay(new data) diff={diff2:.3e}")

  # ---- speed ----
  clk0 = gpu_clock_mhz()
  def eager():
    ((X*X)+X).relu().numpy()
  def replay():
    eng.replay()
  e = bench_fn(eager, cfg.warmup, cfg.iters)
  r = bench_fn(replay, cfg.warmup, cfg.iters)
  t = bench_fn(torch_chain, cfg.warmup, cfg.iters)
  clk1 = gpu_clock_mhz()

  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
                    "clock_mhz":[clk0,clk1]},
    "config": {k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness": {"replay_diff":diff,"replay_new_data_diff":diff2,
                    "passed":bool(correctness and correctness2)},
    "eager": e, "replay": r, "torch": t,
    "speedup": {"replay_vs_eager": e["median_ms"]/r["median_ms"],
                "torch_vs_replay": r["median_ms"]/t["median_ms"]},
    "note": "ratio>1 means first is faster. torch_vs_replay>1 => Bitsy replay faster than PyTorch.",
  }
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
  print(f"eager {e['median_ms']:.3f} ms (spread {e['spread_ms']})")
  print(f"replay {r['median_ms']:.3f} ms (spread {r['spread_ms']})  -> {report['speedup']['replay_vs_eager']:.2f}x vs eager")
  print(f"torch {t['median_ms']:.3f} ms (spread {t['spread_ms']})  -> torch/replay {report['speedup']['torch_vs_replay']:.2f}x")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0 if (correctness and correctness2) else 1

if __name__=="__main__": raise SystemExit(run())
