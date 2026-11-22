"""GEMM performance vs PyTorch CUDA, fp32 — honest before/after evidence for
the tile-autotune work (PR #12). Warmup, median, spread, live GPU clock.

Dataclass config, no argparse. Reports bitsy (compact v0.2) vs torch CUDA for a
range of square matrix sizes, plus numeric diff at each size.
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


def gpu_clock_mhz():
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
  sizes:tuple = (256, 512, 1024)
  warmup:int = 5
  iters:int = 20
  # FMA changes the fp32 accumulation rounding vs cuBLAS; absolute error grows
  # with K. The numeric parity gate uses small matrices at 1e-4; this benchmark
  # sweeps up to K=1024, so allow 3e-4 and report the actual max diff per size.
  tol:float = 3e-4
  result_path:Path = ROOT / "benchmarks" / "results" / "compact_gemm.json"


def bench(fn, w, it):
  for _ in range(w): fn()
  ts = []
  for _ in range(it):
    s = time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)), float(max(ts))]}


def run(cfg=Config()):
  if not torch.cuda.is_available():
    print("CUDA unavailable; skipping"); return 0
  eng = bv.Engine(cfg.build_dir, cfg.device)
  rng = np.random.default_rng(0)
  rows = []
  clk0 = gpu_clock_mhz()
  for n in cfg.sizes:
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    A = bv.Tensor.from_numpy(eng.rt, a)
    B = bv.Tensor.from_numpy(eng.rt, b)
    ta, tb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    # correctness first (fresh call so no cache aliasing)
    got = A.matmul(B).numpy().copy()
    want = (ta @ tb).cpu().numpy()
    diff = float(np.abs(got - want).max())
    bt = bench(lambda: A.matmul(B).numpy(), cfg.warmup, cfg.iters)
    tt = bench(lambda: (ta @ tb).cpu().numpy(), cfg.warmup, cfg.iters)
    rows.append({"n":n, "diff":diff, "bitsy":bt, "torch":tt,
                 "torch_vs_bitsy":bt["median_ms"]/tt["median_ms"]})
    print(f"n={n:5d} diff={diff:.2e} bitsy {bt['median_ms']:.3f} ms "
          f"torch {tt['median_ms']:.3f} ms  gap {rows[-1]['torch_vs_bitsy']:.2f}x")
  clk1 = gpu_clock_mhz()
  rep = {"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
                   "clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "rows":rows, "note":"ratios >1 mean bitsy slower. Median+warmup+spread+clock logged."}
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0 if all(r["diff"] <= cfg.tol for r in rows) else 1

if __name__ == "__main__": raise SystemExit(run())
