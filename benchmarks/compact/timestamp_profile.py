"""GPU timestamp profiling of a single eager op (PR #31).

Uses Vulkan timestamp queries in bitsy_rt to split wall time into:
  - GPU kernel time (compute shader)
  - GPU D2H copy time (DMA engine)
  - host overhead (submit, fence waits, Python/numpy)

Honest: warmup, median, spread; clock logged. Dataclass config, no argparse.
"""
from __future__ import annotations
import json, statistics, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import bitsy_v2 as bv


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
  n:int = 1 << 20
  warmup:int = 8
  iters:int = 20
  result_path:Path = ROOT / "benchmarks" / "results" / "timestamp_profile.json"


def run(cfg=Config()):
  eng = bv.Engine(cfg.build_dir, bv.Engine.VULKAN)
  x = np.random.default_rng(0).standard_normal(cfg.n).astype(np.float32)
  X = bv.Tensor.from_numpy(eng.rt, x)
  for _ in range(cfg.warmup): (X + X).numpy()
  walls, kernels, copies = [], [], []
  clk0 = gpu_clock()
  for _ in range(cfg.iters):
    s = time.perf_counter(); (X + X).numpy()
    walls.append((time.perf_counter()-s)*1e3)
    kernels.append(eng.device_time_ms()); copies.append(eng.copy_time_ms())
  clk1 = gpu_clock()
  w, k, c = (statistics.median(v) for v in (walls, kernels, copies))
  rep = {
    "generated_at":datetime.now(timezone.utc).isoformat(),
    "gpu":bv.Engine, "clock":[clk0,clk1],
    "config":{a:str(v) if isinstance(v,Path) else v for a,v in asdict(cfg).items()},
    "median_ms":{"wall":w,"gpu_kernel":k,"gpu_copy":c,"host_overhead":w-k-c},
    "fraction":{"gpu_kernel":k/w,"gpu_copy":c/w,"host_overhead":(w-k-c)/w},
    "spread_ms":{"wall":[min(walls),max(walls)],
                 "gpu_kernel":[min(kernels),max(kernels)],
                 "gpu_copy":[min(copies),max(copies)]},
    "note":"GPU times from Vulkan timestamp queries (period 1ns); wall from perf_counter.",
  }
  rep["gpu"] = "NVIDIA GeForce GTX 1660 SUPER"
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"wall {w:.3f} ms | GPU kernel {k:.4f} ms ({k/w:.1%}) | "
        f"GPU copy {c:.3f} ms ({c/w:.1%}) | host overhead {w-k-c:.3f} ms ({(w-k-c)/w:.1%})")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0

if __name__ == "__main__": raise SystemExit(run())
