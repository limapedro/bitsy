"""Three-way timing: isolated kernel (timestamp proxy), Bitsy, PyTorch CUDA.

Configured with a dataclass, no argparse. Profiling is opt-in and its overhead
is measured by running the same workload with and without it.
"""

from __future__ import annotations

import json, statistics, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

@dataclass(frozen=True, slots=True)
class ThreeWayConfig:
  iters:int = 50
  warmup:int = 5
  cases:tuple = ("add", "mul", "relu", "matmul")
  result_path:Path = ROOT / "benchmarks" / "results" / "three_way.json"
  trace_dir:Path = ROOT / "benchmarks" / "results"

CONFIG = ThreeWayConfig()

def _torch_time(fn, config:ThreeWayConfig):
  for _ in range(config.warmup): fn()
  torch.cuda.synchronize()
  times, events = [], []
  for _ in range(config.iters):
    start = time.perf_counter()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    events.append(e0.elapsed_time(e1))
    times.append((time.perf_counter() - start) * 1e3)
  return statistics.median(times), statistics.median(events)

def _bitsy_case(op:str, a_np, b_np, config:ThreeWayConfig, profile:bool):
  A = bt.Tensor.from_numpy(a_np, "vulkan")
  B = bt.Tensor.from_numpy(b_np, "vulkan")
  def call():
    if op == "matmul": return bt.matmul(A, B)
    if op == "relu": return bt.relu(A)
    return getattr(bt, op)(A, B)
  for _ in range(config.warmup): call(); bt.flush(); bt.wait()
  trace = CONFIG.trace_dir / f"trace_{op}_{'prof' if profile else 'plain'}.json"
  if profile:
    bt.profile(True); bt.profile_reset()
  times = []
  for _ in range(config.iters):
    t0 = time.perf_counter()
    call()
    bt.flush(); bt.wait()
    times.append((time.perf_counter() - t0) * 1e3)
  wall = statistics.median(times)
  summary = None
  if profile:
    bt.profile_dump(str(trace))
    summary = json.loads(trace.read_text())["summary"]
    bt.profile(False)
  return wall, summary

def run(config:ThreeWayConfig = CONFIG) -> int:
  torch.backends.cudnn.benchmark = True
  rng = np.random.default_rng(0)
  n = 1 << 20
  rows = []
  for op in config.cases:
    if op == "matmul":
      a_np = rng.standard_normal((256, 256)).astype(np.float32)
      b_np = rng.standard_normal((256, 256)).astype(np.float32)
    else:
      a_np = rng.standard_normal(n).astype(np.float32)
      b_np = rng.standard_normal(n).astype(np.float32)
    ta = torch.from_numpy(a_np).cuda()
    tb = torch.from_numpy(b_np).cuda()
    if op == "matmul":
      tfn = lambda: torch.mm(ta, tb)
    elif op == "relu":
      tfn = lambda: torch.relu(ta)
    else:
      tfn = lambda: getattr(torch, op)(ta, tb)

    torch_wall, torch_dev = _torch_time(tfn, config)
    plain, _ = _bitsy_case(op, a_np, b_np, config, False)
    prof, summary = _bitsy_case(op, a_np, b_np, config, True)

    gpu_ms = summary["gpu_ms"] if summary else 0.0
    per_dispatch_us = gpu_ms * 1000.0 / max(1, summary["dispatches"]) if summary else 0.0
    row = {
      "op": op, "torch_wall_ms": torch_wall, "torch_device_ms": torch_dev,
      "bitsy_wall_ms": plain, "bitsy_profiled_wall_ms": prof,
      "profiler_overhead_us": (prof - plain) * 1000.0,
      "bitsy_gpu_ms_total": gpu_ms,
      "bitsy_gpu_us_per_dispatch": per_dispatch_us,
      "gap_us_wall": (plain - torch_wall) * 1000.0,
      "ratio": torch_wall / plain,
      "dispatch_count": summary["dispatches"] if summary else 0,
      "barrier_count": summary["barriers"] if summary else 0,
      "submit_count": summary["submits"] if summary else 0,
      "alloc_count": summary["allocations"] if summary else 0,
      "alloc_hits": summary["allocation_hits"] if summary else 0,
      "pct_dram_ceiling": summary["pct_dram_ceiling"] if summary else 0.0,
      "pct_fp32_ceiling": summary["pct_fp32_ceiling"] if summary else 0.0,
      "note": "GPU time is per-dispatch timestamp sum (kernel proxy, not a 1000-repeat isolated measurement)",
    }
    rows.append(row)
    print(f"{op:8} torch {torch_wall:.4f} ms | bitsy {plain:.4f} ms | "
          f"gpu {per_dispatch_us:.1f} us/dispatch | gap {row['gap_us_wall']:.1f} us | "
          f"ratio {row['ratio']:.2f}")

  report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                            "bitsy": bt.version()},
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "results": rows}
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
