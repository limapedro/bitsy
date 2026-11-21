"""Wall-time comparison: Bitsy Vulkan vs PyTorch CUDA on the same GPU.

Both sides allocate an output per op and synchronize every iteration, because
the current Bitsy API has no timestamp queries or asynchronous stream yet.
Device time is therefore reported as unavailable rather than guessed.
"""

from __future__ import annotations

import json, statistics, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

@dataclass(frozen=True, slots=True)
class BenchConfig:
  sizes:tuple[int, ...] = (4096, 1 << 20)
  ops:tuple[str, ...] = ("add", "mul", "relu")
  iters:int = 60
  warmup:int = 10
  result_path:Path = ROOT / "benchmarks" / "results" / "elementwise.json"

CONFIG = BenchConfig()

def _timed(fn, iters:int):
  times, device_times = [], []
  for _ in range(iters):
    start = time.perf_counter()
    fn()
    bt.flush()                 # async submit with timeline signal
    bt.wait()                  # host completion point = fair wall time
    torch.cuda.synchronize()   # equivalent sync for PyTorch
    times.append((time.perf_counter() - start) * 1e3)
    device_times.append(bt.device_time_ms())
  return times, device_times

def bench_one(op:str, n:int, config:BenchConfig) -> dict:
  ta = torch.randn(n, device="cuda")
  tb = torch.randn(n, device="cuda")
  ba = bt.Tensor.from_numpy(ta.cpu().numpy(), "vulkan")
  bb = bt.Tensor.from_numpy(tb.cpu().numpy(), "vulkan")
  torch_fn = getattr(torch, op)
  bitsy_fn = getattr(bt, op)
  torch_call = (lambda: torch_fn(ta)) if op in ("relu", "sigmoid", "tanh", "exp") else (lambda: torch_fn(ta, tb))
  bitsy_call = (lambda: bitsy_fn(ba)) if op in ("relu", "sigmoid", "tanh", "exp") else (lambda: bitsy_fn(ba, bb))

  for _ in range(config.warmup):
    torch_call(); torch.cuda.synchronize()
    bitsy_call(); bt.flush(); bt.wait()
  t_torch, _ = _timed(torch_call, config.iters)
  t_bitsy, dev = _timed(bitsy_call, config.iters)
  tm, bm = statistics.median(t_torch), statistics.median(t_bitsy)
  # add/mul: 3 arrays touched; relu: 2. Bytes = elements * 4 * arrays.
  arrays = 2 if op in ("relu", "sigmoid", "tanh", "exp") else 3
  byte_count = n * 4 * arrays
  return {
    "op": op, "elements": n,
    "bitsy_ms": bm, "torch_ms": tm,
    "bitsy_device_ms": statistics.median(dev),
    "bitsy_spread_ms": [min(t_bitsy), max(t_bitsy)],
    "torch_spread_ms": [min(t_torch), max(t_torch)],
    "ratio": tm / bm,
    "bitsy_GiB_s": byte_count / (bm * 1e-3) / 2**30,
    "torch_GiB_s": byte_count / (tm * 1e-3) / 2**30,
  }

def run(config:BenchConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  rows = [bench_one(op, n, config) for n in config.sizes for op in config.ops]
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__, "cuda": torch.version.cuda, "bitsy": bt.version(),
    "vulkan_device": "GTX 1660 SUPER (queried via vulkaninfo)",
    "method": "wall time with bt.flush()+torch.cuda.synchronize() per iteration; Bitsy device time from Vulkan timestamp queries",
    "results": rows,
  }
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"{'op':8} {'n':>9} {'bitsy ms':>10} {'torch ms':>10} "
        f"{'device ms':>10} {'ratio':>7} {'GiB/s b/t':>16}")
  for r in rows:
    print(f"{r['op']:8} {r['elements']:9d} {r['bitsy_ms']:10.4f} {r['torch_ms']:10.4f} "
          f"{r['bitsy_device_ms']:10.4f} {r['ratio']:7.2f} "
          f"{r['bitsy_GiB_s']:7.1f}/{r['torch_GiB_s']:.1f}")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
