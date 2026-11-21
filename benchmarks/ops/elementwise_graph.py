"""Three-way elementwise timing: eager vs captured execution, Bitsy vs torch.

Follows the campaign's three-way methodology (isolated/captured execution,
1000+ replays per command buffer) which elementwise.py's per-op sync does not.
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
import bitsy as bt  # noqa: E402
from common.clock import gpu_clock_mhz  # noqa: E402

@dataclass(frozen=True, slots=True)
class ElemGraphConfig:
  ops:tuple = ("add", "mul", "relu")
  sizes:tuple = (4096, 1 << 20)
  iters:int = 100
  trials:int = 5
  warmup:int = 3
  result_path:Path = ROOT / "benchmarks" / "results" / "elementwise_graph.json"

CONFIG = ElemGraphConfig()

def bitsy_eager(op, a, b, config):
  for _ in range(config.warmup): op(a, b); bt.flush(); bt.wait()
  samples = []
  for _ in range(config.trials):
    start = time.perf_counter()
    for _ in range(config.iters): op(a, b); bt.flush(); bt.wait()
    samples.append((time.perf_counter() - start) * 1e3 / config.iters)
  return statistics.median(samples)

def bitsy_captured(op, a, b, config):
  bt.capture_begin(); op(a, b); bt.capture_end()
  bt.replay(config.warmup * config.iters); bt.wait()
  samples = []
  for _ in range(config.trials):
    start = time.perf_counter()
    bt.replay(config.iters); bt.wait()
    samples.append((time.perf_counter() - start) * 1e3 / config.iters)
  return statistics.median(samples)

def torch_eager(op, a, b, config):
  dev = a.device
  for _ in range(config.warmup): op(a, b); torch.cuda.synchronize()
  samples = []
  for _ in range(config.trials):
    start = time.perf_counter()
    for _ in range(config.iters): op(a, b)
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1e3 / config.iters)
  return statistics.median(samples)

def torch_captured(op, a, b, config):
  if a.device.type != "cuda": return None
  stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
    for _ in range(config.warmup): op(a, b)
  torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): op(a, b)
  torch.cuda.synchronize()
  samples = []
  for _ in range(config.trials):
    start = time.perf_counter()
    for _ in range(config.iters): graph.replay()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1e3 / config.iters)
  return statistics.median(samples)

def run(config:ElemGraphConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  rng = np.random.default_rng(0)
  report = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "measurements": {}}
  clock0 = gpu_clock_mhz()
  for n in config.sizes:
    report["measurements"][str(n)] = {}
    for name in config.ops:
      x = rng.standard_normal(n).astype(np.float32)
      y = rng.standard_normal(n).astype(np.float32) if name != "relu" else x
      ba, bb = bt.Tensor.from_numpy(x, "vulkan"), bt.Tensor.from_numpy(y, "vulkan")
      ta, tb = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
      bop = (lambda p, q: bt.add(p, q)) if name == "add" else \
            (lambda p, q: bt.mul(p, q)) if name == "mul" else (lambda p, q: bt.relu(p))
      top = (lambda p, q: torch.add(p, q)) if name == "add" else \
            (lambda p, q: torch.mul(p, q)) if name == "mul" else (lambda p, q: torch.relu(p))
      be = bitsy_eager(bop, ba, bb, config)
      bc = bitsy_captured(bop, ba, bb, config)
      te = torch_eager(top, ta, tb, config)
      tg = torch_captured(top, ta, tb, config)
      m = {"bitsy_eager_ms": be, "bitsy_captured_ms": bc,
           "torch_eager_ms": te, "torch_graph_ms": tg,
           "eager_ratio_torch_over_bitsy": te / be if be else 0,
           "captured_ratio_graph_over_bitsy": (tg / bc if tg and bc else 0)}
      report["measurements"][str(n)][name] = m
      print(f"n={n:>8} {name:>4}: eager {be:.4f} ({m['eager_ratio_torch_over_bitsy']:.2f}x) "
            f"captured {bc:.4f} vs torch graph {tg:.4f} "
            f"({m['captured_ratio_graph_over_bitsy']:.2f}x)")
  report["gpu_clock_mhz"] = (clock0 + gpu_clock_mhz()) / 2
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
