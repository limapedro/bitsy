"""Record-once replay: Bitsy captured command buffer vs PyTorch CUDA graph."""

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
class ReplayConfig:
  elements:int = 1 << 20
  ops:int = 20
  iters:int = 40
  warmup:int = 3
  result_path:Path = ROOT / "benchmarks" / "results" / "replay_chain.json"

CONFIG = ReplayConfig()

def bitsy_times(a, b, config:ReplayConfig):
  def chain():
    x = a
    for _ in range(config.ops): x = bt.add(x, b)
    return x
  for _ in range(config.warmup): chain(); bt.flush(); bt.wait()
  start = time.perf_counter()
  for _ in range(config.iters): chain(); bt.flush(); bt.wait()
  recorded = (time.perf_counter() - start) * 1e3 / config.iters

  bt.capture_begin()
  chain()
  bt.capture_end()
  start = time.perf_counter()
  bt.replay(config.iters)
  bt.wait()
  replayed = (time.perf_counter() - start) * 1e3 / config.iters
  return recorded, replayed

def torch_times(ta, tb, config:ReplayConfig):
  def chain():
    x = ta
    for _ in range(config.ops): x = torch.add(x, tb)
    return x
  for _ in range(config.warmup): chain()
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(config.iters): chain()
  torch.cuda.synchronize()
  eager = (time.perf_counter() - start) * 1e3 / config.iters

  stream = torch.cuda.Stream()
  stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
    for _ in range(config.warmup): chain()
  torch.cuda.current_stream().wait_stream(stream)
  torch.cuda.synchronize()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    chain()
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(config.iters): graph.replay()
  torch.cuda.synchronize()
  graphed = (time.perf_counter() - start) * 1e3 / config.iters
  return eager, graphed

def run(config:ReplayConfig = CONFIG) -> int:
  rng = np.random.default_rng(0)
  a_np = rng.standard_normal(config.elements).astype(np.float32)
  b_np = rng.standard_normal(config.elements).astype(np.float32)
  a = bt.Tensor.from_numpy(a_np, "vulkan")
  b = bt.Tensor.from_numpy(b_np, "vulkan")
  rec_ms, rep_ms = bitsy_times(a, b, config)
  ta = torch.from_numpy(a_np).cuda()
  tb = torch.from_numpy(b_np).cuda()
  eager_ms, graph_ms = torch_times(ta, tb, config)
  report = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "gpu_clock_mhz": gpu_clock_mhz(),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "bitsy_recorded_ms_per_chain": rec_ms,
            "bitsy_replay_ms_per_chain": rep_ms,
            "bitsy_replay_speedup": rec_ms / rep_ms,
            "torch_eager_ms_per_chain": eager_ms,
            "torch_graph_ms_per_chain": graph_ms,
            "replay_vs_graph_ratio": graph_ms / rep_ms}
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"bitsy recorded {rec_ms:.3f} ms/chain | replay {rep_ms:.3f} ms/chain "
        f"({report['bitsy_replay_speedup']:.2f}x)")
  print(f"torch eager {eager_ms:.3f} | graph {graph_ms:.3f} | "
        f"replay/graph ratio {report['replay_vs_graph_ratio']:.2f}")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
