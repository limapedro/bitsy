"""Host overlap: submit several pointwise chains before one completion wait."""

from __future__ import annotations

import json, statistics, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "benchmarks")]
import bitsy as bt  # noqa: E402
from common.clock import gpu_clock_mhz  # noqa: E402

@dataclass(frozen=True, slots=True)
class AsyncConfig:
  elements:int = 1 << 20
  ops:int = 20
  batch:int = 8
  iters:int = 20
  warmup:int = 3
  result_path:Path = ROOT / "benchmarks" / "results" / "async_chain.json"

CONFIG = AsyncConfig()

def measure(mode:str, a, b, config:AsyncConfig):
  def chain():
    x = a
    for _ in range(config.ops): x = bt.add(x, b)
    return x
  if mode == "sync":
    for _ in range(config.warmup): chain(); bt.flush(); bt.wait()
    start = time.perf_counter()
    for _ in range(config.iters):
      chain(); bt.flush(); bt.wait()
    return (time.perf_counter() - start) * 1e3 / config.iters
  for _ in range(config.warmup):
    for _ in range(config.batch): chain()
    bt.flush(); bt.wait()
  start = time.perf_counter()
  for _ in range(config.iters // config.batch):
    for _ in range(config.batch): chain()
    bt.flush(); bt.wait()
  return (time.perf_counter() - start) * 1e3 / (config.iters // config.batch * config.batch)

def run(config:AsyncConfig = CONFIG) -> int:
  rng = np.random.default_rng(0)
  a = bt.Tensor.from_numpy(rng.standard_normal(config.elements).astype(np.float32), "vulkan")
  b = bt.Tensor.from_numpy(rng.standard_normal(config.elements).astype(np.float32), "vulkan")
  sync_ms = measure("sync", a, b, config)
  clock0 = gpu_clock_mhz()
  async_ms = measure("async", a, b, config)
  clock1 = gpu_clock_mhz()
  report = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "gpu_clock_mhz": (clock0 + clock1) / 2,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "sync_ms_per_chain": sync_ms, "async_ms_per_chain": async_ms,
            "speedup": sync_ms / async_ms}
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"sync {sync_ms:.3f} ms/chain | async(batch={config.batch}) "
        f"{async_ms:.3f} ms/chain | speedup {report['speedup']:.2f}x")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
