"""Batched vs submit-per-op Vulkan: a chain of pointwise ops.

Set BT_VK_NO_BATCH=1 for the old submit-per-op path, 0 (default) for batched.
One `bt.flush()` at the end submits the entire chain in batched mode.
"""

from __future__ import annotations

import json, os, statistics, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

@dataclass(frozen=True, slots=True)
class ChainConfig:
  n:int = 1 << 20
  ops:int = 20
  iters:int = 20
  warmup:int = 3
  result_path:Path = ROOT / "benchmarks" / "results" / "batch_chain.json"

CONFIG = ChainConfig()

def run(config:ChainConfig = CONFIG) -> int:
  mode = "eager" if os.environ.get("BT_VK_NO_BATCH") == "1" else "batched"
  rng = np.random.default_rng(0)
  a = bt.Tensor.from_numpy(rng.standard_normal(config.n).astype(np.float32), "vulkan")
  b = bt.Tensor.from_numpy(rng.standard_normal(config.n).astype(np.float32), "vulkan")

  def chain():
    x = a
    for _ in range(config.ops): x = bt.add(x, b)
    bt.flush()

  for _ in range(config.warmup): chain()
  times, device = [], []
  for _ in range(config.iters):
    start = time.perf_counter()
    chain()
    times.append((time.perf_counter() - start) * 1e3)
    device.append(bt.device_time_ms())
  report = {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "ops": config.ops, "elements": config.n,
            "bitsy_ms": statistics.median(times),
            "bitsy_device_ms": statistics.median(device),
            "spread_ms": [min(times), max(times)],
            "gpu": "GTX 1660 SUPER", "bitsy": bt.version()}
  out = config.result_path.parent / f"batch_chain_{mode}.json"
  out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"mode={mode} ops={config.ops} wall={report['bitsy_ms']:.3f} ms "
        f"device={report['bitsy_device_ms']:.3f} ms -> {out}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
