"""Fused add+relu vs separate add then relu, same GPU and fair sync."""

from __future__ import annotations

import json, statistics, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

@dataclass(frozen=True, slots=True)
class FusionConfig:
  n:int = 1 << 20
  pairs:int = 20
  iters:int = 20
  warmup:int = 3
  result_path:Path = ROOT / "benchmarks" / "results" / "fusion.json"

CONFIG = FusionConfig()

def measure(fn, config:FusionConfig):
  for _ in range(config.warmup): fn()
  times, device = [], []
  for _ in range(config.iters):
    start = time.perf_counter()
    fn()
    times.append((time.perf_counter() - start) * 1e3)
    device.append(bt.device_time_ms())
  return statistics.median(times), statistics.median(device)

def run(config:FusionConfig = CONFIG) -> int:
  rng = np.random.default_rng(1)
  a = bt.Tensor.from_numpy(rng.standard_normal(config.n).astype(np.float32), "vulkan")
  b = bt.Tensor.from_numpy(rng.standard_normal(config.n).astype(np.float32), "vulkan")

  def separate():
    x = a
    for _ in range(config.pairs): x = bt.relu(bt.add(x, b))
    bt.flush(); bt.wait()

  def fused():
    x = a
    for _ in range(config.pairs): x = bt.add_relu(x, b)
    bt.flush(); bt.wait()

  sep_ms, sep_dev = measure(separate, config)
  fus_ms, fus_dev = measure(fused, config)
  report = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "gpu": "GTX 1660 SUPER", "bitsy": bt.version(),
            "elements": config.n, "pairs": config.pairs,
            "separate_ms": sep_ms, "fused_ms": fus_ms,
            "speedup": sep_ms / fus_ms,
            "separate_device_ms": sep_dev, "fused_device_ms": fus_dev}
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"separate {sep_ms:.3f} ms (device {sep_dev:.3f}) | "
        f"fused {fus_ms:.3f} ms (device {fus_dev:.3f}) | speedup {report['speedup']:.2f}x")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
