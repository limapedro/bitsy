"""Measure every GEMM variant per shape and write the on-disk autotune cache."""

from __future__ import annotations

import json, os, sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(Path(__file__).resolve().parent)]

import bitsy as bt  # noqa: E402
from gemm import bench, CONFIG as GEMM_CONFIG  # noqa: E402

@dataclass(frozen=True, slots=True)
class TuneConfig:
  variants:tuple = (0, 1, 2, 4, 5, 6, 7)
  result_path:Path = ROOT / "benchmarks" / "results" / "gemm_tune.json"
  cache_path:Path = ROOT / "benchmarks" / "results" / "gemm_autotune.txt"

CONFIG = TuneConfig()

def run(config:TuneConfig = CONFIG) -> int:
  device = torch.cuda.get_device_name(0)
  report = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "device": device, "torch": torch.__version__, "bitsy": bt.version(),
            "measurements": {}}
  cache_lines = []
  for n in GEMM_CONFIG.sizes:
    times = {}
    for variant in config.variants:
      os.environ["BT_GEMM_VARIANT"] = str(variant)
      result = bench(n, GEMM_CONFIG)
      times[variant] = {"bitsy_ms": result["bitsy_ms"],
                        "ratio": result["ratio"],
                        "GFLOPS": result["bitsy_GFLOPS"]}
      print(f"n={n} variant={variant}: {result['bitsy_ms']:.3f} ms, "
            f"ratio {result['ratio']:.2f}")
    best = min(times, key=lambda v: times[v]["bitsy_ms"])
    report["measurements"][str(n)] = {"variants": times, "winner": best}
    cache_lines.append(f"{n} {n} {n} {best}")
    print(f"n={n} winner variant {best}")

  os.environ.pop("BT_GEMM_VARIANT", None)
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  config.cache_path.write_text("\n".join(cache_lines) + "\n", encoding="utf-8")
  print(f"cache: {config.cache_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
