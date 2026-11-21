"""SGEMM wall-time comparison: Bitsy Vulkan vs PyTorch CUDA on the same GPU.

Set BT_GEMM_TILED=0/1 before launching to select the naive or tiled shader.
Every iteration synchronizes on both sides; no device-time claim is made.
"""

from __future__ import annotations

import json, os, statistics, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

@dataclass(frozen=True, slots=True)
class GemmConfig:
  sizes:tuple[int, ...] = (256, 1024)
  iters:int = 30
  warmup:int = 5
  result_path:Path = ROOT / "benchmarks" / "results" / "gemm.json"

CONFIG = GemmConfig()

def _timed(fn, iters:int):
  out, device = [], []
  for _ in range(iters):
    start = time.perf_counter()
    fn()
    bt.flush()
    bt.wait()
    torch.cuda.synchronize()
    out.append((time.perf_counter() - start) * 1e3)
    device.append(bt.device_time_ms())
  return out, device

def bench(n:int, config:GemmConfig) -> dict:
  ta = torch.randn(n, n, device="cuda")
  tb = torch.randn(n, n, device="cuda")
  ba = bt.Tensor.from_numpy(ta.cpu().numpy(), "vulkan")
  bb = bt.Tensor.from_numpy(tb.cpu().numpy(), "vulkan")
  for _ in range(config.warmup):
    torch.mm(ta, tb); torch.cuda.synchronize()
    bt.matmul(ba, bb); bt.flush(); bt.wait()
  tt, _ = _timed(lambda: torch.mm(ta, tb), config.iters)
  bt_times, dev = _timed(lambda: bt.matmul(ba, bb), config.iters)
  tm, bm = statistics.median(tt), statistics.median(bt_times)
  flops = 2 * n * n * n
  return {"n": n, "torch_ms": tm, "bitsy_ms": bm,
          "bitsy_device_ms": statistics.median(dev), "ratio": tm / bm,
          "torch_GFLOPS": flops / (tm * 1e-3) / 1e9,
          "bitsy_GFLOPS": flops / (bm * 1e-3) / 1e9,
          "torch_spread_ms": [min(tt), max(tt)],
          "bitsy_spread_ms": [min(bt_times), max(bt_times)]}

def run(config:GemmConfig = CONFIG) -> int:
  mode = { "0": "naive", "1": "tiled", "2": "rb", "3": "rb_splitk" }.get(
      os.environ.get("BT_GEMM_VARIANT", ""), "auto")
  rows = [bench(n, config) for n in config.sizes]
  report = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "bitsy": bt.version(),
            "method": "wall time with bt.flush()+torch.cuda.synchronize() per iteration; device time from timestamp queries",
            "results": rows}
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"mode={mode}")
  print(f"{'n':>6} {'bitsy ms':>10} {'torch ms':>10} {'device ms':>10} "
        f"{'ratio':>7} {'b GFLOPS':>10} {'t GFLOPS':>10}")
  for r in rows:
    print(f"{r['n']:6d} {r['bitsy_ms']:10.3f} {r['torch_ms']:10.3f} "
          f"{r['bitsy_device_ms']:10.3f} {r['ratio']:7.2f} "
          f"{r['bitsy_GFLOPS']:10.1f} {r['torch_GFLOPS']:10.1f}")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
