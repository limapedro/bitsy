"""Profile PyTorch CUDA kernels for the ops in the three-way comparison."""

from __future__ import annotations

import json, sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True, slots=True)
class ProfileConfig:
  iters:int = 20
  warmup:int = 5
  result_path:Path = ROOT / "benchmarks" / "results" / "torch_kernels.json"

CONFIG = ProfileConfig()

def kernels_for(op:str, config:ProfileConfig):
  n = 1 << 20
  a = torch.randn(n, device="cuda")
  b = torch.randn(n, device="cuda")
  if op == "matmul":
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    fn = lambda: torch.mm(a, b)
  elif op == "relu":
    fn = lambda: torch.relu(a)
  else:
    fn = lambda: getattr(torch, op)(a, b)
  for _ in range(config.warmup): fn()
  torch.cuda.synchronize()
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(config.iters): fn()
    torch.cuda.synchronize()
  events = []
  for e in prof.key_averages():
    if e.device_time_total > 0 or "cuda" in e.key.lower() or "gemm" in e.key.lower() or "elementwise" in e.key.lower():
      events.append({"name": e.key, "calls": e.count,
                     "device_us_total": e.device_time_total,
                     "device_us_per_call": e.device_time_total / max(1, e.count),
                     "cpu_us_total": e.cpu_time_total})
  events.sort(key=lambda e: -e["device_us_total"])
  return events[:12]

def run(config:ProfileConfig = CONFIG) -> int:
  torch.backends.cudnn.benchmark = True
  result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "ops": {}}
  for op in ("add", "mul", "relu", "matmul"):
    ev = kernels_for(op, config)
    result["ops"][op] = ev
    print(f"{op}: {ev[0]['name'] if ev else 'no CUDA kernels'} "
          f"({round(ev[0]['device_us_per_call'], 1) if ev else 0} us/call)")
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
