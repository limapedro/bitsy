"""End-to-end training-step wall time: Bitsy Vulkan vs PyTorch CUDA.

Mirrors the step bodies in mnist_mlp.py on synthetic data so the benchmark is
self-contained. BITSY_DEFERRED_UPLOAD selects the transfer path; each mode runs
in its own process and writes its own JSON. Dataclass config, no argparse.
"""

from __future__ import annotations

import json, os, statistics, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "benchmarks")]
import bitsy as bt  # noqa: E402
from common.clock import gpu_clock_mhz  # noqa: E402

DEFERRED = os.environ.get("BITSY_DEFERRED_UPLOAD", "1") != "0"

@dataclass(frozen=True, slots=True)
class StepConfig:
  in_features:int = 784
  hidden:int = 64
  out_features:int = 10
  batch:int = 64
  iters:int = 200
  warmup:int = 20
  lr:float = 0.1
  seed:int = 0
  deferred:bool = DEFERRED
  result_path:Path = (ROOT / "benchmarks" / "results" /
                      ("train_step_deferred.json" if DEFERRED else "train_step_baseline.json"))

CONFIG = StepConfig()

def make_data(config:StepConfig):
  rng = np.random.default_rng(config.seed)
  x = rng.standard_normal((config.batch, config.in_features)).astype(np.float32)
  y = np.zeros((config.batch, config.out_features), np.float32)
  y[np.arange(config.batch), rng.integers(0, config.out_features, config.batch)] = 1.0
  w1 = (rng.standard_normal((config.in_features, config.hidden)) * 0.1).astype(np.float32)
  b1 = np.zeros((1, config.hidden), np.float32)
  w2 = (rng.standard_normal((config.hidden, config.out_features)) * 0.1).astype(np.float32)
  b2 = np.zeros((1, config.out_features), np.float32)
  return x, y, {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

def bitsy_step_factory(arrays, config:StepConfig):
  dev = "vulkan"
  m1 = bt.nn.Linear(config.in_features, config.hidden, np.random.default_rng(config.seed), dev)
  m2 = bt.nn.Linear(config.hidden, config.out_features, np.random.default_rng(config.seed), dev)
  m1.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w1"], dev))
  m1.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b1"], dev))
  m2.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w2"], dev))
  m2.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b2"], dev))
  model = bt.nn.Sequential(m1, bt.nn.Tanh(), m2)
  opt = bt.optim.SGD(model.parameters(), config.lr)
  def step(x, y):
    xb = bt.autograd.Tensor.from_numpy(x, dev)
    yb = bt.autograd.Tensor.from_numpy(y, dev)
    opt.zero_grad()
    pred = model(xb)
    loss = bt.nn.cross_entropy(pred, yb)
    loss.backward()
    opt.step()
    return float(loss.numpy().ravel()[0])  # single inherent readback sync
  return step

def torch_step_factory(arrays, config:StepConfig):
  torch.manual_seed(config.seed)
  model = torch.nn.Sequential(
    torch.nn.Linear(config.in_features, config.hidden), torch.nn.Tanh(),
    torch.nn.Linear(config.hidden, config.out_features)).cuda()
  with torch.no_grad():
    model[0].weight.copy_(torch.from_numpy(arrays["w1"].T.copy()).cuda())
    model[0].bias.copy_(torch.from_numpy(arrays["b1"][0].copy()).cuda())
    model[2].weight.copy_(torch.from_numpy(arrays["w2"].T.copy()).cuda())
    model[2].bias.copy_(torch.from_numpy(arrays["b2"][0].copy()).cuda())
  opt = torch.optim.SGD(model.parameters(), lr=config.lr)
  def step(x, y):
    xb = torch.from_numpy(x).cuda()
    yb = torch.from_numpy(y).cuda()
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    return float(loss.item())
  return step

def time_steps(step, x, y, config:StepConfig):
  for _ in range(config.warmup): step(x, y)
  a0 = bt.stage_alloc_count()
  samples = []
  for _ in range(config.iters):
    start = time.perf_counter()
    step(x, y)
    samples.append((time.perf_counter() - start) * 1e3)
  return samples, bt.stage_alloc_count() - a0

def summarize(samples):
  med = statistics.median(samples)
  s = sorted(samples)
  return {"median_ms": med, "stdev_ms": statistics.pstdev(samples),
          "p10_ms": s[len(s) // 10], "p90_ms": s[9 * len(s) // 10]}

def run(config:StepConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  x, y, arrays = make_data(config)
  bstep = bitsy_step_factory(arrays, config)
  tstep = torch_step_factory(arrays, config)
  clock0 = gpu_clock_mhz()
  bs, b_stage_allocs = time_steps(bstep, x, y, config)
  ts, _ = time_steps(tstep, x, y, config)
  clock1 = gpu_clock_mhz()
  bsum, tsum = summarize(bs), summarize(ts)
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "gpu_clock_mhz": (clock0 + clock1) / 2,
    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "bitsy": bt.version()},
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    "bitsy": bsum, "torch": tsum,
    "throughput_ratio_torch_over_bitsy": tsum["median_ms"] / bsum["median_ms"],
    "bitsy_stage_allocs_during_timed_steps": b_stage_allocs,
    "note": "wall time includes one completion wait per step on each side; "
            "stage_allocs is the host-visible staging allocation count during "
            "the 200 timed steps (after warmup)",
  }
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"mode={'deferred' if config.deferred else 'baseline'} clock={report['gpu_clock_mhz']:.0f} MHz")
  print(f"bitsy median {bsum['median_ms']:.3f} ms/step (spread {bsum['stdev_ms']:.3f}), "
        f"torch {tsum['median_ms']:.3f} ms/step (spread {tsum['stdev_ms']:.3f})")
  print(f"throughput ratio torch/bitsy = {report['throughput_ratio_torch_over_bitsy']:.3f} "
        f"(target >= 0.85 for bitsy/torch)")
  print(f"bitsy stage allocs during 200 timed steps: {b_stage_allocs}")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
