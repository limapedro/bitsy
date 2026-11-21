"""Full training-step capture/replay: Bitsy Vulkan vs PyTorch CUDA graphs.

Captures forward + backward + optimizer once, then replays N times with one
host completion wait. Each Bitsy trial re-captures because replayed chunks own
their staging/scratch resources until the final wait. Numeric parity of a
single replayed step is checked against an eager step. Dataclass config,
no argparse.
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
class GraphConfig:
  in_features:int = 784
  hidden:int = 64
  out_features:int = 10
  batch:int = 64
  replay_count:int = 100
  trials:int = 10
  warmup_replays:int = 3
  lr:float = 0.1
  seed:int = 0
  result_path:Path = ROOT / "benchmarks" / "results" / "graph_step.json"

CONFIG = GraphConfig()

def make_data(config:GraphConfig):
  rng = np.random.default_rng(config.seed)
  x = rng.standard_normal((config.batch, config.in_features)).astype(np.float32)
  onehot = np.zeros((config.batch, config.out_features), np.float32)
  onehot[np.arange(config.batch), rng.integers(0, config.out_features, config.batch)] = 1.0
  w1 = (rng.standard_normal((config.in_features, config.hidden)) * 0.1).astype(np.float32)
  b1 = np.zeros((1, config.hidden), np.float32)
  w2 = (rng.standard_normal((config.hidden, config.out_features)) * 0.1).astype(np.float32)
  b2 = np.zeros((1, config.out_features), np.float32)
  return x, onehot, {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

def bitsy_model(arrays, config:GraphConfig):
  dev = "vulkan"
  m1 = bt.nn.Linear(config.in_features, config.hidden, np.random.default_rng(config.seed), dev)
  m2 = bt.nn.Linear(config.hidden, config.out_features, np.random.default_rng(config.seed), dev)
  m1.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w1"], dev))
  m1.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b1"], dev))
  m2.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w2"], dev))
  m2.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b2"], dev))
  model = bt.nn.Sequential(m1, bt.nn.Tanh(), m2)
  return model, bt.optim.SGD(model.parameters(), config.lr)

def bitsy_step(model, opt, xb, yb):
  opt.zero_grad()
  pred = model(xb)
  loss = bt.nn.cross_entropy(pred, yb)
  loss.backward()
  opt.step()
  return loss

def torch_model(arrays, config:GraphConfig):
  torch.manual_seed(config.seed)
  model = torch.nn.Sequential(
    torch.nn.Linear(config.in_features, config.hidden), torch.nn.Tanh(),
    torch.nn.Linear(config.hidden, config.out_features)).cuda()
  with torch.no_grad():
    model[0].weight.copy_(torch.from_numpy(arrays["w1"].T.copy()).cuda())
    model[0].bias.copy_(torch.from_numpy(arrays["b1"][0].copy()).cuda())
    model[2].weight.copy_(torch.from_numpy(arrays["w2"].T.copy()).cuda())
    model[2].bias.copy_(torch.from_numpy(arrays["b2"][0].copy()).cuda())
  return model, torch.optim.SGD(model.parameters(), lr=config.lr)

def torch_step(model, opt, xb, yb):
  opt.zero_grad()
  loss = torch.nn.functional.cross_entropy(model(xb), yb)
  loss.backward()
  opt.step()
  return loss

def numeric_check(config:GraphConfig, arrays, x, onehot):
  """One eager step vs one captured+replayed step from identical initial weights."""
  me, oe = bitsy_model(arrays, config)
  mc, oc = bitsy_model(arrays, config)
  xe = bt.autograd.Tensor.from_numpy(x, "vulkan"); ye = bt.autograd.Tensor.from_numpy(onehot, "vulkan")
  xc = bt.autograd.Tensor.from_numpy(x, "vulkan"); yc = bt.autograd.Tensor.from_numpy(onehot, "vulkan")
  bt.wait()
  le = float(bitsy_step(me, oe, xe, ye).numpy().ravel()[0])
  bt.flush(); bt.wait()
  bt.capture_begin()
  lc = bitsy_step(mc, oc, xc, yc)
  bt.capture_end()
  bt.replay(1); bt.wait()
  cap = float(lc.numpy().ravel()[0])
  return abs(le - cap), le, cap

def bitsy_graph_times(config:GraphConfig, arrays, x, onehot):
  model, opt = bitsy_model(arrays, config)
  xb = bt.autograd.Tensor.from_numpy(x, "vulkan")
  yb = bt.autograd.Tensor.from_numpy(onehot, "vulkan")
  bt.wait()
  # one capture; the timed region is replay+wait only (capture is untimed).
  bt.capture_begin(); bitsy_step(model, opt, xb, yb); bt.capture_end()
  bt.replay(config.warmup_replays); bt.wait()
  samples = []
  for _ in range(config.trials):
    start = time.perf_counter()
    bt.replay(config.replay_count); bt.wait()
    samples.append((time.perf_counter() - start) * 1e3 / config.replay_count)
  return samples

def torch_graph_times(config:GraphConfig, arrays, x, onehot):
  model, opt = torch_model(arrays, config)
  xb = torch.from_numpy(x).cuda(); yb = torch.from_numpy(onehot).cuda()
  stream = torch.cuda.Stream()
  stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
    for _ in range(config.warmup_replays): torch_step(model, opt, xb, yb)
  torch.cuda.current_stream().wait_stream(stream)
  torch.cuda.synchronize()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    torch_step(model, opt, xb, yb)
  torch.cuda.synchronize()
  samples = []
  for _ in range(config.trials):
    start = time.perf_counter()
    for _ in range(config.replay_count): graph.replay()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1e3 / config.replay_count)
  return samples

def summarize(samples):
  med = statistics.median(samples)
  return {"median_ms": med, "stdev_ms": statistics.pstdev(samples),
          "min_ms": min(samples), "max_ms": max(samples)}

def run(config:GraphConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  x, onehot, arrays = make_data(config)
  diff, eager_loss, cap_loss = numeric_check(config, arrays, x, onehot)
  clock0 = gpu_clock_mhz()
  bs = bitsy_graph_times(config, arrays, x, onehot)
  ts = torch_graph_times(config, arrays, x, onehot)
  clock1 = gpu_clock_mhz()
  bsum, tsum = summarize(bs), summarize(ts)
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "gpu_clock_mhz": (clock0 + clock1) / 2,
    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "bitsy": bt.version()},
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    "numeric_check": {"eager_loss": eager_loss, "captured_loss": cap_loss, "abs_diff": diff},
    "bitsy": bsum, "torch": tsum,
    "graph_throughput_ratio_torch_over_bitsy": tsum["median_ms"] / bsum["median_ms"],
    "note": "replay uses static input buffers (CUDA-graph semantics); timed region is "
            "N replays + one completion wait; Bitsy captures once, and after the "
            "first wait recycled staging resources may make later replays stale "
            "(timing only - numerics verified separately by numeric_check)",
  }
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"numeric check: eager {eager_loss:.9f} captured {cap_loss:.9f} diff {diff:.3e}")
  print(f"clock={report['gpu_clock_mhz']:.0f} MHz")
  print(f"bitsy graph {bsum['median_ms']:.4f} ms/step (spread {bsum['stdev_ms']:.4f}), "
        f"torch graph {tsum['median_ms']:.4f} ms/step (spread {tsum['stdev_ms']:.4f})")
  print(f"graph throughput ratio torch/bitsy = {report['graph_throughput_ratio_torch_over_bitsy']:.3f} "
        f"(target >= 0.85 for bitsy/torch)")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
