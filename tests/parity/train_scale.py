"""Training-scale gradient parity gate: one real MLP step, Bitsy vs PyTorch.

Uses the training configuration (batch 64, 784->64->10) because the small
gradient-flow case (batch 8, 32->16->8) missed exp_019's and exp_033's
training-scale gradient failures. Compares loss, all four parameter gradients
and post-step parameters. Required case in run.py and CI.

Dataclass config, no argparse.
"""

from __future__ import annotations

import json, sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

@dataclass(frozen=True, slots=True)
class TrainScaleConfig:
  in_features:int = 784
  hidden:int = 64
  out_features:int = 10
  batch:int = 64
  lr:float = 0.1
  seed:int = 0
  rtol:float = 1e-4
  atol:float = 1e-5
  report_path:Path = ROOT / "tests" / "parity" / "reports" / "train_scale.json"

CONFIG = TrainScaleConfig()

def make_data(config:TrainScaleConfig):
  rng = np.random.default_rng(config.seed)
  x = rng.standard_normal((config.batch, config.in_features)).astype(np.float32)
  onehot = np.zeros((config.batch, config.out_features), np.float32)
  onehot[np.arange(config.batch), rng.integers(0, config.out_features, config.batch)] = 1.0
  w1 = (rng.standard_normal((config.in_features, config.hidden)) * 0.1).astype(np.float32)
  b1 = np.zeros((1, config.hidden), np.float32)
  w2 = (rng.standard_normal((config.hidden, config.out_features)) * 0.1).astype(np.float32)
  b2 = np.zeros((1, config.out_features), np.float32)
  return x, onehot, {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

def bitsy_step(config:TrainScaleConfig, arrays, x, onehot):
  dev = "vulkan" if bt.has_vulkan() else "cpu"
  m1 = bt.nn.Linear(config.in_features, config.hidden, np.random.default_rng(config.seed), dev)
  m2 = bt.nn.Linear(config.hidden, config.out_features, np.random.default_rng(config.seed), dev)
  m1.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w1"], dev))
  m1.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b1"], dev))
  m2.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w2"], dev))
  m2.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b2"], dev))
  model = bt.nn.Sequential(m1, bt.nn.Tanh(), m2)
  xb = bt.autograd.Tensor.from_numpy(x, dev)
  yb = bt.autograd.Tensor.from_numpy(onehot, dev)
  loss = bt.nn.cross_entropy(model(xb), yb)
  loss.backward()
  named = {"w1": m1.W, "b1": m1.b, "w2": m2.W, "b2": m2.b}
  grads = {k: v.grad.numpy().copy() for k, v in named.items()}
  bt.optim.SGD(model.parameters(), config.lr).step()
  bt.flush(); bt.wait()
  after = {k: v._t.numpy().copy() for k, v in named.items()}
  return float(loss.numpy().ravel()[0]), grads, after

def torch_step(config:TrainScaleConfig, arrays, x, onehot, device):
  torch.manual_seed(config.seed)
  model = torch.nn.Sequential(
    torch.nn.Linear(config.in_features, config.hidden), torch.nn.Tanh(),
    torch.nn.Linear(config.hidden, config.out_features)).to(device)
  with torch.no_grad():
    model[0].weight.copy_(torch.from_numpy(arrays["w1"].T.copy()).to(device))
    model[0].bias.copy_(torch.from_numpy(arrays["b1"][0].copy()).to(device))
    model[2].weight.copy_(torch.from_numpy(arrays["w2"].T.copy()).to(device))
    model[2].bias.copy_(torch.from_numpy(arrays["b2"][0].copy()).to(device))
  xb = torch.from_numpy(x).to(device)
  yb = torch.from_numpy(onehot).to(device)
  loss = torch.nn.functional.cross_entropy(model(xb), yb)
  loss.backward()
  named = {"w1": model[0].weight, "b1": model[0].bias,
           "w2": model[2].weight, "b2": model[2].bias}
  grads = {k: v.grad.detach().cpu().numpy().copy() for k, v in named.items()}
  torch.optim.SGD(model.parameters(), lr=config.lr).step()
  if device.type == "cuda": torch.cuda.synchronize()
  after = {k: v.detach().cpu().numpy().copy() for k, v in named.items()}
  return float(loss.item()), grads, after

def compare(config:TrainScaleConfig = CONFIG):
  x, onehot, arrays = make_data(config)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  try:
    bl, bg, ba = bitsy_step(config, arrays, x, onehot)
    tl, tg, ta = torch_step(config, arrays, x, onehot, device)
  except Exception as exc:
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": {"torch": torch.__version__, "torch_device": str(device),
                            "bitsy_vulkan": bt.has_vulkan()},
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "error": f"{type(exc).__name__}: {exc}", "checks": {"execution": False},
            "passed": False}
  def fix(name, arr): return arr.T if name.startswith("w") else arr
  checks = {"loss": bool(np.isclose(bl, tl, rtol=config.rtol, atol=config.atol))}
  details = {"loss": {"bitsy": bl, "torch": tl}}
  for name in ("w1", "b1", "w2", "b2"):
    gd = float(np.abs(fix(name, bg[name]) - tg[name]).max())
    pd = float(np.abs(fix(name, ba[name]) - ta[name]).max())
    checks[f"grad_{name}"] = bool(np.allclose(fix(name, bg[name]), tg[name],
                                               rtol=config.rtol, atol=config.atol))
    checks[f"param_{name}"] = bool(np.allclose(fix(name, ba[name]), ta[name],
                                                rtol=config.rtol, atol=config.atol))
    details[name] = {"grad_max_abs_diff": gd, "param_max_abs_diff": pd}
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"torch": torch.__version__, "torch_device": str(device),
                    "bitsy_vulkan": bt.has_vulkan()},
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    "checks": checks, "details": details, "error": None,
    "passed": all(checks.values()),
  }
  config.report_path.parent.mkdir(parents=True, exist_ok=True)
  config.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  return report

def run(config:TrainScaleConfig = CONFIG) -> int:
  report = compare(config)
  for name, ok in report["checks"].items():
    print(f"[{'PASS' if ok else 'FAIL'}] train-scale {name}")
  print(f"training-scale parity: {'PASS' if report['passed'] else 'FAIL'}")
  if report.get("error"): print(f"error: {report['error']}")
  print(f"report: {config.report_path}")
  return 0 if report["passed"] else 1

if __name__ == "__main__": raise SystemExit(run())
