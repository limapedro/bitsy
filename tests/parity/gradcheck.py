"""Gradient parity: one tiny MLP, same inputs, PyTorch CUDA vs Bitsy Vulkan.

Writes tests/parity/reports/gradcheck.json with loss and every parameter
gradient together with max absolute differences.
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

def run() -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  rng = np.random.default_rng(7)
  xn = rng.standard_normal((4, 3)).astype(np.float32)
  wn = (rng.standard_normal((3, 2)) * 0.1).astype(np.float32)
  bn = np.zeros((1, 2), np.float32)
  target = np.zeros((4, 2), np.float32)
  target[np.arange(4), np.array([0, 1, 0, 1])] = 1.0

  xt = torch.tensor(xn, device="cuda")
  wt = torch.tensor(wn, device="cuda", requires_grad=True)
  btp = torch.tensor(bn, device="cuda", requires_grad=True)
  tt = torch.tensor(target, device="cuda")
  loss_t = 0.5 * ((torch.tanh(xt @ wt + torch.ones((4, 1), device="cuda") @ btp) - tt) ** 2).sum()
  loss_t.backward()

  xb = bt.autograd.Tensor.from_numpy(xn, "vulkan")
  W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(wn, "vulkan"))
  B = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(bn, "vulkan"))
  linear = bt.nn.Linear(3, 2, np.random.default_rng(0), "vulkan")
  linear.W, linear.b = W, B
  loss_b = bt.nn.mse_loss(bt.nn.Tanh()(linear(xb)),
                          bt.autograd.Tensor.from_numpy(target, "vulkan"))
  loss_b.backward()

  report = {
    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "bitsy": bt.version(),
    "loss_torch": loss_t.item(), "loss_bitsy": float(loss_b.numpy().ravel()[0]),
    "loss_abs_diff": abs(loss_t.item() - float(loss_b.numpy().ravel()[0])),
    "grads": {
      "W": {"torch": wt.grad.cpu().numpy().tolist(), "bitsy": W.grad.numpy().tolist(),
            "max_abs_diff": float(np.abs(W.grad.numpy() - wt.grad.cpu().numpy()).max())},
      "b": {"torch": btp.grad.cpu().numpy().tolist(), "bitsy": B.grad.numpy().tolist(),
            "max_abs_diff": float(np.abs(B.grad.numpy() - btp.grad.cpu().numpy()).max())},
    },
  }
  out = ROOT / "tests" / "parity" / "reports" / "gradcheck.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"loss diff {report['loss_abs_diff']:.3e}, "
        f"W grad diff {report['grads']['W']['max_abs_diff']:.3e}, "
        f"b grad diff {report['grads']['b']['max_abs_diff']:.3e}")
  print(f"report: {out}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
