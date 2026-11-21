"""Conv2d forward/backward parity for both backends against PyTorch CUDA."""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

def run() -> int:
  rng = np.random.default_rng(3)
  x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
  w = (rng.standard_normal((4, 3, 3, 3)) * 0.2).astype(np.float32)
  b = (rng.standard_normal((4,)) * 0.1).astype(np.float32)
  out = {}
  for dev in ("cpu", "vulkan"):
    xt = torch.tensor(x, device="cuda", requires_grad=True)
    wt = torch.tensor(w, device="cuda", requires_grad=True)
    bt_ = torch.tensor(b, device="cuda", requires_grad=True)
    yt = torch.nn.functional.conv2d(xt, wt, bt_, padding=1, stride=2)
    torch.nn.functional.conv2d(xt, wt, bt_, padding=1, stride=2).square().sum().backward()

    X = bt.autograd.Tensor.from_numpy(x, dev, True)
    W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(w, dev))
    B = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(b, dev))
    conv = bt.nn.Conv2d(3, 4, np.random.default_rng(0), dev, stride=2)
    conv.W, conv.b = W, B
    y = conv(X)
    (y * y).sum().backward()
    out[dev] = {
      "forward": float(np.abs(y.numpy() - yt.detach().cpu().numpy()).max()),
      "dW": float(np.abs(W.grad.numpy() - wt.grad.cpu().numpy()).max()),
      "db": float(np.abs(B.grad.numpy() - bt_.grad.cpu().numpy()).max()),
      "dX": float(np.abs(X.grad.numpy() - xt.grad.cpu().numpy()).max()),
    }
    print(dev, out[dev])
  report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "bitsy": bt.version(), "results": out}
  path = ROOT / "tests" / "parity" / "reports" / "conv_check.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"report: {path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
