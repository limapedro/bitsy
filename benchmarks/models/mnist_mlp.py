"""Train the same tiny MLP on MNIST with PyTorch CUDA and Bitsy Vulkan.

Same seed, same init arrays, same batches, same sigmoid+MSE objective. The
committed JSON is the evidence: per-epoch losses, accuracy and parameter drift.
"""

from __future__ import annotations

import json, sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

@dataclass(frozen=True, slots=True)
class TrainConfig:
  train_n:int = 4096
  test_n:int = 512
  epochs:int = 10
  batch:int = 64
  hidden:int = 64
  lr:float = 0.1
  seed:int = 0
  result_path:Path = ROOT / "benchmarks" / "results" / "mnist_mlp_parity.json"

CONFIG = TrainConfig()

def load_data(config:TrainConfig):
  train = MNIST(root=str(ROOT / "data"), train=True)
  test = MNIST(root=str(ROOT / "data"), train=False)
  def prep(ds, n):
    x = ds.data[:n].numpy().reshape(n, -1).astype(np.float32) / 255.0
    y = np.zeros((n, 10), np.float32)
    y[np.arange(n), ds.targets[:n].numpy()] = 1.0
    return x, y, ds.targets[:n].numpy()
  return prep(train, config.train_n), prep(test, config.test_n)

def init_arrays(config:TrainConfig):
  rng = np.random.default_rng(config.seed)
  return {
    "w1": (rng.standard_normal((784, config.hidden)) * 0.1).astype(np.float32),
    "b1": np.zeros((1, config.hidden), np.float32),
    "w2": (rng.standard_normal((config.hidden, 10)) * 0.1).astype(np.float32),
    "b2": np.zeros((1, 10), np.float32),
  }

def torch_model(arrays, config:TrainConfig):
  torch.manual_seed(config.seed)
  model = torch.nn.Sequential(
    torch.nn.Linear(784, config.hidden), torch.nn.Tanh(),
    torch.nn.Linear(config.hidden, 10)).cuda()
  with torch.no_grad():
    model[0].weight.copy_(torch.from_numpy(arrays["w1"].T.copy()).cuda())
    model[0].bias.copy_(torch.from_numpy(arrays["b1"][0].copy()).cuda())
    model[2].weight.copy_(torch.from_numpy(arrays["w2"].T.copy()).cuda())
    model[2].bias.copy_(torch.from_numpy(arrays["b2"][0].copy()).cuda())
  return model, torch.optim.SGD(model.parameters(), lr=config.lr)

def bitsy_model(arrays, config:TrainConfig):
  dev = "vulkan"
  m1 = bt.nn.Linear(784, config.hidden, np.random.default_rng(config.seed), dev)
  m2 = bt.nn.Linear(config.hidden, 10, np.random.default_rng(config.seed), dev)
  m1.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w1"], dev))
  m1.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b1"], dev))
  m2.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["w2"], dev))
  m2.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(arrays["b2"], dev))
  model = bt.nn.Sequential(m1, bt.nn.Tanh(), m2)
  return model, bt.optim.SGD(model.parameters(), config.lr)

def train_torch(model, opt, x, y, labels, config:TrainConfig, perm):
  losses = []
  xt = torch.from_numpy(x).cuda()
  for start in range(0, config.train_n, config.batch):
    idx = torch.from_numpy(perm[start:start + config.batch].copy()).cuda()
    target = torch.from_numpy(y[idx.cpu().numpy()]).cuda()
    opt.zero_grad()
    pred = model(xt[idx])
    batch_labels = torch.from_numpy(labels[idx.cpu().numpy()]).cuda()
    loss = torch.nn.functional.cross_entropy(pred, batch_labels)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    losses.append(loss.item())
  return losses

def train_bitsy(model, opt, x, y, labels, config:TrainConfig, perm):
  losses = []
  for start in range(0, config.train_n, config.batch):
    idx = perm[start:start + config.batch]
    xb = bt.autograd.Tensor.from_numpy(x[idx], "vulkan")
    yb = bt.autograd.Tensor.from_numpy(y[idx], "vulkan")
    opt.zero_grad()
    pred = model(xb)
    loss = bt.nn.cross_entropy(pred, yb)
    loss.backward()
    opt.step()
    losses.append(float(loss.numpy().ravel()[0]))
  return losses

def accuracy_torch(model, x, labels, config:TrainConfig) -> float:
  with torch.no_grad():
    pred = model(torch.from_numpy(x[:config.test_n]).cuda())
    return float((pred.argmax(1).cpu() == torch.from_numpy(labels[:config.test_n])).float().mean())

def accuracy_bitsy(model, x, labels, config:TrainConfig) -> float:
  pred = model(bt.autograd.Tensor.from_numpy(x[:config.test_n], "vulkan")).numpy()
  return float((pred.argmax(1) == labels[:config.test_n]).mean())

def run(config:TrainConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  (xtr, ytr, ytr_labels), (xte, yte, yte_labels) = load_data(config)
  arrays = init_arrays(config)
  tmodel, topt = torch_model(arrays, config)
  bmodel, bopt = bitsy_model(arrays, config)
  rng = np.random.default_rng(config.seed + 123)
  report:dict = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "bitsy": bt.version(), "bitsy_vulkan": bt.has_vulkan()},
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    "epochs": [],
  }
  for epoch in range(config.epochs):
    perm = rng.permutation(config.train_n)
    tl = train_torch(tmodel, topt, xtr, ytr, ytr_labels, config, perm)
    bl = train_bitsy(bmodel, bopt, xtr, ytr, ytr_labels, config, perm)
    ta, ba = accuracy_torch(tmodel, xte, yte_labels, config), accuracy_bitsy(bmodel, xte, yte_labels, config)
    report["epochs"].append({
      "epoch": epoch + 1,
      "torch_loss": sum(tl) / len(tl), "bitsy_loss": sum(bl) / len(bl),
      "loss_abs_diff": abs(sum(tl) / len(tl) - sum(bl) / len(bl)),
      "torch_accuracy": ta, "bitsy_accuracy": ba,
      "torch_batch_losses": tl, "bitsy_batch_losses": bl,
    })
    print(f"epoch {epoch+1}: torch {sum(tl)/len(tl):.5f} acc {ta:.3f} | "
          f"bitsy {sum(bl)/len(bl):.5f} acc {ba:.3f}")
  wp = list(bmodel.parameters())[0].numpy()
  tw = tmodel[0].weight.detach().cpu().numpy().T
  report["final_w1_max_abs_diff"] = float(np.abs(wp - tw).max())
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"final W1 max diff: {report['final_w1_max_abs_diff']:.3e}")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
