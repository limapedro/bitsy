"""CIFAR-10 small CNN: PyTorch CUDA vs Bitsy Vulkan loss-curve parity.

The dataset comes from the Hugging Face mirror (gitignored, regenerable). Same
seed, same init arrays, same batches, same cross-entropy objective.
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
class TrainConfig:
  train_n:int = 2048
  test_n:int = 512
  epochs:int = 3
  batch:int = 64
  lr:float = 0.05
  seed:int = 0
  result_path:Path = ROOT / "benchmarks" / "results" / "cifar_cnn_parity.json"

CONFIG = TrainConfig()

def load_data(config:TrainConfig):
  # Exported once from hf:uoft-cs/cifar10 by tools/export_cifar.py. Keeping the
  # datasets library out of this process avoids its Triton import collision.
  z = np.load(ROOT / "data" / "cifar_subset.npz")
  return (z["xtr"][:config.train_n], z["ytr"][:config.train_n]), \
         (z["xte"][:config.test_n], z["yte"][:config.test_n])

def init_arrays(config:TrainConfig):
  rng = np.random.default_rng(config.seed)
  return {
    "c1w": (rng.standard_normal((8, 3, 3, 3)) * 0.1).astype(np.float32),
    "c1b": np.zeros((8,), np.float32),
    "c2w": (rng.standard_normal((16, 8, 3, 3)) * 0.1).astype(np.float32),
    "c2b": np.zeros((16,), np.float32),
    "lw": (rng.standard_normal((4096, 10)) * 0.01).astype(np.float32),
    "lb": np.zeros((10,), np.float32),
  }

def torch_model(a, config:TrainConfig):
  torch.manual_seed(config.seed)
  m = torch.nn.Sequential(
    torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.Tanh(),
    torch.nn.Conv2d(8, 16, 3, padding=1, stride=2), torch.nn.Tanh(),
    torch.nn.Flatten(), torch.nn.Linear(4096, 10)).cuda()
  with torch.no_grad():
    m[0].weight.copy_(torch.from_numpy(a["c1w"]).cuda())
    m[0].bias.copy_(torch.from_numpy(a["c1b"]).cuda())
    m[2].weight.copy_(torch.from_numpy(a["c2w"]).cuda())
    m[2].bias.copy_(torch.from_numpy(a["c2b"]).cuda())
    m[5].weight.copy_(torch.from_numpy(a["lw"].T.copy()).cuda())
    m[5].bias.copy_(torch.from_numpy(a["lb"]).cuda())
  return m, torch.optim.SGD(m.parameters(), lr=config.lr)

def bitsy_model(a, config:TrainConfig):
  dev = "vulkan"
  rng = np.random.default_rng(config.seed)
  c1 = bt.nn.Conv2d(3, 8, rng, dev)
  c2 = bt.nn.Conv2d(8, 16, rng, dev, stride=2)
  fc = bt.nn.Linear(4096, 10, rng, dev)
  c1.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(a["c1w"], dev))
  c1.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(a["c1b"], dev))
  c2.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(a["c2w"], dev))
  c2.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(a["c2b"], dev))
  fc.W = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(a["lw"], dev))
  fc.b = bt.nn.Parameter(bt.autograd.Tensor.from_numpy(a["lb"].reshape(1, 10), dev))
  model = bt.nn.Sequential(c1, bt.nn.Tanh(), c2, bt.nn.Tanh(), Flatten(), fc)
  return model, bt.optim.SGD(model.parameters(), config.lr)

class Flatten(bt.nn.Module):
  def forward(self, x): return x.reshape(x.shape[0], 4096)

def train_torch(m, opt, x, y, config:TrainConfig, perm):
  xt = torch.from_numpy(x).cuda()
  losses = []
  for s in range(0, config.train_n, config.batch):
    idx = perm[s:s + config.batch]
    xb = xt[torch.from_numpy(idx.copy()).cuda()]
    yb = torch.from_numpy(y[idx].copy()).cuda()
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(m(xb), yb)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    losses.append(loss.item())
  return losses

def train_bitsy(m, opt, x, y, config:TrainConfig, perm):
  losses = []
  for s in range(0, config.train_n, config.batch):
    idx = perm[s:s + config.batch]
    xb = bt.autograd.Tensor.from_numpy(x[idx], "vulkan")
    onehot = np.zeros((len(idx), 10), np.float32)
    onehot[np.arange(len(idx)), y[idx]] = 1.0
    yb = bt.autograd.Tensor.from_numpy(onehot, "vulkan")
    opt.zero_grad()
    loss = bt.nn.cross_entropy(m(xb), yb)
    loss.backward()
    opt.step()
    losses.append(float(loss.numpy().ravel()[0]))
  return losses

def accuracy_torch(m, x, y, config:TrainConfig) -> float:
  with torch.no_grad():
    pred = m(torch.from_numpy(x[:config.test_n]).cuda())
    return float((pred.argmax(1).cpu() == torch.from_numpy(y[:config.test_n])).float().mean())

def bitsy_logits(m, x, batch:int=64):
  out = []
  for i in range(0, len(x), batch):
    out.append(m(bt.autograd.Tensor.from_numpy(x[i:i + batch], "vulkan")).numpy())
  return np.concatenate(out)

def accuracy_bitsy(m, x, y, config:TrainConfig) -> float:
  pred = bitsy_logits(m, x[:config.test_n])
  return float((pred.argmax(1) == y[:config.test_n]).mean())

def run(config:TrainConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  (xtr, ytr), (xte, yte) = load_data(config)
  a = init_arrays(config)
  tm, topt = torch_model(a, config)
  bm, bopt = bitsy_model(a, config)
  rng = np.random.default_rng(config.seed + 7)
  report:dict = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "bitsy": bt.version(), "bitsy_vulkan": bt.has_vulkan(),
                    "dataset": "hf:uoft-cs/cifar10"},
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    "epochs": [],
  }
  for epoch in range(config.epochs):
    perm = rng.permutation(config.train_n)
    tl = train_torch(tm, topt, xtr, ytr, config, perm)
    bl = train_bitsy(bm, bopt, xtr, ytr, config, perm)
    ta, ba = accuracy_torch(tm, xte, yte, config), accuracy_bitsy(bm, xte, yte, config)
    report["epochs"].append({
      "epoch": epoch + 1, "torch_loss": sum(tl) / len(tl), "bitsy_loss": sum(bl) / len(bl),
      "loss_abs_diff": abs(sum(tl) / len(tl) - sum(bl) / len(bl)),
      "torch_accuracy": ta, "bitsy_accuracy": ba,
      "torch_batch_losses": tl, "bitsy_batch_losses": bl,
    })
    print(f"epoch {epoch+1}: torch {sum(tl)/len(tl):.5f} acc {ta:.3f} | "
          f"bitsy {sum(bl)/len(bl):.5f} acc {ba:.3f}")
  wp = list(bm.parameters())[0].numpy()  # first conv weights
  tw = tm[0].weight.detach().cpu().numpy()
  report["final_conv1_w_max_abs_diff"] = float(np.abs(wp - tw).max())
  torch_params = list(tm.parameters())
  bitsy_params = list(bm.parameters())
  report["per_param_max_abs_diff"] = [
    float(np.abs(b.numpy() - (t.detach().cpu().numpy().T
                               if b.shape == t.shape[::-1] else t.detach().cpu().numpy())).max())
    for b, t in zip(bitsy_params, torch_params)
  ]
  with torch.no_grad():
    tl = tm(torch.from_numpy(xte[:config.test_n]).cuda()).cpu().numpy()
  bl = bitsy_logits(bm, xte[:config.test_n])
  report["final_logits_max_abs_diff"] = float(np.abs(tl - bl).max())
  report["final_argmax_agreement"] = float((tl.argmax(1) == bl.argmax(1)).mean())
  config.result_path.parent.mkdir(parents=True, exist_ok=True)
  config.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"final conv1 W diff: {report['final_conv1_w_max_abs_diff']:.3e}")
  print(f"report: {config.result_path}")
  return 0

if __name__ == "__main__": raise SystemExit(run())
