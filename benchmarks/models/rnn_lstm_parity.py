"""RNN and LSTM parity + performance: Bitsy Vulkan vs PyTorch CUDA.

Weights are copied from PyTorch into Bitsy (and vice versa through the gate),
so forward hidden/cell states, input grads and weight grads are directly
comparable. Dataclass config, no argparse; median/spread timing with warmup,
correct device sync and live GPU-clock logging.
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
from common.clock import gpu_clock_mhz  # noqa: E402
import bitsy as bt  # noqa: E402
from bitsy import nn  # noqa: E402
BTensor = bt.autograd.Tensor

@dataclass(frozen=True, slots=True)
class RNNConfig:
  batch:int = 3
  time:int = 4
  input_size:int = 5
  hidden_size:int = 6
  iters:int = 20
  warmup:int = 3
  tol:float = 1e-5
  result_path:Path = ROOT / "benchmarks" / "results" / "rnn_lstm_parity.json"

CONFIG = RNNConfig()

def _seed_data(cfg:RNNConfig):
  rng = np.random.default_rng(7)
  x = rng.standard_normal((cfg.batch, cfg.time, cfg.input_size)).astype(np.float32)
  y = rng.standard_normal((cfg.batch, cfg.hidden_size)).astype(np.float32)
  return x, y

def _torch_rnn(cfg:RNNConfig):
  return torch.nn.RNN(cfg.input_size, cfg.hidden_size, batch_first=True).cuda()

def _bitsy_rnn(cfg:RNNConfig, tmodel, dev:str):
  m = nn.RNN(cfg.input_size, cfg.hidden_size, dev, seed=0)
  m.cell.W_ih = nn.Parameter(BTensor.from_numpy(
      np.ascontiguousarray(tmodel.weight_ih_l0.detach().cpu().numpy().T), dev))
  m.cell.W_hh = nn.Parameter(BTensor.from_numpy(
      np.ascontiguousarray(tmodel.weight_hh_l0.detach().cpu().numpy().T), dev))
  b = (tmodel.bias_ih_l0.detach().cpu().numpy() +
       tmodel.bias_hh_l0.detach().cpu().numpy()).reshape(1, -1)
  m.cell.b = nn.Parameter(BTensor.from_numpy(np.ascontiguousarray(b), dev))
  return m

def rnn_parity(cfg:RNNConfig, x, y, dev:str) -> dict:
  trnn = _torch_rnn(cfg)
  brnn = _bitsy_rnn(cfg, trnn, dev)
  bx = BTensor.from_numpy(x, dev, True)
  _, outs = brnn(bx)
  nn.mse_loss(outs[-1], BTensor.from_numpy(y, dev)).backward()
  tx = torch.from_numpy(x.copy()).cuda().requires_grad_(True)
  tout, th = trnn(tx)
  torch.nn.functional.mse_loss(tout[:, -1, :], torch.from_numpy(y.copy()).cuda()).backward()
  hdiff = max(float(np.abs(o.numpy() - tout[:, t, :].detach().cpu().numpy()).max())
              for t, o in enumerate(outs))
  return {
    "h_max_abs_diff": hdiff,
    "dx_max_abs_diff": float(np.abs(bx.grad.numpy().reshape(x.shape) - tx.grad.detach().cpu().numpy()).max()),
    "dWih_max_abs_diff": float(np.abs(
        brnn.cell.W_ih.grad.numpy() - trnn.weight_ih_l0.grad.detach().cpu().numpy().T).max()),
  }

def _torch_lstm(cfg:RNNConfig):
  return torch.nn.LSTM(cfg.input_size, cfg.hidden_size, batch_first=True).cuda()

def _bitsy_lstm(cfg:RNNConfig, tmodel, dev:str):
  m = nn.LSTM(cfg.input_size, cfg.hidden_size, dev, seed=0)
  H = cfg.hidden_size
  wih = tmodel.weight_ih_l0.detach().cpu().numpy()
  whh = tmodel.weight_hh_l0.detach().cpu().numpy()
  bias = (tmodel.bias_ih_l0.detach().cpu().numpy() +
          tmodel.bias_hh_l0.detach().cpu().numpy())
  for name, arr in (("W_i", wih[0:H]), ("W_f", wih[H:2*H]),
                    ("W_g", wih[2*H:3*H]), ("W_o", wih[3*H:4*H])):
    setattr(m.cell, name, nn.Parameter(BTensor.from_numpy(np.ascontiguousarray(arr.T), dev)))
  for name, arr in (("U_i", whh[0:H]), ("U_f", whh[H:2*H]),
                    ("U_g", whh[2*H:3*H]), ("U_o", whh[3*H:4*H])):
    setattr(m.cell, name, nn.Parameter(BTensor.from_numpy(np.ascontiguousarray(arr.T), dev)))
  m.cell.b = nn.Parameter(BTensor.from_numpy(np.ascontiguousarray(bias.reshape(1, -1)), dev))
  return m

def lstm_parity(cfg:RNNConfig, x, y, dev:str) -> dict:
  tlstm = _torch_lstm(cfg)
  blstm = _bitsy_lstm(cfg, tlstm, dev)
  bx = BTensor.from_numpy(x, dev, True)
  (h, c), outs = blstm(bx)
  nn.mse_loss(outs[-1], BTensor.from_numpy(y, dev)).backward()
  tx = torch.from_numpy(x.copy()).cuda().requires_grad_(True)
  tout, (th, tc) = tlstm(tx)
  torch.nn.functional.mse_loss(tout[:, -1, :], torch.from_numpy(y.copy()).cuda()).backward()
  hdiff = max(float(np.abs(o.numpy() - tout[:, t, :].detach().cpu().numpy()).max())
              for t, o in enumerate(outs))
  return {
    "h_max_abs_diff": hdiff,
    "c_max_abs_diff": float(np.abs(c.numpy() - tc.detach().cpu().numpy()).max()),
    "dx_max_abs_diff": float(np.abs(bx.grad.numpy().reshape(x.shape) - tx.grad.detach().cpu().numpy()).max()),
  }

def _timed(fn, warmup:int, iters:int, sync) -> dict:
  for _ in range(warmup): sync()
  clock0 = gpu_clock_mhz()
  samples = []
  for _ in range(iters):
    st = time.perf_counter(); fn(); sync(); samples.append((time.perf_counter()-st)*1e3)
  return {"median_ms": float(statistics.median(samples)),
          "spread_ms": [float(min(samples)), float(max(samples))],
          "clock_mhz": (clock0 + gpu_clock_mhz()) * 0.5}

def run(cfg:RNNConfig = CONFIG) -> int:
  if not (torch.cuda.is_available() and bt.has_vulkan()):
    print("needs CUDA and Vulkan; skipping")
    return 0
  x, y = _seed_data(cfg)
  dev = "vulkan"
  rnn = rnn_parity(cfg, x, y, dev)
  lstm = lstm_parity(cfg, x, y, dev)
  report:dict = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "bitsy": bt.version(), "bitsy_vulkan": bt.has_vulkan()},
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    "rnn": rnn, "lstm": lstm,
  }
  def all_close(d): return all(v <= cfg.tol for v in d.values())
  report["passed"] = all_close(rnn) and all_close(lstm)
  # Forward performance (median/spread), separate fresh models.
  trnn, tlstm = _torch_rnn(cfg), _torch_lstm(cfg)
  brnn, blstm = _bitsy_rnn(cfg, trnn, dev), _bitsy_lstm(cfg, tlstm, dev)
  bx = BTensor.from_numpy(x, dev)
  tx = torch.from_numpy(x.copy()).cuda()
  report["rnn_forward_timing"] = {
    "bitsy": _timed(lambda: brnn(bx), cfg.warmup, cfg.iters, lambda: (bt.flush(), bt.wait())),
    "torch": _timed(lambda: trnn(tx), cfg.warmup, cfg.iters, torch.cuda.synchronize),
  }
  report["lstm_forward_timing"] = {
    "bitsy": _timed(lambda: blstm(bx), cfg.warmup, cfg.iters, lambda: (bt.flush(), bt.wait())),
    "torch": _timed(lambda: tlstm(tx), cfg.warmup, cfg.iters, torch.cuda.synchronize),
  }
  report["rnn_speed_ratio_torch_over_bitsy"] = (
      report["rnn_forward_timing"]["torch"]["median_ms"] /
      report["rnn_forward_timing"]["bitsy"]["median_ms"])
  report["lstm_speed_ratio_torch_over_bitsy"] = (
      report["lstm_forward_timing"]["torch"]["median_ms"] /
      report["lstm_forward_timing"]["bitsy"]["median_ms"])
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"RNN  h {rnn['h_max_abs_diff']:.3e} dx {rnn['dx_max_abs_diff']:.3e} dWih {rnn['dWih_max_abs_diff']:.3e}")
  print(f"LSTM h {lstm['h_max_abs_diff']:.3e} c {lstm['c_max_abs_diff']:.3e} dx {lstm['dx_max_abs_diff']:.3e}")
  print(f"passed={report['passed']} | RNN ratio {report['rnn_speed_ratio_torch_over_bitsy']:.2f} "
        f"LSTM ratio {report['lstm_speed_ratio_torch_over_bitsy']:.2f}")
  return 0 if report["passed"] else 1

if __name__ == "__main__": raise SystemExit(run())
