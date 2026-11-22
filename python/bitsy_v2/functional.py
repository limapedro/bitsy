"""Losses — pure compositions over compact primitives."""
from __future__ import annotations
import numpy as np
from .tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
  d = pred - target
  return (d * d).mean()


def cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
  """labels = integer class indices (host data)."""
  lp = logits.log_softmax(axis=-1)
  n = lp.shape[0]
  oh = np.zeros(lp.shape, dtype=np.float32)
  oh[np.arange(n), np.asarray(labels, dtype=np.int64).reshape(-1)] = 1.0
  target = Tensor.from_numpy(lp.rt, oh)
  # reduce over both axes to a scalar, then mean over batch (PyTorch default)
  total = (lp * target).sum(axis=0).sum(axis=0)  # scalar
  return -(total * (1.0 / n))
