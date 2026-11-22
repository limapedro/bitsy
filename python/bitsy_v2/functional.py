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


# ---- PR #9: additional losses (pure compositions) ----
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
  return (pred - target).abs().mean()


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
  """Huber loss, PyTorch default beta=1.0, reduction='mean'."""
  d = (pred - target).abs()
  b = Tensor.scalar(pred.rt, beta)
  return d.lt(b).where(
    0.5 * d * d * Tensor.scalar(pred.rt, 1.0 / beta),
    d - 0.5 * b).mean()


def bce_loss(input: Tensor, target: Tensor) -> Tensor:
  """Binary cross entropy on probabilities (inputs should be in (0,1))."""
  one = Tensor.scalar(input.rt, 1.0)
  return -(target * input.log() + (one - target) * (one - input).log()).mean()


def bce_with_logits_loss(logits: Tensor, target: Tensor) -> Tensor:
  """Numerically stable: max(x,0) - x*t + log(1 + exp(-|x|))."""
  t = Tensor._coerce(logits.rt, target)
  return (logits.maximum(Tensor.scalar(logits.rt, 0.0)) - logits * t
          + (Tensor.scalar(logits.rt, 1.0) + logits.abs().neg().exp()).log()).mean()


def nll_loss(log_probs: Tensor, labels: np.ndarray) -> Tensor:
  """Negative log likelihood; labels are integer class indices (host data)."""
  n = log_probs.shape[0]
  oh = np.zeros(log_probs.shape, dtype=np.float32)
  oh[np.arange(n), np.asarray(labels, dtype=np.int64).reshape(-1)] = 1.0
  target = Tensor.from_numpy(log_probs.rt, oh)
  return -(log_probs * target).sum(axis=0).sum(axis=0) * Tensor.scalar(log_probs.rt, 1.0 / n)


# ---- PR #9: logical helpers (0/1 float tensors) ----
def logical_not(x: Tensor) -> Tensor:
  z, o = Tensor.scalar(x.rt, 0.0), Tensor.scalar(x.rt, 1.0)
  return x.ne(z).where(z, o)


def logical_and(a: Tensor, b: Tensor) -> Tensor:
  z, o = Tensor.scalar(a.rt, 0.0), Tensor.scalar(a.rt, 1.0)
  return (a.ne(z) * b.ne(z)).gt(z).where(o, z)


def logical_or(a: Tensor, b: Tensor) -> Tensor:
  z, o = Tensor.scalar(a.rt, 0.0), Tensor.scalar(a.rt, 1.0)
  return (a.ne(z) + b.ne(z)).gt(z).where(o, z)


def logical_xor(a: Tensor, b: Tensor) -> Tensor:
  z, o = Tensor.scalar(a.rt, 0.0), Tensor.scalar(a.rt, 1.0)
  return a.ne(z).ne(b.ne(z)).where(o, z)
