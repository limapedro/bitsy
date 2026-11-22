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


def isclose(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> Tensor:
  """Elementwise |a-b| <= atol + rtol*|b| (PyTorch default rtol/atol)."""
  tol = Tensor.scalar(a.rt, atol) + Tensor.scalar(a.rt, rtol) * b.abs()
  return (a - b).abs().le(tol).where(Tensor.scalar(a.rt, 1.0),
                                      Tensor.scalar(a.rt, 0.0))


def allclose(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> Tensor:
  return isclose(a, b, rtol, atol).all()


# ---- PR #15: float classification + alias/hypot helpers ----
def _scalar(x: Tensor, v: float) -> Tensor:
  return Tensor.scalar(x.rt, v)


def isnan(x: Tensor) -> Tensor:
  """NaN != NaN is the standard IEEE NaN test."""
  return x.ne(x).where(_scalar(x, 1.0), _scalar(x, 0.0))


def isinf(x: Tensor) -> Tensor:
  inf = Tensor.from_numpy(x.rt, np.array([np.inf], dtype=np.float32))
  return x.abs().eq(inf).where(_scalar(x, 1.0), _scalar(x, 0.0))


def isfinite(x: Tensor) -> Tensor:
  return logical_not(logical_or(isnan(x), isinf(x)))


def nan_to_num(x: Tensor, nan: float = 0.0, posinf: float = 3.4028235e38,
               neginf: float = -3.4028235e38) -> Tensor:
  z = _scalar(x, 0.0)
  out = isnan(x).where(_scalar(x, nan), x)
  pinf = Tensor.from_numpy(x.rt, np.array([np.inf], dtype=np.float32))
  ninf = Tensor.from_numpy(x.rt, np.array([-np.inf], dtype=np.float32))
  out = out.eq(pinf).where(_scalar(x, posinf), out)
  out = out.eq(ninf).where(_scalar(x, neginf), out)
  return out


def greater(a: Tensor, b: Tensor) -> Tensor:
  return a.gt(b).where(_scalar(a, 1.0), _scalar(a, 0.0))


def less(a: Tensor, b: Tensor) -> Tensor:
  return a.lt(b).where(_scalar(a, 1.0), _scalar(a, 0.0))


def equal(a: Tensor, b: Tensor) -> Tensor:
  return a.eq(b).where(_scalar(a, 1.0), _scalar(a, 0.0))


def hypot(a: Tensor, b: Tensor) -> Tensor:
  return ((a * a + b * b).sqrt())


def copysign(a: Tensor, b: Tensor) -> Tensor:
  z = _scalar(a, 0.0)
  return b.lt(z).where(-a.abs(), a.abs())


# ---- PR #16: more losses (pure compositions) ----
def hinge_loss(pred: Tensor, target: Tensor) -> Tensor:
  """target in {-1,+1}: mean(max(0, 1 - pred*target))."""
  return (Tensor.scalar(pred.rt, 1.0) - pred * target).relu().mean()


def poisson_nll_loss(input: Tensor, target: Tensor, log_input: bool = True) -> Tensor:
  if log_input:
    return (input.exp() - target * input).mean()
  return (input - target * input.log()).mean()


def kl_div(input: Tensor, target: Tensor) -> Tensor:
  """target must be positive; mean(target*(log(target) - input))."""
  return (target * (target.log() - input)).mean()
