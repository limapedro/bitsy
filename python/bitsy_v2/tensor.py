"""Tensor — a thin lazy wrapper over the v0.2 compact C runtime.

Every op builds a C node; nothing executes until realize()/numpy()/backward().
All compute is in the C+Vulkan core — this file adds zero C lines.
"""
from __future__ import annotations

import ctypes as C
import numpy as np

# op-kind constants mirrored from bitsy_rt.h
(SUM, MEAN, MAX, MIN) = (0, 1, 2, 3)


class Tensor:
  __slots__ = ("rt", "h", "shape", "requires_grad", "grad")

  def __init__(self, rt, h, shape, requires_grad=False):
    self.rt, self.h = rt, h
    self.shape = tuple(shape)
    self.requires_grad = requires_grad
    self.grad = None

  # ---- construction ----
  @staticmethod
  def from_numpy(rt, arr, requires_grad=False) -> "Tensor":
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return Tensor(rt, rt._raw_tensor(a, requires_grad), a.shape, requires_grad)

  @staticmethod
  def zeros(rt, shape) -> "Tensor":
    return Tensor.from_numpy(rt, np.zeros(shape, dtype=np.float32))

  @staticmethod
  def scalar(rt, v: float) -> "Tensor":
    return Tensor.from_numpy(rt, np.array([v], dtype=np.float32))

  # ---- execution / data ----
  def realize(self) -> "Tensor":
    self.rt.lib.rt_realize(self.rt.ctx, self.h)
    return self

  def numpy(self) -> np.ndarray:
    return self.rt.get(self.h, self.shape).reshape(self.shape)

  def backward(self, params=None) -> None:
    self.rt.backward(self.h)
    # materialize grads into the given leaf parameters (or all registered ones)
    leaves = params if params is not None else getattr(self.rt, "leaves", [])
    for leaf in leaves:
      if getattr(leaf, "requires_grad", False):
        leaf.grad = self.rt.grad(leaf.h, leaf.shape).reshape(leaf.shape)

  # ---- dtype/format helpers ----
  def __repr__(self):
    return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

  @staticmethod
  def _coerce(rt, x):
    return x if isinstance(x, Tensor) else Tensor.scalar(rt, float(x))

  @staticmethod
  def _bshape(a, b):
    return tuple(np.broadcast_shapes(a, b))

  # ---- arithmetic ----
  def _binary(self, name, other):
    o = Tensor._coerce(self.rt, other)
    shape = Tensor._bshape(self.shape, o.shape)
    return Tensor(self.rt, self.rt.binary(name, self.h, o.h), shape,
                  self.requires_grad or o.requires_grad)

  def __add__(self, o): return self._binary("add", o)
  def __radd__(self, o): return Tensor._coerce(self.rt, o)._binary("add", self)
  def __sub__(self, o): return self._binary("sub", o)
  def __rsub__(self, o): return Tensor._coerce(self.rt, o)._binary("sub", self)
  def __mul__(self, o): return self._binary("mul", o)
  def __rmul__(self, o): return Tensor._coerce(self.rt, o)._binary("mul", self)
  def __truediv__(self, o): return self._binary("div", o)
  def __rtruediv__(self, o): return Tensor._coerce(self.rt, o)._binary("div", self)
  def __pow__(self, o): return self.pow(o)

  def _unary(self, name):
    return Tensor(self.rt, self.rt.unary(name, self.h), self.shape, self.requires_grad)

  def __neg__(self): return self._unary("neg")
  def neg(self): return self._unary("neg")
  def exp(self): return self._unary("exp")
  def log(self): return self._unary("log")
  def sqrt(self): return self._unary("sqrt")
  def tanh(self): return self._unary("tanh")
  def sigmoid(self): return self._unary("sigm")
  def relu(self): return self._unary("relu")
  def gelu(self): return self._unary("gelu")
  def abs(self): return self._unary("abs")

  def pow(self, o): return self._binary("pow", o)
  def maximum(self, o): return self._binary("maximum", o)
  def minimum(self, o): return self._binary("minimum", o)

  # ---- comparisons / logical (PR #9: C primitives already exist) ----
  def gt(self, o): return self._binary("gt", o)
  def ge(self, o): return self._binary("ge", o)
  def lt(self, o): return self._binary("lt", o)
  def le(self, o): return self._binary("le", o)
  def eq(self, o): return self._binary("eq", o)
  def ne(self, o): return self._binary("ne", o)

  def where(self, a, b):
    cond, x, y = self, Tensor._coerce(self.rt, a), Tensor._coerce(self.rt, b)
    shape = Tensor._bshape(Tensor._bshape(self.shape, x.shape), y.shape)
    h = self.rt.where(cond.h, x.h, y.h)
    return Tensor(self.rt, h, shape,
                  self.requires_grad or x.requires_grad or y.requires_grad)

  # ---- activations (PR #9: pure compositions over existing primitives) ----
  def softplus(self) -> "Tensor":
    # stable: max(x,0) + log(1 + exp(-|x|))
    return self.maximum(Tensor.scalar(self.rt, 0.0)) + (
      Tensor.scalar(self.rt, 1.0) + self.abs().neg().exp()).log()

  def silu(self) -> "Tensor":
    return self * self.sigmoid()

  def mish(self) -> "Tensor":
    return self * self.softplus().tanh()

  def leaky_relu(self, negative_slope: float = 0.01) -> "Tensor":
    return self.gt(Tensor.scalar(self.rt, 0.0)).where(
      self, self * Tensor.scalar(self.rt, negative_slope))

  def elu(self, alpha: float = 1.0) -> "Tensor":
    return self.gt(Tensor.scalar(self.rt, 0.0)).where(
      self, (self.exp() - Tensor.scalar(self.rt, 1.0)) * Tensor.scalar(self.rt, alpha))

  def selu(self) -> "Tensor":
    alpha, scale = 1.6732632423543772, 1.0507009873554805
    return self.elu(alpha) * Tensor.scalar(self.rt, scale)

  def hardtanh(self, lo: float = -1.0, hi: float = 1.0) -> "Tensor":
    return self.minimum(Tensor.scalar(self.rt, hi)).maximum(Tensor.scalar(self.rt, lo))

  def hardswish(self) -> "Tensor":
    z = Tensor.scalar(self.rt, 0.0)
    return self.le(Tensor.scalar(self.rt, -3.0)).where(z,
      self.ge(Tensor.scalar(self.rt, 3.0)).where(
        self, self * (self + Tensor.scalar(self.rt, 3.0)) * Tensor.scalar(self.rt, 1.0/6.0)))

  def square(self) -> "Tensor": return self * self
  def reciprocal(self) -> "Tensor": return Tensor.scalar(self.rt, 1.0) / self
  def rsqrt(self) -> "Tensor": return self.sqrt().reciprocal()
  def exp2(self) -> "Tensor": return (self * Tensor.scalar(self.rt, np.log(2.0))).exp()
  def log2(self) -> "Tensor": return self.log() * Tensor.scalar(self.rt, 1.0/np.log(2.0))
  def clamp(self, lo: float, hi: float) -> "Tensor": return self.hardtanh(lo, hi)

  def lerp(self, end: "Tensor", weight) -> "Tensor":
    w = Tensor._coerce(self.rt, weight)
    return self + (end - self) * w

  # ---- reductions / movement ----
  def sum(self, axis=0, keepdim=False) -> "Tensor":
    h = self.rt.reduce(self.h, SUM, axis, 1 if keepdim else 0)
    shape = list(self.shape)
    if keepdim: shape[axis] = 1
    else: shape.pop(axis)
    return Tensor(self.rt, h, tuple(shape) or (1,), self.requires_grad)

  def mean(self, axis=0, keepdim=False) -> "Tensor":
    h = self.rt.reduce(self.h, MEAN, axis, 1 if keepdim else 0)
    shape = list(self.shape)
    if keepdim: shape[axis] = 1
    else: shape.pop(axis)
    return Tensor(self.rt, h, tuple(shape) or (1,), self.requires_grad)

  def max(self, axis=0, keepdim=False) -> "Tensor":
    h = self.rt.reduce(self.h, MAX, axis, 1 if keepdim else 0)
    shape = list(self.shape)
    if keepdim: shape[axis] = 1
    else: shape.pop(axis)
    return Tensor(self.rt, h, tuple(shape) or (1,), self.requires_grad)

  # ---- more reductions (PR #9) ----
  def var(self, axis=0, unbiased: bool = True) -> "Tensor":
    d = self - self.mean(axis=axis, keepdim=True)
    n = self.shape[axis]
    denom = (n - 1) if unbiased else n
    return (d * d).sum(axis=axis) * Tensor.scalar(self.rt, 1.0 / denom)

  def std(self, axis=0, unbiased: bool = True) -> "Tensor":
    return self.var(axis=axis, unbiased=unbiased).sqrt()

  def logsumexp(self, axis=0) -> "Tensor":
    m = self.max(axis=axis, keepdim=True)
    sh = (self - m).exp().sum(axis=axis, keepdim=True)
    return m + sh.log()

  def norm(self, p: float = 2.0, axis=0) -> "Tensor":
    return ((self.abs() ** Tensor.scalar(self.rt, p)).sum(axis=axis)
            ** Tensor.scalar(self.rt, 1.0 / p))

  def trans(self) -> "Tensor":
    return Tensor(self.rt, self.rt.transpose(self.h), self.shape[::-1], self.requires_grad)

  def reshape(self, *shape) -> "Tensor":
    return Tensor(self.rt, self.rt.reshape(self.h, shape), shape, self.requires_grad)

  def T(self): return self.trans()

  # ---- fused compositions (zero new C) ----
  def softmax(self, axis=-1) -> "Tensor":
    if axis < 0: axis += len(self.shape)
    m = self.max(axis=axis, keepdim=True)
    e = (self - m).exp()
    return e / e.sum(axis=axis, keepdim=True)

  def log_softmax(self, axis=-1) -> "Tensor":
    if axis < 0: axis += len(self.shape)
    m = self.max(axis=axis, keepdim=True)
    return self - m - (self - m).exp().sum(axis=axis, keepdim=True).log()

  def matmul(self, other: "Tensor") -> "Tensor":
    # (M,K) @ (K,N) via gemm; shape inferred
    M, K = self.shape[-2], self.shape[-1]
    N = other.shape[-1]
    return Tensor(self.rt, self.rt.gemm(self.h, other.h),
                  (M, N), self.requires_grad or other.requires_grad)

  def linear(self, weight: "Tensor", bias: "Tensor" = None) -> "Tensor":
    # weight stored (out,in); y = x @ weight.T + bias
    return self.matmul(weight.T()) + (bias if bias is not None else 0)
