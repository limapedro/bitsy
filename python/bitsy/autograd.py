"""Minimal dynamic autograd over the Bitsy C core (interim, see ARCHITECTURE_LOG).

Gradients are Bitsy tensors too, so every backward op runs on the same backend
as its forward op. No NumPy is used for compute, only for creating leaf data.
"""

from __future__ import annotations

import numpy as np

from . import _ctypes as C

def _scalar(value:float, like:"Tensor") -> "Tensor":
  return Tensor.from_numpy(np.array(value, dtype=np.float32), like.device)

def _unbroadcast_device(g:"Tensor", shape:tuple[int, ...]) -> "Tensor" | None:
  """Sum a gradient back via device matmul/view (exp_016: no host sync).

  A reduction along an axis is a matmul with a ones vector, which keeps the
  whole backward graph on the GPU so a training step can be captured as a
  single command buffer. Returns None for shapes this interim path cannot
  express (middle axes, >2D), which then use the CPU fallback below.
  """
  out = g
  while len(out.shape) > len(shape):
    if out.shape[0] == 1:
      out = out.reshape(*out.shape[1:])
    else:
      ones = Tensor.from_numpy(np.ones((1, out.shape[0]), np.float32), out.device)
      out = ones @ out
      out = out.reshape(*out.shape[1:])
  for axis, (got, want) in enumerate(zip(out.shape, shape)):
    if want == 1 and got != 1:
      if axis == len(out.shape) - 1:
        ones = Tensor.from_numpy(np.ones((got, 1), np.float32), out.device)
        out = out @ ones
      elif axis == 0:
        ones = Tensor.from_numpy(np.ones((1, got), np.float32), out.device)
        out = ones @ out
      else:
        return None
  return out if out.shape == shape else None

def _unbroadcast(g:"Tensor", shape:tuple[int, ...]) -> "Tensor":
  """Sum a gradient back to a parent's shape (2D on device, CPU fallback)."""
  if g.shape == shape: return g
  if len(g.shape) <= 2:
    dev = _unbroadcast_device(g, shape)
    if dev is not None: return dev
  arr = g.numpy()
  while arr.ndim > len(shape): arr = arr.sum(axis=0)
  for axis, (got, want) in enumerate(zip(arr.shape, shape)):
    if want == 1 and got != 1: arr = arr.sum(axis=axis, keepdims=True)
  return Tensor.from_numpy(arr.astype(np.float32), g.device)

class Tensor:
  """A tensor with a dynamic graph; compute is delegated to `_ctypes.Tensor`."""
  __slots__ = ("_t", "shape", "device", "requires_grad", "grad", "_ctx")

  def __init__(self, data, requires_grad:bool=False, device:str="cpu", _ctx=None) -> None:
    self._t = data if isinstance(data, C.Tensor) else C.Tensor.from_numpy(data, device)
    self.shape, self.device = self._t.shape, self._t.device
    self.requires_grad, self.grad, self._ctx = requires_grad, None, _ctx

  @classmethod
  def from_numpy(cls, array, device:str="cpu", requires_grad:bool=False) -> "Tensor":
    return cls(C.Tensor.from_numpy(array, device), requires_grad, device)

  def numpy(self): return self._t.numpy()
  def __repr__(self) -> str: return f"Tensor(shape={self.shape}, device={self.device}, requires_grad={self.requires_grad})"

  def _op(self, ct, parents, backward) -> "Tensor":
    ctx = (backward, parents) if any(p.requires_grad for p in parents) else None
    return Tensor(ct, ctx is not None, self.device, ctx)

  def _coerce(self, other) -> "Tensor":
    return other if isinstance(other, Tensor) else _scalar(float(other), self)

  def __add__(self, other) -> "Tensor":
    b = self._coerce(other)
    out = C.add(self._t, b._t)
    return self._op(out, (self, b), lambda g: (g, g))
  __radd__ = __add__

  def __sub__(self, other) -> "Tensor":
    b = self._coerce(other)
    out = C.sub(self._t, b._t)
    return self._op(out, (self, b), lambda g: (g, self._op(C.mul(g._t, _scalar(-1, self)._t), (), None)))
  __rsub__ = lambda self, other: self._coerce(other).__sub__(self)

  def __mul__(self, other) -> "Tensor":
    b = self._coerce(other)
    out = C.mul(self._t, b._t)
    return self._op(out, (self, b), lambda g: (self._op(C.mul(g._t, b._t), (), None),
                                               self._op(C.mul(g._t, self._t), (), None)))
  __rmul__ = __mul__

  def __truediv__(self, other) -> "Tensor":
    b = self._coerce(other)
    out = C.div(self._t, b._t)
    return self._op(out, (self, b), lambda g: (self._op(C.div(g._t, b._t), (), None),))

  def __matmul__(self, other) -> "Tensor":
    out = C.matmul(self._t, other._t)
    aT, bT = C.transpose(self._t), C.transpose(other._t)
    def back(g):
      return (self._op(C.matmul(g._t, bT), (), None),
              self._op(C.matmul(aT, g._t), (), None))
    return self._op(out, (self, other), back)

  def tanh(self) -> "Tensor":
    out = C.tanh(self._t)
    def back(g):
      one = _scalar(1, self)
      sq = self._op(C.mul(out, out), (), None)
      return (self._op(C.mul(g._t, self._op(C.sub(one._t, sq._t), (), None)._t), (), None),)
    return self._op(out, (self,), back)

  def sigmoid(self) -> "Tensor":
    out = C.sigmoid(self._t)
    def back(g):
      one = _scalar(1, self)
      d = self._op(C.mul(out, self._op(C.sub(one._t, out), (), None)._t), (), None)
      return (self._op(C.mul(g._t, d._t), (), None),)
    return self._op(out, (self,), back)

  def sum(self) -> "Tensor":
    n = 1
    for d in self.shape: n *= d
    col = Tensor.from_numpy(np.ones((n, 1), np.float32), self.device)
    return self.reshape(1, n) @ col

  def reshape(self, *shape) -> "Tensor":
    ct = C.view(self._t, tuple(shape))
    return self._op(ct, (self,), lambda g: (self._op(C.view(g._t, self.shape), (), None),))

  def exp(self) -> "Tensor":
    out = C.exp(self._t)
    return self._op(out, (self,), lambda g: (self._op(C.mul(g._t, out), (), None),))

  def log(self) -> "Tensor":
    out = C.log(self._t)
    return self._op(out, (self,), lambda g: (self._op(C.div(g._t, self._t), (), None),))

  def __neg__(self) -> "Tensor": return _scalar(0, self) - self

  def relu(self) -> "Tensor":
    out = C.relu(self._t)
    return self._op(out, (self,), lambda g: (g,))

  def _topo(self) -> list["Tensor"]:
    seen, order = set(), []
    def visit(node):
      if id(node) in seen: return
      seen.add(id(node))
      if node._ctx:
        for parent in node._ctx[1]: visit(parent)
      order.append(node)
    visit(self)
    return order

  def backward(self, gradient=None) -> None:
    grad = gradient if gradient is not None else Tensor.from_numpy(np.ones(self.shape, np.float32), self.device)
    accum:dict[int, Tensor] = {id(self): grad}
    for node in reversed(self._topo()):
      if not node._ctx: continue
      fn, parents = node._ctx
      for parent, dg in zip(parents, fn(accum[id(node)])):
        if not parent.requires_grad: continue
        dg = _unbroadcast(dg, parent.shape)
        prev = accum.get(id(parent))
        accum[id(parent)] = self._op(C.add(prev._t, dg._t), (), None) if prev else dg
        parent.grad = accum[id(parent)]
    if self.requires_grad: self.grad = accum[id(self)]
