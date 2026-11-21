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
      out = reduce_sum_op(out, 0, False)
  for axis, (got, want) in enumerate(zip(out.shape, shape)):
    if want == 1 and got != 1:
      if axis == len(out.shape) - 1:
        out = reduce_sum_op(out, 1, True)
      elif axis == 0:
        out = reduce_sum_op(out, 0, True)
      else:
        return None
  return out if out.shape == shape else None

def reduce_sum_op(t:"Tensor", axis:int, keepdim:bool) -> "Tensor":
  """Autograd-aware 2D reduction via the exp_018 dedicated kernel."""
  out = C.reduce_sum(t._t, axis, keepdim)
  def back(g):
    shp = list(t.shape)
    shp[axis] = 1
    gg = g.reshape(*shp)
    # exp_019: zero-dispatch broadcast view; reduce_sum is stride-aware now.
    # exp_023: zero-dispatch broadcast view was retried after the offset fix
    # and STILL failed the gradient gate; keep the dedicated kernel.
    return (Tensor(C.broadcast_grad(gg._t, t.shape), False, t.device),)
  return t._op(out, (t,), back)

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
    # exp_022: a fused dual-output gradient kernel was tried here; it passed
    # isolated checks but broke end-to-end gradients (the 43-case gate caught
    # it), so the unfused path is restored. See FINDINGS.
    return self._op(out, (self, b), lambda g: (self._op(C.mul(g._t, b._t), (), None),
                                               self._op(C.mul(g._t, self._t), (), None)))
  __rmul__ = __mul__

  def __truediv__(self, other) -> "Tensor":
    b = self._coerce(other)
    out = C.div(self._t, b._t)
    return self._op(out, (self, b), lambda g: (self._op(C.div(g._t, b._t), (), None),))

  def __matmul__(self, other) -> "Tensor":
    out = C.matmul(self._t, other._t)
    # exp_017: zero-dispatch strided views instead of materialising transposes.
    aT, bT = C.transpose_view(self._t), C.transpose_view(other._t)
    def back(g):
      return (self._op(C.matmul(g._t, bT), (), None),
              self._op(C.matmul(aT, g._t), (), None))
    return self._op(out, (self, other), back)

  def tanh(self) -> "Tensor":
    out = C.tanh(self._t)
    def back(g):
      # exp_019: one fused dispatch instead of square + subtract + mul.
      return (self._op(C.tanh_backward(g._t, out), (), None),)
    return self._op(out, (self,), back)

  def sigmoid(self) -> "Tensor":
    out = C.sigmoid(self._t)
    def back(g):
      one = _scalar(1, self)
      d = self._op(C.mul(out, self._op(C.sub(one._t, out), (), None)._t), (), None)
      return (self._op(C.mul(g._t, d._t), (), None),)
    return self._op(out, (self,), back)

  def ce_loss(self, onehot:"Tensor") -> "Tensor":
    out = C.ce_loss(self._t, onehot._t)
    def back(g):
      scale = -1.0 / self.shape[0]
      return (g * onehot * scale, None)
    return self._op(out, (self, onehot), back)

  def log_softmax(self) -> "Tensor":
    out = C.log_softmax(self._t)
    def back(g):
      # d/dx = g - exp(y) * sum_cols(g)
      s = reduce_sum_op(g, 1, True)
      e = self._op(C.exp(out), (), None)
      return (g - e * s,)
    return self._op(out, (self,), back)

  def sum(self) -> "Tensor":
    n = 1
    for d in self.shape: n *= d
    # exp_018: dedicated reduction instead of a ones-vector GEMM.
    return reduce_sum_op(self.reshape(1, n), 1, True)

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
