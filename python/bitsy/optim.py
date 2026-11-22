"""SGD over Bitsy parameters; updates are computed on the same device."""

from __future__ import annotations

import numpy as np

from . import _ctypes as C
from .autograd import Tensor
from .nn import Parameter

class AdamW:
  """AdamW matching torch.optim.AdamW (defaults betas=(0.9,0.999), eps=1e-8)."""
  def __init__(self, params:list, lr:float=1e-3, betas:tuple=(0.9, 0.999),
               eps:float=1e-8, weight_decay:float=0.01) -> None:
    self.params, self.lr, self.betas, self.eps = params, lr, betas, eps
    self.weight_decay = weight_decay
    self.t = 0
    self.state = {}
    for p in params:
      dev = p._t.device
      shape = p._t.shape
      self.state[id(p)] = (
        Tensor.from_numpy(np.zeros(shape, np.float32), dev),
        Tensor.from_numpy(np.zeros(shape, np.float32), dev),
      )
    # autograd.Tensor wrappers (raw C tensors have no arithmetic operators)
    self._wrapped = {id(p): Tensor(p._t, False, p._t.device) for p in params}

  def zero_grad(self) -> None:
    for p in self.params: p.grad = None

  def step(self) -> None:
    self.t += 1
    b1, b2 = self.betas
    c1 = float(1.0 / (1.0 - b1 ** self.t))
    c2 = float(1.0 / (1.0 - b2 ** self.t))
    for p in self.params:
      if p.grad is None: continue
      m, v = self.state[id(p)]
      pt = self._wrapped[id(p)]
      g = p.grad
      m = b1 * m + (1.0 - b1) * g
      v = b2 * v + (1.0 - b2) * (g * g)
      self.state[id(p)] = (m, v)
      mhat, vhat = c1 * m, c2 * v
      denom = vhat.sqrt() + self.eps
      new_t = pt - self.lr * (mhat / denom + self.weight_decay * pt)
      self._wrapped[id(p)] = new_t
      p._t = new_t._t

class SGD:
  def __init__(self, params:list[Parameter], lr:float) -> None:
    self.params, self.lr = params, lr

  def zero_grad(self) -> None:
    for p in self.params: p.grad = None

  def step(self) -> None:
    if 0 < len(self.params) <= 4:
      # exp_029: all parameter updates in one dispatch.
      ws = [p._t for p in self.params]
      gs = [p.grad._t if p.grad is not None else None for p in self.params]
      outs = C.sgd_step_fused(ws + [None] * (4 - len(ws)),
                              gs + [None] * (4 - len(gs)), self.lr)
      for p, o in zip(self.params, outs):
        if o is not None: p._t = o
        p.grad = None
      return
    for p in self.params:
      if p.grad is None: continue
      p._t = C.sgd_step(p._t, p.grad._t, self.lr)
      p.grad = None
