"""SGD over Bitsy parameters; updates are computed on the same device."""

from __future__ import annotations

import numpy as np

from . import _ctypes as C
from .autograd import Tensor
from .nn import Parameter

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
