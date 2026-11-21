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
    for p in self.params:
      if p.grad is None: continue
      # exp_017: fused W - lr*grad (one dispatch, lr as a push constant).
      p._t = C.sgd_step(p._t, p.grad._t, self.lr)
      p.grad = None
