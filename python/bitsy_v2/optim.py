"""SGD and AdamW — stateful optimizers over Parameter tensors.

Updates computed from grads (numpy) then written back as new leaf tensors via
the C runtime — no new C lines. Matches PyTorch update semantics.
"""
from __future__ import annotations
import numpy as np
from .tensor import Tensor


class SGD:
  def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
    self.params, self.lr, self.momentum, self.wd = params, lr, momentum, weight_decay
    self.v = {id(p): np.zeros_like(p.numpy()) for p in params}

  def zero_grad(self):
    for p in self.params: p.grad = None

  def step(self):
    for p in self.params:
      if p.grad is None: continue
      g = p.grad.astype(np.float32)
      if self.wd: g = g + self.wd * p.numpy()
      if self.momentum:
        self.v[id(p)] = self.momentum*self.v[id(p)] + g
        g = self.v[id(p)]
      new = (p.numpy() - self.lr * g).astype(np.float32)
      p.h = p.rt._raw_tensor(new, True)


class AdamW:
  def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
    self.params, self.lr, self.b1, self.b2 = params, lr, betas[0], betas[1]
    self.eps, self.wd = eps, weight_decay
    self.t = 0
    self.m = {id(p): np.zeros_like(p.numpy()) for p in params}
    self.v = {id(p): np.zeros_like(p.numpy()) for p in params}

  def zero_grad(self):
    for p in self.params: p.grad = None

  def step(self):
    self.t += 1
    for p in self.params:
      if p.grad is None: continue
      g = p.grad.astype(np.float32)
      self.m[id(p)] = self.b1*self.m[id(p)] + (1-self.b1)*g
      self.v[id(p)] = self.b2*self.v[id(p)] + (1-self.b2)*(g*g)
      mh = self.m[id(p)]/(1-self.b1**self.t)
      vh = self.v[id(p)]/(1-self.b2**self.t)
      up = mh/(np.sqrt(vh)+self.eps) + self.wd*p.numpy()
      new = (p.numpy() - self.lr*up).astype(np.float32)
      p.h = p.rt._raw_tensor(new, True)
