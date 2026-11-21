"""Minimal nn surface (Module, Linear, activations, MSELoss) over Bitsy ops."""

from __future__ import annotations

import numpy as np

from . import _ctypes as C
from .autograd import Tensor

class Parameter(Tensor):
  def __init__(self, tensor:Tensor) -> None:
    super().__init__(tensor._t, True, tensor.device)

class Module:
  def __call__(self, x:Tensor) -> Tensor: return self.forward(x)
  def parameters(self) -> list[Parameter]:
    out = []
    for value in vars(self).values():
      if isinstance(value, Parameter): out.append(value)
      elif isinstance(value, Module): out.extend(value.parameters())
      elif isinstance(value, (list, tuple)):
        for item in value:
          if isinstance(item, Module): out.extend(item.parameters())
    return out

class Linear(Module):
  def __init__(self, in_features:int, out_features:int, rng, device:str="cpu") -> None:
    w = (rng.standard_normal((in_features, out_features)) * 0.1).astype(np.float32)
    b = np.zeros((1, out_features), dtype=np.float32)
    self.W = Parameter(Tensor.from_numpy(w, device))
    self.b = Parameter(Tensor.from_numpy(b, device))
    self._ones:dict[int, Tensor] = {}

  def forward(self, x:Tensor) -> Tensor:
    n = x.shape[0]
    if n not in self._ones:
      self._ones[n] = Tensor.from_numpy(np.ones((n, 1), np.float32), x.device)
    return x @ self.W + self._ones[n] @ self.b

class Conv2d(Module):
  """NCHW 3x3 convolution with pad/stride; gradients run on the same device."""
  def __init__(self, in_ch:int, out_ch:int, rng, device:str="cpu", pad:int=1, stride:int=1) -> None:
    w = (rng.standard_normal((out_ch, in_ch, 3, 3)) * 0.1).astype(np.float32)
    self.W = Parameter(Tensor.from_numpy(w, device))
    self.b = Parameter(Tensor.from_numpy(np.zeros((out_ch,), np.float32), device))
    self.pad, self.stride = pad, stride

  def forward(self, x:Tensor) -> Tensor:
    ct = C.conv2d(x._t, self.W._t, self.b._t, self.pad, self.stride)
    def back(g):
      dx = C.conv2d_backward_x(g._t, self.W._t, x.shape, self.pad, self.stride)
      dw = C.conv2d_backward_w(x._t, g._t, self.W.shape, self.pad, self.stride)
      db = C.conv2d_backward_b(g._t, self.b.shape[0])
      return (Tensor(dx, device=x.device), Tensor(dw, device=x.device),
              Tensor(db, device=x.device))
    ctx = (back, (x, self.W, self.b))
    return Tensor(ct, True, x.device, ctx)

class Tanh(Module):
  def forward(self, x:Tensor) -> Tensor: return x.tanh()

class Sigmoid(Module):
  def forward(self, x:Tensor) -> Tensor: return x.sigmoid()

class ReLU(Module):
  def forward(self, x:Tensor) -> Tensor: return x.relu()

class Sequential(Module):
  def __init__(self, *layers:Module) -> None: self.layers = layers
  def forward(self, x:Tensor) -> Tensor:
    for layer in self.layers: x = layer(x)
    return x

def log_softmax(logits:Tensor) -> Tensor:
  col = Tensor.from_numpy(np.ones((logits.shape[1], 1), np.float32), logits.device)
  return logits - (logits.exp() @ col).log()

def cross_entropy(logits:Tensor, onehot:Tensor) -> Tensor:
  B = logits.shape[0]
  col = Tensor.from_numpy(np.ones((logits.shape[1], 1), np.float32), logits.device)
  row = Tensor.from_numpy(np.ones((1, B), np.float32), logits.device)
  return row @ ((-onehot * log_softmax(logits)) @ col) * (1.0 / B)

def mse_loss(pred:Tensor, target:Tensor) -> Tensor:
  d = pred - target
  sq = d * d
  col = Tensor.from_numpy(np.ones((sq.shape[1], 1), np.float32), sq.device)
  row = Tensor.from_numpy(np.ones((1, sq.shape[0]), np.float32), sq.device)
  return row @ (sq @ col) * 0.5
