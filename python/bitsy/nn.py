"""Minimal nn surface (Module, Linear, activations, MSELoss) over Bitsy ops."""

from __future__ import annotations

import numpy as np

from . import _ctypes as C
from .autograd import Tensor, reduce_sum_op

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
    # exp_024: fused GEMM+bias epilogue replaces (x@W) + (ones@b) + add.
    ct = C.matmul_bias(x._t, self.W._t, self.b._t)
    Wt = C.transpose_view(self.W._t)
    xt = C.transpose_view(x._t)
    def back(g):
      return (g._op(C.matmul(g._t, Wt), (), None),
              g._op(C.matmul(xt, g._t), (), None),
              reduce_sum_op(g, 0, True))
    return x._op(ct, (x, self.W, self.b), back)

  def forward_tanh(self, x:Tensor) -> Tensor:
    """Fused forward for Linear followed by Tanh (exp_036)."""
    ct = C.matmul_bias_tanh(x._t, self.W._t, self.b._t)
    Wt = C.transpose_view(self.W._t)
    xt = C.transpose_view(x._t)
    def back(g):
      pre = g._op(C.tanh_backward(g._t, ct), (), None)  # dL/d(x@W+b)
      return (pre._op(C.matmul(pre._t, Wt), (), None),
              pre._op(C.matmul(xt, pre._t), (), None),
              reduce_sum_op(pre, 0, True))
    return x._op(ct, (x, self.W, self.b), back)

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
    layers = self.layers
    i = 0
    while i < len(layers):
      layer = layers[i]
      if (isinstance(layer, Linear) and i + 1 < len(layers) and
          isinstance(layers[i + 1], Tanh)):
        x = layer.forward_tanh(x)  # one dispatch instead of GEMM+bias + tanh
        i += 2
      else:
        x = layer(x)
        i += 1
    return x

def log_softmax(logits:Tensor) -> Tensor:
  # exp_027: one fused dispatch instead of exp + reduce + log + subtract.
  return logits.log_softmax()

def cross_entropy(logits:Tensor, onehot:Tensor) -> Tensor:
  # exp_028: one fused reduction kernel instead of mul + two reductions + scale.
  return logits.log_softmax().ce_loss(onehot)

def mse_loss(pred:Tensor, target:Tensor) -> Tensor:
  d = pred - target
  sq = d * d
  per_row = reduce_sum_op(sq, 1, True)
  return reduce_sum_op(per_row, 0, True) * 0.5
