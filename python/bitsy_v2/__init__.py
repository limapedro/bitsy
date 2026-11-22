"""bitsy_v2 — lean Python nn/optim over the compact C+Vulkan core.

All layers are pure Python compositions; zero new C lines. fp32, lazy, functional
style. Complexity lives in `bitsy_rt` (C+Vulkan).
"""
from __future__ import annotations

import ctypes as C
import numpy as np
from pathlib import Path
from .tensor import Tensor
from . import nn, optim, functional

__all__ = ["Engine", "Tensor", "nn", "optim", "functional"]


class _RT:
  """Adapter exposing the methods Tensor expects, backed by ctypes."""
  def __init__(self, lib, device):
    self.lib, self.ctx = lib, lib.rt_create(device)
    self.leaves = []
  def __del__(self): self.lib.rt_destroy(self.ctx)
  def _raw_tensor(self, arr, requires_grad):
    h = self.lib.rt_from_host(self.ctx,
      arr.ctypes.data_as(C.POINTER(C.c_float)), arr.ndim,
      (C.c_int64*arr.ndim)(*arr.shape), 1 if requires_grad else 0)
    return h
  tensor = _raw_tensor
  def get(self, h, shape):
    out = np.zeros(shape, dtype=np.float32)
    self.lib.rt_to_host(h, out.ctypes.data_as(C.POINTER(C.c_float)))
    return out
  def binary(self, name, a, b): return getattr(self.lib, f"rt_{name}")(self.ctx, a, b)
  def unary(self, name, a): return getattr(self.lib, f"rt_{name}")(self.ctx, a)
  def gemm(self, a, b, bias=None, ta=0, tb=0):
    return self.lib.rt_gemm(self.ctx, a, b, bias or C.c_void_p(), ta, tb)
  def reduce(self, a, kind, axis, keepdim): return self.lib.rt_reduce(self.ctx, a, kind, axis, keepdim)
  def where(self, c, a, b): return self.lib.rt_where(self.ctx, c, a, b)
  def transpose(self, a): return self.lib.rt_transpose(self.ctx, a)
  def reshape(self, a, shape):
    return self.lib.rt_reshape(self.ctx, a, len(shape), (C.c_int64*len(shape))(*shape))
  def backward(self, loss): self.lib.rt_backward(self.ctx, loss)
  def grad(self, t, shape): return self.get(self.lib.rt_grad(t), shape)


class Engine:
  """Owns the C runtime context and the shared library. One per device."""
  CPU, VULKAN = 0, 1

  def __init__(self, build_dir=None, device=VULKAN):
    import sys
    from ._binding import load
    root = Path(__file__).resolve().parents[2]
    build_dir = Path(build_dir) if build_dir else (root / "build-rtvk")
    self.lib = load(build_dir)
    self.rt = _RT(self.lib, device)

  def _raw_tensor(self, arr, requires_grad=False):
    return self.rt._raw_tensor(arr, requires_grad)

  def tensor(self, arr, requires_grad=False):
    # arr must be a numpy float32 array
    return Tensor(self, self.rt.tensor(arr, requires_grad), arr.shape, requires_grad)

  # delegate runtime ops so Tensor(self.rt, ...) works with Engine as rt
  def __getattr__(self, name):
    return getattr(self.rt, name)

  def dispatch(self): return self.lib.rt_dispatch_count(self.rt.ctx)
  def reset_dispatch(self): self.lib.rt_reset_dispatch_count(self.rt.ctx)
