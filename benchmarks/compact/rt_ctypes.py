"""Minimal ctypes binding to the v0.2 compact runtime (bitsy_rt.dll).

Used by parity + benchmark scripts. Dataclass config, no argparse.
"""
from __future__ import annotations

import ctypes as C
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(build_dir: Path) -> C.CDLL:
  dll = build_dir / ("bitsy_rt.dll" if sys.platform == "win32" else "libbitsy_rt.so")
  lib = C.CDLL(str(dll))
  lib.rt_create.argtypes = [C.c_int]; lib.rt_create.restype = C.c_void_p
  lib.rt_destroy.argtypes = [C.c_void_p]
  lib.rt_from_host.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int,
                               C.POINTER(C.c_int64), C.c_int]
  lib.rt_from_host.restype = C.c_void_p
  lib.rt_backward.argtypes = [C.c_void_p, C.c_void_p]
  lib.rt_grad.argtypes = [C.c_void_p]; lib.rt_grad.restype = C.c_void_p
  lib.rt_to_host.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
  lib.rt_realize.argtypes = [C.c_void_p, C.c_void_p]
  lib.rt_dispatch_count.argtypes = [C.c_void_p]; lib.rt_dispatch_count.restype = C.c_int
  lib.rt_reset_dispatch_count.argtypes = [C.c_void_p]
  # binary ops
  for n in ("add", "sub", "mul", "div", "pow", "maximum", "minimum",
            "gt", "ge", "lt", "le", "eq", "ne"):
    fn = getattr(lib, f"rt_{n}")
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]; fn.restype = C.c_void_p
  # unary ops
  for n in ("neg", "exp", "log", "sqrt", "tanh", "sigm", "relu", "gelu", "abs", "sign"):
    fn = getattr(lib, f"rt_{n}")
    fn.argtypes = [C.c_void_p, C.c_void_p]; fn.restype = C.c_void_p
  lib.rt_where.argtypes = [C.c_void_p]*4; lib.rt_where.restype = C.c_void_p
  lib.rt_transpose.argtypes = [C.c_void_p, C.c_void_p]; lib.rt_transpose.restype = C.c_void_p
  lib.rt_reshape.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.POINTER(C.c_int64)]
  lib.rt_reshape.restype = C.c_void_p
  lib.rt_gemm.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int]
  lib.rt_gemm.restype = C.c_void_p
  lib.rt_reduce.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int]
  lib.rt_reduce.restype = C.c_void_p
  return lib


class RT:
  CPU, VULKAN = 0, 1

  def __init__(self, lib: C.CDLL, device: int = CPU):
    self.lib = lib
    self.ctx = lib.rt_create(device)

  def __del__(self):
    self.lib.rt_destroy(self.ctx)

  def tensor(self, arr: np.ndarray, requires_grad: bool = False) -> C.c_void_p:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    dims = (C.c_int64 * a.ndim)(*a.shape)
    flat = a.flatten()
    return self.lib.rt_from_host(self.ctx, flat.ctypes.data_as(C.POINTER(C.c_float)),
                                 a.ndim, dims, 1 if requires_grad else 0)

  def backward(self, loss): self.lib.rt_backward(self.ctx, loss)
  def grad(self, t, shape): return self.get(self.lib.rt_grad(t), shape)

  def get(self, t: C.c_void_p, shape) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    self.lib.rt_to_host(t, out.ctypes.data_as(C.POINTER(C.c_float)))
    return out

  def binary(self, name: str, a, b):
    return getattr(self.lib, f"rt_{name}")(self.ctx, a, b)

  def unary(self, name: str, a):
    return getattr(self.lib, f"rt_{name}")(self.ctx, a)

  def gemm(self, a, b, bias=None, ta=0, tb=0):
    return self.lib.rt_gemm(self.ctx, a, b, bias or C.c_void_p(), ta, tb)

  def reduce(self, a, kind=0, axis=0, keepdim=0):
    return self.lib.rt_reduce(self.ctx, a, kind, axis, keepdim)

  def where(self, cond, a, b):
    return self.lib.rt_where(self.ctx, cond, a, b)

  def transpose(self, a): return self.lib.rt_transpose(self.ctx, a)
  def reshape(self, a, shape):
    dims=(C.c_int64*len(shape))(*shape)
    return self.lib.rt_reshape(self.ctx, a, len(shape), dims)

  def reset_dispatch(self): self.lib.rt_reset_dispatch_count(self.ctx)
  def dispatch(self) -> int: return self.lib.rt_dispatch_count(self.ctx)
