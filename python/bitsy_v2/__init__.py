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

from .compiled import Compiled, CompiledMany
__all__ = ["Engine", "Tensor", "nn", "optim", "functional", "Compiled", "CompiledMany"]


class _ZeroCopyArray(np.ndarray):
  """numpy view of C-owned HOST_CACHED readback memory; keeps RT alive."""
  pass


class _RT:
  """Adapter exposing the methods Tensor expects, backed by ctypes."""
  def __init__(self, lib, device):
    self.lib, self.ctx = lib, lib.rt_create(device)
    self.leaves = []
    # Reusable host readback buffers. Allocating a fresh 4 MB numpy array on
    # every numpy() call faults ~1000 pages (~0.55 ms); reusing a small pool
    # cuts readback to the raw memcpy (~0.22 ms). Like PyTorch's `.numpy()`,
    # the returned array shares storage and is overwritten by a later read of
    # the same shape — callers needing a long-lived snapshot should .copy().
    self._read_pool = {}
    self._read_idx = {}
    self._rb_slot = 0
  def __del__(self): self.lib.rt_destroy(self.ctx)
  def _raw_tensor(self, arr, requires_grad):
    h = self.lib.rt_from_host(self.ctx,
      arr.ctypes.data_as(C.POINTER(C.c_float)), arr.ndim,
      (C.c_int64*arr.ndim)(*arr.shape), 1 if requires_grad else 0)
    return h
  tensor = _raw_tensor
  def get(self, h, shape):
    # Zero-copy: C copies device -> persistent HOST_CACHED ring buffer, and we
    # wrap that memory directly as numpy (no second memcpy). The ring rotates
    # so concurrently-held arrays do not alias; the view keeps RT alive.
    n = 1
    for d in shape: n *= d
    if n == 0: return np.empty(shape, dtype=np.float32)
    ptr = self.lib.rt_read_ptr(self.ctx, h, self._rb_slot)
    self._rb_slot = (self._rb_slot + 1) % 32
    arr = np.ctypeslib.as_array(ptr, shape=(n,)).reshape(shape).view(_ZeroCopyArray)
    arr._rt = self
    return arr
  def binary(self, name, a, b): return getattr(self.lib, f"rt_{name}")(self.ctx, a, b)
  def unary(self, name, a): return getattr(self.lib, f"rt_{name}")(self.ctx, a)
  def gemm(self, a, b, bias=None, ta=0, tb=0):
    return self.lib.rt_gemm(self.ctx, a, b, bias or C.c_void_p(), ta, tb)
  def reduce(self, a, kind, axis, keepdim): return self.lib.rt_reduce(self.ctx, a, kind, axis, keepdim)
  def softmax(self, a, bias, kind): return self.lib.rt_softmax(self.ctx, a, bias, kind)
  def layernorm(self, x, w, b, res, eps): return self.lib.rt_layernorm(self.ctx, x, w, b, res, eps)
  def where(self, c, a, b): return self.lib.rt_where(self.ctx, c, a, b)
  def transpose(self, a): return self.lib.rt_transpose(self.ctx, a)
  def reshape(self, a, shape):
    return self.lib.rt_reshape(self.ctx, a, len(shape), (C.c_int64*len(shape))(*shape))
  def backward(self, loss): self.lib.rt_backward(self.ctx, loss)
  def grad(self, t, shape): return self.get(self.lib.rt_grad(t), shape)
  def grad_refresh(self, t, shape):
    """Re-read a leaf's gradient from device after a graph replay."""
    return self.get(self.lib.rt_grad(t), shape).reshape(shape)


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

  def upload(self, t, arr):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    self.lib.rt_upload(self.rt.ctx, t.h, a.ctypes.data_as(C.POINTER(C.c_float)))
  def capture_begin(self): self.lib.rt_capture_begin(self.rt.ctx)
  def capture_end(self): self.lib.rt_capture_end(self.rt.ctx)
  def replay(self): self.lib.rt_replay(self.rt.ctx)
  def replay_submit_only(self): self.lib.rt_replay_submit_only(self.rt.ctx)
  def replay_submit_async(self): self.lib.rt_replay_submit_async(self.rt.ctx)
  def replay_n(self, n): self.lib.rt_replay_n(self.rt.ctx, n)
  def sgd_update(self, w, grad, lr, wd=0.0): self.lib.rt_sgd_update(self.rt.ctx, w, grad, lr, wd)
  def zero_grad(self, t):
    h = t.h if hasattr(t,"h") else t
    self.lib.rt_zero_grad(h)
  def compile(self, fn, shapes):
    """Capture ``fn`` (a pure callable over tensors) into a replayable object.

    ``shapes`` gives the input shapes; ``fn`` is run once inside a capture so
    the returned Compiled replays the whole graph in one submit per call, with
    fresh inputs written through CPU-mapped buffers.
    """
    inputs = [self.tensor(np.zeros(tuple(s), np.float32)) for s in shapes]
    # Map inputs BEFORE capture: the graph records their device address, so a
    # later remap would leave the captured dispatch reading the old buffer.
    ptrs = [self.map_input(t) for t in inputs]
    self.capture_begin()
    out = fn(*inputs)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    for t in outs:
      if hasattr(t, "realize"): t.realize()
    self.capture_end()
    return Compiled(self, fn, inputs, outs, ptrs)

  def compile_many(self, fn, shapes, K, zero_grads=None):
    """Capture K graphs with distinct mapped inputs; one call runs all K in a
    single submit. Returns a callable taking K input batches (the amortized
    captured path as a normal Python API)."""
    inputs, ptrs, outs = [], [], []
    for gi in range(K):
      ins = [self.tensor(np.zeros(tuple(s), np.float32)) for s in shapes]
      ps = [self.map_input(t) for t in ins]
      self.stream_begin(gi)
      out = fn(*ins)
      o = out if isinstance(out, (tuple, list)) else (out,)
      for t in o:
        if hasattr(t, "realize"): t.realize()
      self.stream_end(gi)
      for leaf in (zero_grads or []):
        if hasattr(leaf, "h"):
          self.zero_grad(leaf)
      inputs.append(ins); ptrs.append(ps); outs.append(o)
    return CompiledMany(self, fn, inputs, outs, ptrs)

  def stream_begin(self, idx): self.lib.rt_stream_begin(self.rt.ctx, idx)
  def stream_end(self, idx): self.lib.rt_stream_end(self.rt.ctx, idx)
  def stream_submit(self, n): self.lib.rt_stream_submit(self.rt.ctx, n)
  def map_input(self, t):
    h = t.h if hasattr(t,"h") else t
    return self.lib.rt_tensor_map(self.rt.ctx, h)
  def flush_input(self, t):
    h = t.h if hasattr(t,"h") else t
    self.lib.rt_tensor_flush(self.rt.ctx, h)
  def grad_tensor(self, leaf):
    h = leaf.h if hasattr(leaf, "h") else leaf
    return self.lib.rt_grad(h)
  def device_time_ms(self): return self.lib.rt_device_time_ms(self.rt.ctx)
  def copy_time_ms(self): return self.lib.rt_copy_time_ms(self.rt.ctx)
  def dispatch(self): return self.lib.rt_dispatch_count(self.rt.ctx)
  def reset_dispatch(self): self.lib.rt_reset_dispatch_count(self.rt.ctx)
