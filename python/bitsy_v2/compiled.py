"""High-level captured-callable API (PR #40).

Turns a pure-functional callable over bitsy tensors into a replayable object:
inputs are CPU-mapped HOST_CACHED buffers, so each call writes fresh data with a
memcpy and replays the pre-recorded graph in ONE submit. This makes the
amortized/captured execution path the DEFAULT for any repeated computation,
rather than something only hand-written benchmarks use.
"""
from __future__ import annotations
import ctypes as C
import numpy as np


class Compiled:
  """A captured function. ``__call__`` writes fresh inputs and replays one submit.

  Weights/captured constants live in the graph as leaf tensors and update in
  place (e.g. device SGD), so a compiled training step replays repeated updates.
  """
  def __init__(self, engine, fn, inputs, outputs, ptrs=None):
    self.eng = engine
    self.fn = fn
    self.inputs = list(inputs)
    self.ptrs = list(ptrs) if ptrs is not None else [engine.map_input(t) for t in self.inputs]
    self.outs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    self.out_shapes = [tuple(t.shape) for t in self.outs]

  def __call__(self, *arrays):
    if len(arrays) != len(self.ptrs):
      raise ValueError(f"expected {len(self.ptrs)} inputs, got {len(arrays)}")
    for ptr, t, a in zip(self.ptrs, self.inputs, arrays):
      aa = np.ascontiguousarray(a, dtype=np.float32)
      C.memmove(ptr, aa.ctypes.data_as(C.POINTER(C.c_float)), aa.nbytes)
      self.eng.flush_input(t)
    self.eng.replay()
    out = [self.eng.get(t.h, s).reshape(s) for t, s in zip(self.outs, self.out_shapes)]
    return out[0] if len(out) == 1 else tuple(out)


class AutoCapture:
  """Caches compiled callables by key so recurring shapes automatically use the
  captured/replay path after the first call. This makes amortized execution the
  default: callers write a normal function, and repeat calls skip re-capture.
  """
  def __init__(self, engine, K=6):
    self.eng = engine
    self.K = K
    self._cache = {}

  def run(self, key, fn, shapes, batches, zero_grads=None, mode="auto"):
    """Execute ``batches`` (K groups of inputs) through fn.

    mode="auto" picks captured replay for training steps (zero_grads provided —
    measured faster because PyTorch/our eager pay backward+optimizer host cost)
    and eager execution for inference/single ops (capture replay measured slower
    than eager there). mode="capture" / "eager" force a specific path.
    """
    if mode == "auto":
      mode = "capture" if zero_grads else "eager"
    if mode == "eager":
      out=[]
      for batch in batches:
        grows=[]
        for group in batch:
          ts=[self.eng.tensor(np.asarray(g,dtype=np.float32)) for g in group]
          grows.append(fn(*ts))
        out.append(grows)
      return out
    c = self._cache.get(key)
    if c is None:
      c = self.eng.compile_many(fn, shapes, self.K, zero_grads=zero_grads)
      self._cache[key] = c
    return c(batches)

  def infer(self, key, fn, shapes, inputs):
    """Inference convenience: ``inputs`` is a list of single input arrays.

    Uses the adaptive default, which selects eager execution for inference
    workloads where captured replay was measured to be slower.
    """
    return self.run(key, fn, shapes, [[[x] for x in inputs]])


class CompiledMany:
  """K captured graphs with distinct CPU-mapped inputs, submitted in ONE call.

  This is the amortized/captured path exposed as a normal Python callable: one
  ``__call__`` takes K fresh input batches, writes them with memcpys, submits K
  graph executions in a single vkQueueSubmit2, and returns K outputs.
  """
  def __init__(self, engine, fn, inputs, outputs, ptrs):
    self.eng = engine
    self.fn = fn
    self.inputs = inputs          # [K][num_inputs]
    self.ptrs = ptrs              # [K][num_inputs]
    self.outs = outputs           # [K][num_outputs]
    self.out_shapes = [[tuple(t.shape) for t in g] for g in outputs]
    self.K = len(inputs)

  def __call__(self, batches, read=False):
    if len(batches) != self.K:
      raise ValueError(f"expected {self.K} input batches, got {len(batches)}")
    for gi in range(self.K):
      for ptr, t, a in zip(self.ptrs[gi], self.inputs[gi], batches[gi]):
        aa = np.ascontiguousarray(a, dtype=np.float32)
        C.memmove(ptr, aa.ctypes.data_as(C.POINTER(C.c_float)), aa.nbytes)
        self.eng.flush_input(t)
    self.eng.stream_submit(self.K)
    if not read:
      # Return device tensors (PyTorch-style async): no forced host readback.
      return [o[0] if len(o) == 1 else tuple(o) for o in self.outs]
    result = []
    for gi in range(self.K):
      o = [self.eng.get(t.h, s).reshape(s)
           for t, s in zip(self.outs[gi], self.out_shapes[gi])]
      result.append(o[0] if len(o) == 1 else tuple(o))
    return result
