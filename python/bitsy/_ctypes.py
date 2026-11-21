"""Transitional ctypes binding for the Bitsy C core (M0/M2 bring-up, not M6 API).

No PyTorch at runtime. NumPy is imported lazily only inside tensor conversions.
"""

from __future__ import annotations

import ctypes, os
from pathlib import Path
from typing import Any

__all__ = ["BitsyError", "Tensor", "add", "add_relu", "sub", "mul", "div", "relu", "sigmoid", "flush", "wait", "capture_begin", "capture_end", "replay",
"profile", "profile_reset", "profile_dump", "device_time_ms",
           "tanh", "exp", "log", "matmul", "mm", "transpose", "view",
           "conv2d", "conv2d_backward_x", "conv2d_backward_w", "conv2d_backward_b",
           "device", "dtype_size", "has_vulkan", "last_error", "shutdown", "version"]

# *** errors and ctypes structs ***

class BitsyError(RuntimeError):
  def __init__(self, message:str, status:int) -> None:
    super().__init__(f"{message} (status {status})")
    self.message, self.status = message, status

class _Device(ctypes.Structure): _fields_ = [("type", ctypes.c_int), ("index", ctypes.c_int)]
class _Shape(ctypes.Structure): _fields_ = [("ndim", ctypes.c_int64), ("dims", ctypes.c_int64 * 8)]
class _ConvParams(ctypes.Structure): _fields_ = [("pad", ctypes.c_int), ("stride", ctypes.c_int)]

def _shape_struct(shape:tuple[int, ...]) -> _Shape:
  arr = (ctypes.c_int64 * 8)(*([0] * 8))
  for i, d in enumerate(shape): arr[i] = d
  return _Shape(len(shape), arr)

# *** load and sign the C ABI ***

def _find_library() -> ctypes.CDLL:
  if (override := os.environ.get("BT_LIBRARY_PATH")): return ctypes.CDLL(override)
  build = Path(__file__).resolve().parents[2] / "build"
  for name in ("bitsy.dll", "libbitsy.so", "libbitsy.dylib"):
    if (lib := build / name).is_file():
      if hasattr(os, "add_dll_directory"): os.add_dll_directory(str(build))
      return ctypes.CDLL(str(lib))
  raise BitsyError("Bitsy shared library not found; build it or set BT_LIBRARY_PATH", -1)

_lib = _find_library()

def _fn(name:str, argtypes:list, restype:Any) -> None:
  fn = getattr(_lib, name)
  fn.argtypes, fn.restype = argtypes, restype

_ptr, _i64p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)
_fn("bt_init", [], ctypes.c_int)
_fn("bt_shutdown", [], ctypes.c_int)
_fn("bt_version", [], ctypes.c_char_p)
_fn("bt_get_last_error", [], ctypes.c_char_p)
_fn("bt_status_name", [ctypes.c_int], ctypes.c_char_p)
_fn("bt_dtype_size", [ctypes.c_int], ctypes.c_size_t)
_fn("bt_has_vulkan", [], ctypes.c_int)
_fn("bt_profiler_enable", [ctypes.c_int], None)
_fn("bt_profiler_reset", [], None)
_fn("bt_profiler_dump_json", [ctypes.c_char_p], ctypes.c_int)
_fn("bt_vulkan_flush", [], ctypes.c_int)
_fn("bt_vulkan_wait", [], ctypes.c_int)
_fn("bt_vulkan_capture_begin", [], ctypes.c_int)
_fn("bt_vulkan_capture_end", [], ctypes.c_int)
_fn("bt_vulkan_replay", [ctypes.c_int], ctypes.c_int)
_fn("bt_vulkan_stage_alloc_count", [], ctypes.c_ulong)
_fn("bt_vulkan_device_time_ms", [], ctypes.c_double)
_fn("bt_device_cpu", [], _Device)
_fn("bt_device_from_string", [ctypes.c_char_p, ctypes.POINTER(_Device)], ctypes.c_int)
_fn("bt_tensor_create", [_i64p, ctypes.c_int64, ctypes.c_int, _Device, _ptr], _ptr)
_fn("bt_tensor_shape", [_ptr], ctypes.POINTER(_Shape))
_fn("bt_tensor_release", [_ptr], None)
_fn("bt_tensor_copy_to_host", [_ptr, _ptr, ctypes.c_size_t], ctypes.c_int)
for _name in ("add", "sub", "mul", "div", "add_relu"):
  _fn(f"bt_{_name}", [_ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
for _name in ("relu", "sigmoid", "tanh", "exp", "log"):
  _fn(f"bt_{_name}", [_ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_matmul", [_ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_transpose", [_ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_tensor_transpose_view", [_ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_sgd_step", [_ptr, _ptr, ctypes.c_float, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_reduce_sum", [_ptr, ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_tanh_backward", [_ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_broadcast_grad", [_ptr, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_dual_mul", [_ptr, _ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_matmul_bias", [_ptr, _ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_log_softmax", [_ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_ce_loss", [_ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_sgd_step_fused", [ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_float, ctypes.POINTER(ctypes.POINTER(_ptr))], ctypes.c_int)
_fn("bt_tensor_offset_view", [_ptr, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_tensor_view", [_ptr, _i64p, ctypes.c_int64, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_conv2d", [_ptr, _ptr, _ptr, _ConvParams, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_conv2d_backward_x", [_ptr, _ptr, _Shape, _ConvParams, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_conv2d_backward_w", [_ptr, _ptr, _Shape, _ConvParams, ctypes.POINTER(_ptr)], ctypes.c_int)
_fn("bt_conv2d_backward_b", [_ptr, ctypes.c_int64, ctypes.POINTER(_ptr)], ctypes.c_int)

def _check(status:int) -> None:
  if not status: return
  msg = (_lib.bt_get_last_error() or b"").decode()
  name = (_lib.bt_status_name(status) or b"unknown").decode()
  raise BitsyError(f"{name}: {msg}", status)

if _lib.bt_init(): raise BitsyError("bt_init failed", -1)

# *** thin API ***

def version() -> str: return (_lib.bt_version() or b"").decode()
def last_error() -> str: return (_lib.bt_get_last_error() or b"").decode()
def dtype_size(dtype:int) -> int: return int(_lib.bt_dtype_size(dtype))
def has_vulkan() -> bool: return bool(_lib.bt_has_vulkan())
def flush() -> None: _check(_lib.bt_vulkan_flush())
def wait() -> None: _check(_lib.bt_vulkan_wait())
def capture_begin() -> None: _check(_lib.bt_vulkan_capture_begin())
def capture_end() -> None: _check(_lib.bt_vulkan_capture_end())
def replay(count:int) -> None: _check(_lib.bt_vulkan_replay(count))
def stage_alloc_count() -> int: return int(_lib.bt_vulkan_stage_alloc_count())
def profile(enabled:bool=True) -> None: _lib.bt_profiler_enable(1 if enabled else 0)
def profile_reset() -> None: _lib.bt_profiler_reset()
def profile_dump(path:str) -> None: _check(_lib.bt_profiler_dump_json(path.encode()))
def device_time_ms() -> float: return float(_lib.bt_vulkan_device_time_ms())
def shutdown() -> None: _check(_lib.bt_shutdown())

def device(name:str) -> _Device:
  d = _Device()
  if not _lib.bt_device_from_string(name.encode(), ctypes.byref(d)):
    raise BitsyError(last_error(), -1)
  return d

BT_FLOAT32 = 0

class Tensor:
  """A minimal float32 tensor handle used by the parity harness."""
  __slots__ = ("_ptr", "_shape", "_device")

  def __init__(self, ptr:int, shape:tuple[int, ...], dev:_Device) -> None:
    if not ptr: raise BitsyError("bt_tensor_create returned NULL", -1)
    self._ptr, self._shape, self._device = ptr, shape, dev

  @classmethod
  def from_numpy(cls, array:Any, dev:str="cpu") -> Tensor:
    import numpy as np
    a = np.asarray(array, dtype=np.float32)
    if a.ndim: a = np.ascontiguousarray(a)  # ascontiguousarray promotes 0-d to 1-d
    dims = (ctypes.c_int64 * len(a.shape))(*a.shape)
    d = dev if isinstance(dev, _Device) else device(dev)
    ptr = _lib.bt_tensor_create(dims, len(a.shape), BT_FLOAT32, d,
                                ctypes.c_void_p(a.ctypes.data))
    return cls(ptr, tuple(map(int, a.shape)), d)

  @property
  def shape(self) -> tuple[int, ...]: return self._shape
  @property
  def device(self) -> str: return "vulkan" if self._device.type == 1 else "cpu"

  def numpy(self) -> Any:
    import numpy as np
    out = np.empty(self._shape, dtype=np.float32)
    _check(_lib.bt_tensor_copy_to_host(self._ptr, ctypes.c_void_p(out.ctypes.data), out.nbytes))
    return out

  def __del__(self) -> None:
    if _lib is None: return  # module globals are cleared at interpreter exit
    if (ptr := getattr(self, "_ptr", 0)): _lib.bt_tensor_release(ctypes.c_void_p(ptr))
    self._ptr = 0

  def __repr__(self) -> str: return f"Tensor(shape={self._shape}, device={self.device})"

def _op(name:str, a:Tensor, b:Tensor|None=None) -> Tensor:
  out = _ptr()
  fn = getattr(_lib, f"bt_{name}")
  if b is None: _check(fn(_ptr(a._ptr), ctypes.byref(out)))
  else: _check(fn(_ptr(a._ptr), _ptr(b._ptr), ctypes.byref(out)))
  shape = _lib.bt_tensor_shape(out)
  dims = tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim)))
  return Tensor(out.value, dims, a._device)

def add(a:Tensor, b:Tensor) -> Tensor: return _op("add", a, b)
def add_relu(a:Tensor, b:Tensor) -> Tensor: return _op("add_relu", a, b)
def sub(a:Tensor, b:Tensor) -> Tensor: return _op("sub", a, b)
def mul(a:Tensor, b:Tensor) -> Tensor: return _op("mul", a, b)
def div(a:Tensor, b:Tensor) -> Tensor: return _op("div", a, b)
def relu(a:Tensor) -> Tensor: return _op("relu", a)
def sigmoid(a:Tensor) -> Tensor: return _op("sigmoid", a)
def tanh(a:Tensor) -> Tensor: return _op("tanh", a)
def exp(a:Tensor) -> Tensor: return _op("exp", a)
def log(a:Tensor) -> Tensor: return _op("log", a)
def matmul(a:Tensor, b:Tensor) -> Tensor: return _op("matmul", a, b)
mm = matmul

def _from_handle(handle, dev) -> Tensor:
  shape = _lib.bt_tensor_shape(handle)
  dims = tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim)))
  return Tensor(handle.value, dims, dev)

def view(a:Tensor, shape:tuple[int, ...]) -> Tensor:
  out = _ptr()
  dims = (ctypes.c_int64 * len(shape))(*shape)
  _check(_lib.bt_tensor_view(_ptr(a._ptr), dims, len(shape), ctypes.byref(out)))
  return _from_handle(out, a._device)

def conv2d(x:Tensor, w:Tensor, b:Tensor, pad:int=1, stride:int=1) -> Tensor:
  out = _ptr()
  _check(_lib.bt_conv2d(_ptr(x._ptr), _ptr(w._ptr), _ptr(b._ptr),
                        _ConvParams(pad, stride), ctypes.byref(out)))
  return _from_handle(out, x._device)

def conv2d_backward_x(dy:Tensor, w:Tensor, x_shape:tuple[int,...], pad:int=1, stride:int=1) -> Tensor:
  out = _ptr()
  _check(_lib.bt_conv2d_backward_x(_ptr(dy._ptr), _ptr(w._ptr), _shape_struct(x_shape),
                                   _ConvParams(pad, stride), ctypes.byref(out)))
  return _from_handle(out, dy._device)

def conv2d_backward_w(x:Tensor, dy:Tensor, w_shape:tuple[int,...], pad:int=1, stride:int=1) -> Tensor:
  out = _ptr()
  _check(_lib.bt_conv2d_backward_w(_ptr(x._ptr), _ptr(dy._ptr), _shape_struct(w_shape),
                                   _ConvParams(pad, stride), ctypes.byref(out)))
  return _from_handle(out, x._device)

def conv2d_backward_b(dy:Tensor, out_channels:int) -> Tensor:
  out = _ptr()
  _check(_lib.bt_conv2d_backward_b(_ptr(dy._ptr), out_channels, ctypes.byref(out)))
  return _from_handle(out, dy._device)

def sgd_step_fused(ws, grads, lr:float):
  """Fused up-to-four-parameter SGD update (exp_029)."""
  warr = (_ptr * 4)(*[ _ptr(w._ptr) if w is not None else _ptr() for w in ws ])
  garr = (_ptr * 4)(*[ _ptr(g._ptr) if g is not None else _ptr() for g in grads ])
  slots = [_ptr() for _ in range(4)]
  out_ptrs = (ctypes.POINTER(_ptr) * 4)(*[ctypes.pointer(s) for s in slots])
  _check(_lib.bt_sgd_step_fused(warr, garr, ctypes.c_float(lr), out_ptrs))
  return [Tensor(slots[i].value, ws[i].shape, ws[i]._device) if ws[i] is not None else None
          for i in range(4)]

def ce_loss(ls:Tensor, onehot:Tensor) -> Tensor:
  """Fused cross-entropy scalar loss (exp_028)."""
  out = _ptr()
  _check(_lib.bt_ce_loss(_ptr(ls._ptr), _ptr(onehot._ptr), ctypes.byref(out)))
  return Tensor(out.value, (1, 1), ls._device)

def log_softmax(x:Tensor) -> Tensor:
  """Fused 2D log_softmax in one dispatch (exp_027)."""
  out = _ptr()
  _check(_lib.bt_log_softmax(_ptr(x._ptr), ctypes.byref(out)))
  shape = _lib.bt_tensor_shape(out)
  dims = tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim)))
  return Tensor(out.value, dims, x._device)

def matmul_bias(a:Tensor, b:Tensor, bias:Tensor) -> Tensor:
  """Fused a @ b + bias (bias shape (1,N)), one dispatch when aligned."""
  out = _ptr()
  _check(_lib.bt_matmul_bias(_ptr(a._ptr), _ptr(b._ptr), _ptr(bias._ptr), ctypes.byref(out)))
  shape = _lib.bt_tensor_shape(out)
  dims = tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim)))
  return Tensor(out.value, dims, a._device)

def dual_mul(g:Tensor, a:Tensor, b:Tensor, shape:tuple[int, ...]) -> tuple[Tensor, Tensor]:
  """Fused g*a and g*b into one packed buffer; returns two tensors sharing it."""
  out = _ptr()
  _check(_lib.bt_dual_mul(_ptr(g._ptr), _ptr(a._ptr), _ptr(b._ptr), ctypes.byref(out)))
  dims = (ctypes.c_int64 * len(shape))(*shape)
  views = []
  for off in (0, g.shape[0] * g.shape[1] if len(g.shape) == 2 else 1):
    v = _ptr()
    _check(_lib.bt_tensor_offset_view(out, ctypes.c_int64(off), dims, len(shape), ctypes.byref(v)))
    views.append(Tensor(v.value, tuple(shape), g._device))
  return views[0], views[1]

def broadcast_grad(g:Tensor, shape:tuple[int, ...]) -> Tensor:
  """Contiguous full-shape gradient from a reduction gradient (exp_020)."""
  dims = (ctypes.c_int64 * len(shape))(*shape)
  out = _ptr()
  _check(_lib.bt_broadcast_grad(_ptr(g._ptr), dims, len(shape), ctypes.byref(out)))
  return Tensor(out.value, tuple(shape), g._device)

def tanh_backward(g:Tensor, t:Tensor) -> Tensor:
  """Fused g * (1 - t*t) in one dispatch (exp_019)."""
  out = _ptr()
  _check(_lib.bt_tanh_backward(_ptr(g._ptr), _ptr(t._ptr), ctypes.byref(out)))
  return Tensor(out.value, g.shape, g._device)

def reduce_sum(a:Tensor, axis:int, keepdim:bool) -> Tensor:
  """Dedicated 2D axis reduction (exp_018), replacing ones-vector GEMMs."""
  out = _ptr()
  _check(_lib.bt_reduce_sum(_ptr(a._ptr), ctypes.c_uint32(axis),
                            ctypes.c_int(1 if keepdim else 0), ctypes.byref(out)))
  shape = _lib.bt_tensor_shape(out)
  dims = tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim)))
  return Tensor(out.value, dims, a._device)

def sgd_step(w:Tensor, grad:Tensor, lr:float) -> Tensor:
  """Fused w - lr*grad in one dispatch (exp_017)."""
  out = _ptr()
  _check(_lib.bt_sgd_step(_ptr(w._ptr), _ptr(grad._ptr), ctypes.c_float(lr), ctypes.byref(out)))
  return Tensor(out.value, w.shape, w._device)

def transpose_view(a:Tensor) -> Tensor:
  """Zero-dispatch 2D transpose view (shares storage)."""
  out = _ptr()
  _check(_lib.bt_tensor_transpose_view(_ptr(a._ptr), ctypes.byref(out)))
  shape = _lib.bt_tensor_shape(out)
  dims = tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim)))
  return Tensor(out.value, dims, a._device)

def transpose(a:Tensor) -> Tensor:
  out = _ptr()
  _check(_lib.bt_transpose(_ptr(a._ptr), ctypes.byref(out)))
  shape = _lib.bt_tensor_shape(out)
  dims = tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim)))
  return Tensor(out.value, dims, a._device)
