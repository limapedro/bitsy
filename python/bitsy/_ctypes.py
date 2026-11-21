"""Transitional ctypes binding for the Bitsy C core (M0/M2 bring-up, not M6 API).

No PyTorch at runtime. NumPy is imported lazily only inside tensor conversions.
"""

from __future__ import annotations

import ctypes, os
from pathlib import Path
from typing import Any

__all__ = ["BitsyError", "Tensor", "add", "sub", "mul", "div", "relu", "sigmoid",
           "tanh", "exp", "device", "dtype_size", "has_vulkan", "last_error",
           "shutdown", "version"]

# *** errors and ctypes structs ***

class BitsyError(RuntimeError):
  def __init__(self, message:str, status:int) -> None:
    super().__init__(f"{message} (status {status})")
    self.message, self.status = message, status

class _Device(ctypes.Structure): _fields_ = [("type", ctypes.c_int), ("index", ctypes.c_int)]
class _Shape(ctypes.Structure): _fields_ = [("ndim", ctypes.c_int64), ("dims", ctypes.c_int64 * 8)]

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
_fn("bt_device_cpu", [], _Device)
_fn("bt_device_from_string", [ctypes.c_char_p, ctypes.POINTER(_Device)], ctypes.c_int)
_fn("bt_tensor_create", [_i64p, ctypes.c_int64, ctypes.c_int, _Device, _ptr], _ptr)
_fn("bt_tensor_shape", [_ptr], ctypes.POINTER(_Shape))
_fn("bt_tensor_release", [_ptr], None)
_fn("bt_tensor_copy_to_host", [_ptr, _ptr, ctypes.c_size_t], ctypes.c_int)
for _name in ("add", "sub", "mul", "div"):
  _fn(f"bt_{_name}", [_ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)
for _name in ("relu", "sigmoid", "tanh", "exp"):
  _fn(f"bt_{_name}", [_ptr, ctypes.POINTER(_ptr)], ctypes.c_int)

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
def sub(a:Tensor, b:Tensor) -> Tensor: return _op("sub", a, b)
def mul(a:Tensor, b:Tensor) -> Tensor: return _op("mul", a, b)
def div(a:Tensor, b:Tensor) -> Tensor: return _op("div", a, b)
def relu(a:Tensor) -> Tensor: return _op("relu", a)
def sigmoid(a:Tensor) -> Tensor: return _op("sigmoid", a)
def tanh(a:Tensor) -> Tensor: return _op("tanh", a)
def exp(a:Tensor) -> Tensor: return _op("exp", a)
