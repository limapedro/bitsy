"""Transitional ctypes binding for the Bitsy C core (M0 bring-up, not the M6 API).

The Python package never imports PyTorch at runtime. NumPy is imported lazily
only inside the two conversion helpers used by the parity harness.
"""

from __future__ import annotations

import ctypes, os
from pathlib import Path
from typing import Any

__all__ = ["BitsyError", "Tensor", "add", "dtype_size", "last_error", "shutdown", "version"]

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
      # Windows needs an explicit DLL search directory for the CRT dependency.
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
_fn("bt_device_cpu", [], _Device)
_fn("bt_tensor_create", [_i64p, ctypes.c_int64, ctypes.c_int, _Device, _ptr], _ptr)
_fn("bt_tensor_shape", [_ptr], ctypes.POINTER(_Shape))
_fn("bt_tensor_release", [_ptr], None)
_fn("bt_tensor_copy_to_host", [_ptr, _ptr, ctypes.c_size_t], ctypes.c_int)
_fn("bt_add", [_ptr, _ptr, ctypes.POINTER(_ptr)], ctypes.c_int)

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
def shutdown() -> None: _check(_lib.bt_shutdown())

BT_FLOAT32 = 0

class Tensor:
  """A minimal float32 tensor handle used by the M0 parity harness."""
  __slots__ = ("_ptr", "_shape")

  def __init__(self, ptr:int, shape:tuple[int, ...]) -> None:
    if not ptr: raise BitsyError("bt_tensor_create returned NULL", -1)
    self._ptr, self._shape = ptr, shape

  @classmethod
  def from_numpy(cls, array:Any) -> Tensor:
    import numpy as np
    a = np.asarray(array, dtype=np.float32)
    if a.ndim: a = np.ascontiguousarray(a)  # ascontiguousarray promotes 0-d to 1-d
    dims = (ctypes.c_int64 * len(a.shape))(*a.shape)
    ptr = _lib.bt_tensor_create(dims, len(a.shape), BT_FLOAT32,
                                _lib.bt_device_cpu(), ctypes.c_void_p(a.ctypes.data))
    return cls(ptr, tuple(map(int, a.shape)))

  @property
  def shape(self) -> tuple[int, ...]: return self._shape

  def numpy(self) -> Any:
    import numpy as np
    out = np.empty(self._shape, dtype=np.float32)
    _check(_lib.bt_tensor_copy_to_host(self._ptr, ctypes.c_void_p(out.ctypes.data), out.nbytes))
    return out

  def __del__(self) -> None:
    if (ptr := getattr(self, "_ptr", 0)): _lib.bt_tensor_release(ctypes.c_void_p(ptr))
    self._ptr = 0

  def __repr__(self) -> str: return f"Tensor(shape={self._shape})"

def add(a:Tensor, b:Tensor) -> Tensor:
  """Elementwise add with broadcasting; returns a new tensor."""
  out = _ptr()
  _check(_lib.bt_add(_ptr(a._ptr), _ptr(b._ptr), ctypes.byref(out)))
  shape = _lib.bt_tensor_shape(out)
  return Tensor(out.value, tuple(int(shape.contents.dims[i]) for i in range(int(shape.contents.ndim))))
