"""Transitional ctypes binding for the Hagane C core.

This module is M0 bring-up tooling, not the final Python API. It loads the
shared library produced by CMake and exposes only the operators implemented so
far. M6 replaces it with a hand-written CPython C extension and the full
PyTorch-compatible ``hagane.Tensor`` surface.

The Python package has no PyTorch dependency. NumPy is imported lazily only by
the two conversion helpers used by the test harness.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

__all__ = [
    "HaganeError",
    "Tensor",
    "add",
    "dtype_size",
    "last_error",
    "shutdown",
    "version",
]


class HaganeError(RuntimeError):
    """An error reported by the Hagane C core."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(f"{message} (status {status})")
        self.message = message
        self.status = status


class _Device(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("index", ctypes.c_int)]


class _Shape(ctypes.Structure):
    _fields_ = [
        ("ndim", ctypes.c_int64),
        ("dims", ctypes.c_int64 * 8),
    ]


def _find_library() -> ctypes.CDLL:
    override = os.environ.get("HG_LIBRARY_PATH")
    if override:
        return ctypes.CDLL(override)

    root = Path(__file__).resolve().parents[2]
    build = root / "build"
    candidates = (
        build / "hagane.dll",
        build / "libhagane.so",
        build / "libhagane.dylib",
    )
    for candidate in candidates:
        if candidate.is_file():
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(build))
            return ctypes.CDLL(str(candidate))

    raise HaganeError(
        "could not find the Hagane shared library; build it with CMake or set "
        "HG_LIBRARY_PATH",
        -1,
    )


_lib = _find_library()

_lib.hg_init.argtypes = []
_lib.hg_init.restype = ctypes.c_int
_lib.hg_shutdown.argtypes = []
_lib.hg_shutdown.restype = ctypes.c_int
_lib.hg_version.argtypes = []
_lib.hg_version.restype = ctypes.c_char_p
_lib.hg_get_last_error.argtypes = []
_lib.hg_get_last_error.restype = ctypes.c_char_p
_lib.hg_status_name.argtypes = [ctypes.c_int]
_lib.hg_status_name.restype = ctypes.c_char_p
_lib.hg_dtype_size.argtypes = [ctypes.c_int]
_lib.hg_dtype_size.restype = ctypes.c_size_t
_lib.hg_device_cpu.argtypes = []
_lib.hg_device_cpu.restype = _Device
_lib.hg_tensor_create.argtypes = [
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int64,
    ctypes.c_int,
    _Device,
    ctypes.c_void_p,
]
_lib.hg_tensor_create.restype = ctypes.c_void_p
_lib.hg_tensor_shape.argtypes = [ctypes.c_void_p]
_lib.hg_tensor_shape.restype = ctypes.POINTER(_Shape)
_lib.hg_tensor_release.argtypes = [ctypes.c_void_p]
_lib.hg_tensor_release.restype = None
_lib.hg_tensor_copy_to_host.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_lib.hg_tensor_copy_to_host.restype = ctypes.c_int
_lib.hg_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p)]
_lib.hg_add.restype = ctypes.c_int

HG_FLOAT32 = 0
HG_DEVICE_TYPE_CPU = 0


def _check(status: int) -> None:
    if status != 0:
        message = (_lib.hg_get_last_error() or b"").decode("utf-8", "replace")
        name = (_lib.hg_status_name(status) or b"unknown").decode()
        raise HaganeError(f"{name}: {message}", status)


if _lib.hg_init() != 0:
    message = (_lib.hg_get_last_error() or b"").decode("utf-8", "replace")
    raise HaganeError(f"hg_init failed: {message}", -1)


def version() -> str:
    """Return the compiled library version string."""
    return (_lib.hg_version() or b"").decode()


def last_error() -> str:
    """Return the C core's thread-local error message."""
    return (_lib.hg_get_last_error() or b"").decode()


def dtype_size(dtype: int) -> int:
    """Return the byte size of a dtype enum value."""
    return int(_lib.hg_dtype_size(dtype))


def shutdown() -> None:
    """Release process-wide C state. Normally called at interpreter exit."""
    _check(_lib.hg_shutdown())


class Tensor:
    """A minimal float32 tensor handle used by the M0 parity harness."""

    __slots__ = ("_ptr", "_shape")

    def __init__(self, ptr: int, shape: tuple[int, ...]) -> None:
        if not ptr:
            raise HaganeError("hg_tensor_create returned NULL", -1)
        self._ptr = ptr
        self._shape = shape

    @classmethod
    def from_numpy(cls, array: Any) -> "Tensor":
        """Create a CPU float32 tensor from a contiguous NumPy array."""
        import numpy as np

        arr = np.asarray(array, dtype=np.float32)
        # numpy.ascontiguousarray promotes 0-d inputs to 1-d; keep scalars 0-d.
        if arr.ndim > 0:
            arr = np.ascontiguousarray(arr)
        dims = (ctypes.c_int64 * len(arr.shape))(*arr.shape)
        ptr = _lib.hg_tensor_create(
            dims,
            len(arr.shape),
            HG_FLOAT32,
            _lib.hg_device_cpu(),
            ctypes.c_void_p(arr.ctypes.data),
        )
        return cls(ptr, tuple(int(d) for d in arr.shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def numpy(self) -> Any:
        """Copy this tensor into a NumPy array."""
        import numpy as np

        array = np.empty(self._shape, dtype=np.float32)
        _check(_lib.hg_tensor_copy_to_host(
            self._ptr,
            ctypes.c_void_p(array.ctypes.data),
            array.nbytes,
        ))
        return array

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", 0)
        if ptr:
            _lib.hg_tensor_release(ctypes.c_void_p(ptr))
            self._ptr = 0

    def __repr__(self) -> str:
        return f"Tensor(shape={self._shape})"


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise add with broadcasting; returns a new tensor."""
    out = ctypes.c_void_p()
    _check(_lib.hg_add(ctypes.c_void_p(a._ptr), ctypes.c_void_p(b._ptr),
                       ctypes.byref(out)))
    shape_ptr = _lib.hg_tensor_shape(out)
    shape = tuple(int(shape_ptr.contents.dims[i])
                  for i in range(int(shape_ptr.contents.ndim)))
    return Tensor(out.value, shape)
