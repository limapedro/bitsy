"""Hagane (鋼) — a C11 deep-learning framework with a Vulkan backend.

This is the M0 bring-up package. It imports cleanly and exposes the narrow
ctypes-backed helpers needed by the parity harness. The PyTorch-compatible
``hg.Tensor`` API and ``hg.nn`` / ``hg.optim`` subpackages arrive in M6/M7.

``from hagane import add, Tensor`` loads the shared library built by CMake.
"""

from ._ctypes import (
    HaganeError,
    Tensor,
    add,
    dtype_size,
    last_error,
    shutdown,
    version,
)

__version__ = "0.1.0-m0"
__all__ = [
    "HaganeError",
    "Tensor",
    "add",
    "dtype_size",
    "last_error",
    "shutdown",
    "version",
    "__version__",
]
