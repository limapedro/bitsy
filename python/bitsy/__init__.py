"""Bitsy — a C11 deep-learning framework with a Vulkan backend.

M0 bring-up package: it imports cleanly and exposes the narrow ctypes-backed
helpers needed by the parity harness. M6 replaces them with the CPython
extension and the full PyTorch-compatible ``bt.Tensor`` surface.
"""

from ._ctypes import BitsyError, Tensor, add, dtype_size, last_error, shutdown, version

__version__ = "0.2.0-bitsy"
__all__ = ["BitsyError", "Tensor", "add", "dtype_size", "last_error", "shutdown", "version", "__version__"]
