"""Bitsy — a C11 deep-learning framework with a Vulkan backend.

M0/M2 bring-up package: imports cleanly and exposes the ctypes-backed helpers
used by the parity harness. M6 replaces them with the CPython extension and the
full PyTorch-compatible ``bt.Tensor`` surface.
"""

from . import autograd, nn, optim
from ._ctypes import (BitsyError, Tensor, add, sub, mul, div, relu, sigmoid,
                      tanh, exp, log, matmul, mm, transpose, view, conv2d,
                      conv2d_backward_x, conv2d_backward_w, conv2d_backward_b,
                      device, dtype_size, has_vulkan, flush, device_time_ms,
                      last_error, shutdown, version)

__version__ = "0.2.0-bitsy"
__all__ = ["BitsyError", "Tensor", "add", "sub", "mul", "div", "relu", "sigmoid",
           "tanh", "exp", "log", "matmul", "mm", "transpose", "view", "conv2d", "device", "dtype_size", "has_vulkan",
           "flush", "device_time_ms", "last_error", "shutdown", "version", "__version__"]
