"""ctypes binding for the compact C runtime (self-contained, no external paths)."""
from __future__ import annotations
import ctypes as C
import sys
from pathlib import Path


def load(build_dir: Path) -> C.CDLL:
  dll = build_dir / ("bitsy_rt.dll" if sys.platform == "win32" else "libbitsy_rt.so")
  lib = C.CDLL(str(dll))
  lib.rt_create.argtypes=[C.c_int]; lib.rt_create.restype=C.c_void_p
  lib.rt_destroy.argtypes=[C.c_void_p]
  lib.rt_from_host.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_int,C.POINTER(C.c_int64),C.c_int]
  lib.rt_from_host.restype=C.c_void_p
  lib.rt_backward.argtypes=[C.c_void_p,C.c_void_p]
  lib.rt_grad.argtypes=[C.c_void_p]; lib.rt_grad.restype=C.c_void_p
  lib.rt_to_host.argtypes=[C.c_void_p,C.POINTER(C.c_float)]
  lib.rt_realize.argtypes=[C.c_void_p,C.c_void_p]
  lib.rt_dispatch_count.argtypes=[C.c_void_p]; lib.rt_dispatch_count.restype=C.c_int
  lib.rt_reset_dispatch_count.argtypes=[C.c_void_p]
  for n in ("add","sub","mul","div","pow","maximum","minimum","gt","ge","lt","le","eq","ne"):
    fn=getattr(lib,f"rt_{n}"); fn.argtypes=[C.c_void_p]*3; fn.restype=C.c_void_p
  for n in ("neg","exp","log","sqrt","tanh","sigm","relu","gelu","abs","sign"):
    fn=getattr(lib,f"rt_{n}"); fn.argtypes=[C.c_void_p]*2; fn.restype=C.c_void_p
  lib.rt_where.argtypes=[C.c_void_p]*4; lib.rt_where.restype=C.c_void_p
  lib.rt_transpose.argtypes=[C.c_void_p]*2; lib.rt_transpose.restype=C.c_void_p
  lib.rt_reshape.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.POINTER(C.c_int64)]
  lib.rt_reshape.restype=C.c_void_p
  lib.rt_gemm.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int]
  lib.rt_gemm.restype=C.c_void_p
  lib.rt_reduce.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int]
  lib.rt_reduce.restype=C.c_void_p
  return lib
