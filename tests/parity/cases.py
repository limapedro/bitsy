"""Declarative parity cases: one entry per op/device configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

ValueGenerator = Callable[[tuple[int, ...], torch.dtype, int], torch.Tensor]
BINARY = ("add", "add_relu", "sub", "mul", "div")
UNARY = ("relu", "sigmoid", "tanh", "exp", "log", "sqrt", "gelu")
# _op_cases generates the standard normal-valued grid; sqrt/gelu have dedicated
# explicit cases (sqrt needs positive inputs, gelu uses torch.nn.functional).
_GRID_UNARY = ("relu", "sigmoid", "tanh", "exp", "log")

# *** value sets ***

def _tile(values:list[float], shape:tuple[int, ...], dtype:torch.dtype) -> torch.Tensor:
  n = 1
  for dim in shape: n *= dim
  if not n: return torch.empty(shape, dtype=dtype)
  base = torch.tensor(values, dtype=dtype)
  return base[torch.arange(n) % base.numel()].reshape(shape)

def normal(shape:tuple[int, ...], dtype:torch.dtype, seed:int) -> torch.Tensor:
  return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)

def large_small(shape:tuple[int, ...], dtype:torch.dtype, seed:int) -> torch.Tensor:
  return _tile([1e20, -1e20, 1e-20, -1e-20, 1e30, -1e30, 3.14159265, 1.0/3.0], shape, dtype)

def positive(shape:tuple[int, ...], dtype:torch.dtype, seed:int) -> torch.Tensor:
  return normal(shape, dtype, seed).abs() + 0.1

def special(shape:tuple[int, ...], dtype:torch.dtype, seed:int) -> torch.Tensor:
  return _tile([float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.0, -1.0, 2.5], shape, dtype)

VALUE_SETS = {"normal": normal, "large_small": large_small, "special": special,
              "positive": positive}

# *** cases ***

@dataclass(frozen=True, slots=True)
class ParityCase:
  name:str
  op:str
  lhs_shape:tuple[int, ...]
  rhs_shape:tuple[int, ...]
  value_set:str
  dtype:str = "float32"
  expect_error:bool = False
  device:str = "cpu"

def _op_cases(device:str) -> tuple[ParityCase, ...]:
  out = []
  for op in BINARY + _GRID_UNARY:
    shape = (7, 13)
    out.append(ParityCase(f"{op}_{device}", op, shape, shape, "normal", device=device))
  return tuple(out)

CASES:tuple[ParityCase, ...] = (
  ParityCase("add_scalar_scalar", "add", (), (), "normal"),
  ParityCase("add_vector", "add", (3,), (3,), "normal"),
  ParityCase("add_vector_special", "add", (8,), (8,), "special"),
  ParityCase("add_vector_extremes", "add", (8,), (8,), "large_small"),
  ParityCase("add_broadcast_row", "add", (2, 3), (3,), "normal"),
  ParityCase("add_broadcast_axes", "add", (1, 3, 1), (2, 1, 4), "normal"),
  ParityCase("add_scalar_rhs", "add", (2, 2), (), "normal"),
  ParityCase("add_size_one", "add", (1,), (5,), "normal"),
  ParityCase("add_empty", "add", (0, 2), (1, 2), "normal"),
  ParityCase("add_non_power_of_two", "add", (7, 13), (7, 13), "normal"),
  ParityCase("add_large", "add", (4096,), (4096,), "normal"),
  ParityCase("add_broadcast_special", "add", (2, 8), (8,), "special"),
  ParityCase("add_shape_error", "add", (2, 3), (4,), "normal", expect_error=True),
  ParityCase("add_broadcast_vulkan", "add", (2, 3), (3,), "normal", device="vulkan"),
  ParityCase("add_special_vulkan", "add", (8,), (8,), "special", device="vulkan"),
  ParityCase("add_large_vulkan", "add", (4096,), (4096,), "normal", device="vulkan"),
  ParityCase("add_empty_vulkan", "add", (0, 2), (1, 2), "normal", device="vulkan"),
  ParityCase("log_cpu", "log", (7, 13), (7, 13), "positive"),
  ParityCase("log_vulkan", "log", (7, 13), (7, 13), "positive", device="vulkan"),
  # exp_037: new primitives. sqrt uses positive values (torch sqrt of negatives
  # is NaN; Bitsy clamps at 0 — documented divergence, so the gate uses valid
  # inputs). GELU is exact-erf and checked against torch.nn.functional.gelu.
  ParityCase("sqrt_cpu", "sqrt", (7, 13), (7, 13), "positive"),
  ParityCase("sqrt_vulkan", "sqrt", (7, 13), (7, 13), "positive", device="vulkan"),
  ParityCase("gelu_cpu", "gelu", (7, 13), (7, 13), "normal"),
  ParityCase("gelu_vulkan", "gelu", (7, 13), (7, 13), "normal", device="vulkan"),
  ParityCase("matmul_cpu", "matmul", (4, 5), (5, 6), "normal"),
  ParityCase("matmul_vulkan", "matmul", (4, 5), (5, 6), "normal", device="vulkan"),
  ParityCase("matmul_large_vulkan", "matmul", (32, 64), (64, 32), "normal", device="vulkan"),
) + _op_cases("cpu") + _op_cases("vulkan")
