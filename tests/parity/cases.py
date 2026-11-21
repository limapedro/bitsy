"""Declarative parity cases: one entry per op configuration, not per test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

ValueGenerator = Callable[[tuple[int, ...], torch.dtype, int], torch.Tensor]

# *** value sets ***

def _tile(values:list[float], shape:tuple[int, ...], dtype:torch.dtype) -> torch.Tensor:
  n = 1
  for dim in shape: n *= dim
  if not n: return torch.empty(shape, dtype=dtype)  # zero-size tensors do no work
  base = torch.tensor(values, dtype=dtype)
  return base[torch.arange(n) % base.numel()].reshape(shape)

def normal(shape:tuple[int, ...], dtype:torch.dtype, seed:int) -> torch.Tensor:
  return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)

def large_small(shape:tuple[int, ...], dtype:torch.dtype, seed:int) -> torch.Tensor:
  return _tile([1e20, -1e20, 1e-20, -1e-20, 1e30, -1e30, 3.14159265, 1.0/3.0], shape, dtype)

def special(shape:tuple[int, ...], dtype:torch.dtype, seed:int) -> torch.Tensor:
  return _tile([float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.0, -1.0, 2.5], shape, dtype)

VALUE_SETS = {"normal": normal, "large_small": large_small, "special": special}

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
)
