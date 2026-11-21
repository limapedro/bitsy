"""Declarative parity cases: one entry per operator configuration.

Adding an operator means adding one entry here plus its tolerance entry. The
runner automatically exercises forward values, broadcasting, empty tensors,
size-1 dimensions, non-power-of-two sizes, and PyTorch's error behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


ValueGenerator = Callable[[tuple[int, ...], torch.dtype, int], torch.Tensor]


def _tile(values: list[float], shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    if count == 0:
        return torch.empty(shape, dtype=dtype)
    base = torch.tensor(values, dtype=dtype)
    indices = torch.arange(count) % base.numel()
    return base[indices].reshape(shape)


def normal(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def large_small(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    return _tile(
        [1e20, -1e20, 1e-20, -1e-20, 1e30, -1e30, 3.14159265, 1.0 / 3.0],
        shape,
        dtype,
    )


def special(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    return _tile(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.0, -1.0, 2.5],
        shape,
        dtype,
    )


VALUE_SETS: dict[str, ValueGenerator] = {
    "normal": normal,
    "large_small": large_small,
    "special": special,
}


@dataclass(frozen=True)
class ParityCase:
    name: str
    op: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    value_set: str
    dtype: str = "float32"
    expect_error: bool = False


CASES: tuple[ParityCase, ...] = (
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
    ParityCase("add_shape_error", "add", (2, 3), (4,), "normal",
               expect_error=True),
)
