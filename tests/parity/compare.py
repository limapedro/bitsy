"""Comparison functions used by the parity harness."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .tolerances import Tolerance


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def compare(actual: torch.Tensor, expected: torch.Tensor,
            tolerance: Tolerance) -> dict[str, Any]:
    """Compare two float32 tensors using the configured tolerance mode."""
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "reason": f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}",
        }

    if tolerance.mode != "exact":
        raise NotImplementedError(
            f"tolerance mode {tolerance.mode!r} is not implemented yet"
        )

    actual_np = _as_numpy(actual)
    expected_np = _as_numpy(expected)

    if actual_np.dtype == np.float32 and expected_np.dtype == np.float32:
        # Compare bit patterns so that signed zeros are distinguished, while
        # allowing any NaN payload (PyTorch does not promise a specific one).
        actual_bits = actual.contiguous().view(torch.int32).numpy()
        expected_bits = expected.contiguous().view(torch.int32).numpy()
        nan_mask = np.isnan(expected_np)
        equal = (actual_bits == expected_bits) | nan_mask
    else:
        equal = np.array_equal(actual_np, expected_np)

    mismatches = int(np.count_nonzero(~equal))
    result: dict[str, Any] = {
        "passed": mismatches == 0,
        "mode": "exact",
        "elements": int(actual_np.size),
        "mismatches": mismatches,
    }
    if mismatches:
        first = int(np.argmax(~equal))
        result["first_mismatch"] = {
            "index": first,
            "actual": float(actual_np.reshape(-1)[first]),
            "expected": float(expected_np.reshape(-1)[first]),
        }
    return result
