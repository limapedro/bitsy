"""Comparators. Only 'exact' exists in M0; ULP/rtol modes arrive with M1-M5."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .tolerances import Tolerance

def _as_numpy(t:torch.Tensor) -> np.ndarray:
  return t.detach().cpu().contiguous().numpy()

def compare(actual:torch.Tensor, expected:torch.Tensor, tolerance:Tolerance) -> dict[str, Any]:
  if actual.shape != expected.shape:
    return {"passed": False, "reason": f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"}
  if tolerance.mode != "exact":
    raise NotImplementedError(f"tolerance mode {tolerance.mode!r} is not implemented yet")

  a, e = _as_numpy(actual), _as_numpy(expected)
  if a.dtype == np.float32 and e.dtype == np.float32:
    # Compare bit patterns so signed zeros differ, but allow any NaN payload.
    ab = actual.contiguous().view(torch.int32).numpy()
    eb = expected.contiguous().view(torch.int32).numpy()
    equal = (ab == eb) | np.isnan(e)
  else:
    equal = np.array_equal(a, e)

  mismatches = int(np.count_nonzero(~equal))
  ret:dict[str, Any] = {"passed": not mismatches, "mode": "exact",
                        "elements": int(a.size), "mismatches": mismatches}
  if mismatches:
    i = int(np.argmax(~equal))
    ret["first_mismatch"] = {"index": i, "actual": float(a.reshape(-1)[i]),
                             "expected": float(e.reshape(-1)[i])}
  return ret
