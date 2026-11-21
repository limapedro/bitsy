"""Single source of truth for numeric comparison tolerances.

Exact comparison is used for integer/bool ops, data movement, comparisons,
where, min/max selection, and deterministic correctly-rounded fp32 add/sub/
mul/div/sqrt. Transcendentals will use ULP distributions against a float64
reference, and reductions/GEMM/convolution/normalization will use rtol/atol
against both PyTorch and a float64 reference. Loosening any tolerance requires
an ARCHITECTURE_LOG.md entry with numeric justification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    mode: str
    rtol: float = 0.0
    atol: float = 0.0
    ulp_budget: int | None = None
    reason: str = ""


TOLERANCES: dict[tuple[str, str], Tolerance] = {
    ("add", "float32"): Tolerance(
        mode="exact",
        reason=(
            "IEEE-754 addition is correctly rounded and deterministic on both "
            "PyTorch CPU/CUDA and Hagane cpu_ref; results must be bit-identical "
            "apart from NaN payloads."
        ),
    ),
}


def tolerance_for(op: str, dtype: str) -> Tolerance:
    try:
        return TOLERANCES[(op, dtype)]
    except KeyError as exc:
        raise KeyError(
            f"no tolerance registered for op={op!r} dtype={dtype!r}; "
            "add one in tests/parity/tolerances.py with a justification"
        ) from exc
