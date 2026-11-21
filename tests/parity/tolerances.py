"""One tolerance table. Exact means bit-identical float32 patterns (NaN payloads
excluded); loosening a tolerance needs an ARCHITECTURE_LOG entry with numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Tolerance:
  mode:str
  rtol:float = 0.0
  atol:float = 0.0
  ulp_budget:int|None = None
  reason:str = ""

EXACT = Tolerance("exact", reason="correctly-rounded IEEE ops are deterministic")
# Different exp implementations are expected to differ by a few ULPs. A ULP
# distribution against float64 is still to be recorded; this is the M2 bound.
APPROX = Tolerance("rtol_atol", rtol=1e-5, atol=1e-6,
                   reason="transcendentals differ by a few ULPs between libraries")

TOLERANCES:dict[tuple[str, str], Tolerance] = {
  **{(op, "float32"): EXACT for op in ("add", "add_relu", "sub", "mul", "div", "relu")},
  **{(op, "float32"): APPROX for op in ("sigmoid", "tanh", "exp", "log")},
  ("matmul", "float32"): Tolerance("rtol_atol", rtol=1e-5, atol=1e-5,
                                   reason="fp32 accumulation order differs from cuBLAS"),
}

def tolerance_for(op:str, dtype:str) -> Tolerance:
  if (tol := TOLERANCES.get((op, dtype))) is None:
    raise KeyError(f"no tolerance for {op!r}/{dtype!r}; add one in tolerances.py with a reason")
  return tol
