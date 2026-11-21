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

# IEEE add is correctly rounded on both sides, so there is no tolerance to tune.
TOLERANCES:dict[tuple[str, str], Tolerance] = {
  ("add", "float32"): Tolerance("exact", reason="correctly-rounded IEEE addition is deterministic"),
}

def tolerance_for(op:str, dtype:str) -> Tolerance:
  if (tol := TOLERANCES.get((op, dtype))) is None:
    raise KeyError(f"no tolerance for {op!r}/{dtype!r}; add one in tolerances.py with a reason")
  return tol
