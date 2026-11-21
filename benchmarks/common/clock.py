"""Sample the live GPU clock so benchmark ratios can be interpreted."""

from __future__ import annotations

import subprocess


def gpu_clock_mhz() -> float:
  try:
    out = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=clocks.current.graphics", "--format=csv,noheader"],
      text=True, timeout=5)
    return float(out.strip().splitlines()[0].split()[0])
  except Exception:
    return 0.0


def sample_around(fn) -> tuple[object, float]:
  """Run `fn`, returning its value and the average clock during the call."""
  before = gpu_clock_mhz()
  value = fn()
  after = gpu_clock_mhz()
  return value, (before + after) * 0.5
