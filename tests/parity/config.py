"""Dataclass configuration for the parity runner. No argparse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True, slots=True)
class ParityConfig:
  ops:tuple[str, ...] = ("add", "add_relu", "sub", "mul", "div", "relu", "sigmoid", "tanh", "exp",
                         "log", "matmul")
  backends:tuple[str, ...] = ("cpu_ref",)
  include_cuda:bool = True
  seed:int = 1234
  report_path:Path = REPO_ROOT / "tests" / "parity" / "reports" / "latest.json"
  fail_fast:bool = False

CONFIG = ParityConfig()
