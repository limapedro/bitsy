"""Dataclass configuration for the parity harness. No argparse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ParityConfig:
    ops: tuple[str, ...] = ("add",)
    backends: tuple[str, ...] = ("cpu_ref",)
    include_cuda: bool = True
    seed: int = 1234
    report_path: Path = REPO_ROOT / "tests" / "parity" / "reports" / "latest.json"
    fail_fast: bool = False


CONFIG = ParityConfig()
