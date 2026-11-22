"""Enforce the C+Vulkan core line budget (≤ 4096, stretch 2048).

On this documentation-only branch there is no `src/` or `include/`; the tool
prints the target allocation from LINE_BUDGET.md and reports 0 core lines
("planned"). Once implementation lands, it counts real sources and fails CI
above the budget. Dataclass config, no argparse.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_DIRS = (ROOT / "src", ROOT / "include")
EXTS = {".c", ".h"}

@dataclass(frozen=True, slots=True)
class Budget:
  hard:int = 4096
  stretch:int = 2048

def count_core() -> tuple[int, list[tuple[int, Path]]]:
  rows:list[tuple[int, Path]] = []
  for d in CORE_DIRS:
    if not d.exists():
      continue
    for p in sorted(d.rglob("*")):
      if p.is_file() and p.suffix in EXTS:
        n = sum(1 for _ in p.read_text(encoding="utf-8", errors="replace").splitlines())
        rows.append((n, p.relative_to(ROOT)))
  return sum(n for n, _ in rows), rows

def run(cfg:Budget = Budget()) -> int:
  total, rows = count_core()
  if rows:
    print(f"{'lines':>7}  path")
    for n, p in sorted(rows, key=lambda r: -r[0]):
      print(f"{n:>7}  {p}")
    print(f"{'-'*7}  {'-'*40}")
    print(f"{total:>7}  C core total")
  else:
    print("No src/ or include/ on this branch — core is PLANNED (0 lines).")
    print("Target allocation (LINE_BUDGET.md): 2850 core lines, hard cap 4096, stretch 2048.")
    return 0
  if total > cfg.hard:
    print(f"OVER HARD BUDGET by {total - cfg.hard} lines (cap {cfg.hard}).")
    return 1
  if total > cfg.stretch:
    print(f"Within hard budget ({cfg.hard - total} spare); above stretch goal by {total - cfg.stretch}.")
  else:
    print(f"Within stretch budget ({cfg.stretch - total} spare).")
  return 0

if __name__ == "__main__":
  sys.exit(run())
