"""Count lines in the C+Vulkan core and enforce the 4096-line budget.

After the PR #11 legacy cutover there is a single runtime: `src/rt` +
`include/bitsy_rt`. The old 5,710-line reference core was deleted; this tool
no longer reports a separate legacy figure. Dataclass config, no argparse.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 4096
EXTS = {".c", ".h", ".comp"}


@dataclass(frozen=True, slots=True)
class CountConfig:
  budget:int = BUDGET
  fail_over_budget:bool = True


def count_lines(path:Path) -> int:
  return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines())


def collect(dirs:tuple[Path, ...]) -> list[tuple[int, Path]]:
  rows = []
  for d in dirs:
    if not d.exists():
      continue
    for p in sorted(d.rglob("*")):
      if p.is_file() and p.suffix in EXTS:
        rows.append((count_lines(p), p.relative_to(ROOT)))
  return sorted(rows, key=lambda r: -r[0])


def show(title:str, rows:list[tuple[int, Path]]) -> int:
  total = sum(n for n, _ in rows)
  print(f"\n=== {title} ===")
  print(f"{'lines':>7}  path")
  for n, p in rows:
    print(f"{n:>7}  {p}")
  print(f"{'-'*7}  {'-'*40}")
  print(f"{total:>7}  TOTAL")
  return total


def run(cfg:CountConfig = CountConfig()) -> int:
  core = collect((ROOT / "src" / "rt", ROOT / "include" / "bitsy_rt"))
  total = show("v0.2 C+Vulkan core (budget-enforced)", core)
  print(f"\ncompact core={total} (budget {cfg.budget})")
  if total > cfg.budget:
    print(f"OVER BUDGET by {total - cfg.budget} lines.")
    return 1 if cfg.fail_over_budget else 0
  print(f"WITHIN BUDGET ({cfg.budget - total} lines spare).")
  return 0

if __name__ == "__main__":
  sys.exit(run())
