"""Count lines in the C+Vulkan core. Enforces the 4096-line budget on the
v0.2 compact runtime (src/rt + include/bitsy_rt), reporting the legacy core
separately as a reference that will be removed when the rewrite replaces it.

Prints per-file tables and totals for both; exits non-zero if the COMPACT core
exceeds the budget. Dataclass config, no argparse.
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
  compact = collect((ROOT / "src" / "rt", ROOT / "include" / "bitsy_rt"))
  legacy = collect((ROOT / "src", ROOT / "include"))
  # exclude the compact runtime from the legacy figure
  legacy = [(n, p) for n, p in legacy if not (str(p).startswith("src\\rt") or str(p).startswith("src/rt") or "bitsy_rt" in str(p))]
  c_total = show("v0.2 COMPACT core (budget-enforced)", compact)
  l_total = show("legacy core (reference, removed at cutover)", legacy)
  print(f"\ncompact={c_total} (budget {cfg.budget}), legacy reference={l_total}")
  if c_total > cfg.budget:
    print(f"OVER BUDGET by {c_total - cfg.budget} lines.")
    return 1 if cfg.fail_over_budget else 0
  print(f"WITHIN BUDGET ({cfg.budget - c_total} lines spare).")
  return 0


if __name__ == "__main__":
  sys.exit(run())
