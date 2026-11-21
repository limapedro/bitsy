"""Run the Hagane-vs-PyTorch parity suite.

Configuration lives in config.py as a dataclass; there is no argparse. Run
from the repository root with ``python tests/parity/run.py``.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Make the in-tree Python package and the test package importable without
# installing either one. This also lets the file run directly as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

import hagane as hg  # noqa: E402

from tests.parity.cases import CASES, ParityCase, VALUE_SETS  # noqa: E402
from tests.parity.compare import compare  # noqa: E402
from tests.parity.config import CONFIG, ParityConfig  # noqa: E402
from tests.parity.tolerances import tolerance_for  # noqa: E402


def _environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hagane": hg.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _build_inputs(case: ParityCase, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = getattr(torch, case.dtype)
    generator = VALUE_SETS[case.value_set]
    lhs = generator(case.lhs_shape, dtype, seed)
    rhs = generator(case.rhs_shape, dtype, seed + 1)
    return lhs, rhs


def _run_success_case(case: ParityCase, config: ParityConfig) -> dict[str, Any]:
    lhs, rhs = _build_inputs(case, config.seed)
    expected = torch.add(lhs, rhs)
    tolerance = tolerance_for(case.op, case.dtype)

    hg_lhs = hg.Tensor.from_numpy(lhs.numpy())
    hg_rhs = hg.Tensor.from_numpy(rhs.numpy())
    hg_result = torch.from_numpy(hg.add(hg_lhs, hg_rhs).numpy())

    result: dict[str, Any] = {
        "name": case.name,
        "op": case.op,
        "shape": list(expected.shape),
        "dtype": case.dtype,
        "value_set": case.value_set,
        "hagane_vs_torch_cpu": compare(hg_result, expected, tolerance),
    }

    if config.include_cuda and torch.cuda.is_available():
        lhs_cuda = lhs.cuda()
        rhs_cuda = rhs.cuda()
        expected_cuda = torch.add(lhs_cuda, rhs_cuda).cpu()
        result["torch_cuda_vs_torch_cpu"] = compare(expected_cuda, expected, tolerance)
        result["hagane_vs_torch_cuda"] = compare(hg_result, expected_cuda, tolerance)

    result["passed"] = all(
        value["passed"]
        for key, value in result.items()
        if isinstance(value, dict) and "passed" in value
    )
    return result


def _run_error_case(case: ParityCase, config: ParityConfig) -> dict[str, Any]:
    lhs, rhs = _build_inputs(case, config.seed)

    torch_raised = False
    try:
        torch.add(lhs, rhs)
    except RuntimeError:
        torch_raised = True

    hg_raised = False
    hg_message = ""
    try:
        hg.add(hg.Tensor.from_numpy(lhs.numpy()),
               hg.Tensor.from_numpy(rhs.numpy()))
    except hg.HaganeError as exc:
        hg_raised = True
        hg_message = str(exc)

    return {
        "name": case.name,
        "op": case.op,
        "dtype": case.dtype,
        "expect_error": True,
        "torch_raised": torch_raised,
        "hagane_raised": hg_raised,
        "hagane_message": hg_message,
        "passed": torch_raised and hg_raised,
    }


def run(config: ParityConfig = CONFIG) -> int:
    results: list[dict[str, Any]] = []
    for case in CASES:
        if case.op not in config.ops:
            continue
        if case.expect_error:
            result = _run_error_case(case, config)
        else:
            result = _run_success_case(case, config)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {case.name}")
        if not result["passed"] and config.fail_fast:
            break

    passed = sum(1 for r in results if r["passed"])
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "config": {
            "ops": list(config.ops),
            "backends": list(config.backends),
            "include_cuda": config.include_cuda,
            "seed": config.seed,
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
        "cases": results,
    }

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2) + "\n",
                                  encoding="utf-8")

    print(f"\n{passed}/{len(results)} parity cases passed. "
          f"Report: {config.report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(run())
