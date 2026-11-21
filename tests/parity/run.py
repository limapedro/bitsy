"""Run the Bitsy-vs-PyTorch parity suite (config: config.py, no argparse)."""

from __future__ import annotations

import json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy as bt  # noqa: E402

from tests.parity.cases import BINARY, CASES, ParityCase, UNARY, VALUE_SETS  # noqa: E402
from tests.parity.compare import compare  # noqa: E402
from tests.parity.config import CONFIG, ParityConfig  # noqa: E402
from tests.parity.tolerances import tolerance_for  # noqa: E402

# *** environment ***

def environment() -> dict[str, Any]:
  info:dict[str, Any] = {
    "platform": platform.platform(), "python": platform.python_version(),
    "torch": torch.__version__, "bitsy": bt.version(),
    "cuda_available": torch.cuda.is_available(), "bitsy_vulkan": bt.has_vulkan(),
  }
  if info["cuda_available"]:
    info.update(cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
  return info

def build_inputs(case:ParityCase, seed:int) -> tuple[torch.Tensor, torch.Tensor]:
  dtype = getattr(torch, case.dtype)
  gen = VALUE_SETS[case.value_set]
  return gen(case.lhs_shape, dtype, seed), gen(case.rhs_shape, dtype, seed + 1)

# *** cases ***

def success_case(case:ParityCase, config:ParityConfig) -> dict[str, Any]:
  lhs, rhs = build_inputs(case, config.seed)
  on_gpu = case.device == "vulkan"
  if on_gpu:
    lhs, rhs = lhs.cuda(), rhs.cuda()
  torch_op = (lambda x, y: torch.relu(torch.add(x, y))) if case.op == "add_relu" \
             else getattr(torch, case.op)
  bitsy_op = getattr(bt, case.op)

  if case.op in UNARY:
    expected = torch_op(lhs)
    got = torch.from_numpy(bitsy_op(bt.Tensor.from_numpy(lhs.cpu().numpy(), case.device)).numpy())
  else:
    expected = torch_op(lhs, rhs)
    got = torch.from_numpy(bitsy_op(
      bt.Tensor.from_numpy(lhs.cpu().numpy(), case.device),
      bt.Tensor.from_numpy(rhs.cpu().numpy(), case.device)).numpy())

  tolerance = tolerance_for(case.op, case.dtype)
  result:dict[str, Any] = {
    "name": case.name, "op": case.op, "device": case.device,
    "shape": list(expected.shape), "dtype": case.dtype, "value_set": case.value_set,
    "bitsy_vs_torch": compare(got, expected.cpu(), tolerance),
  }
  if not on_gpu and config.include_cuda and torch.cuda.is_available():
    cuda_expected = torch_op(lhs.cuda()) if case.op in UNARY else torch_op(lhs.cuda(), rhs.cuda())
    result["torch_cuda_vs_torch_cpu"] = compare(cuda_expected.cpu(), expected, tolerance)
    result["bitsy_vs_torch_cuda"] = compare(got, cuda_expected.cpu(), tolerance)
  result["passed"] = all(v["passed"] for v in result.values()
                         if isinstance(v, dict) and "passed" in v)
  return result

def error_case(case:ParityCase, config:ParityConfig) -> dict[str, Any]:
  lhs, rhs = build_inputs(case, config.seed)
  torch_raised = False
  try: torch.add(lhs, rhs)
  except RuntimeError: torch_raised = True
  bitsy_raised, message = False, ""
  try: bt.add(bt.Tensor.from_numpy(lhs.numpy()), bt.Tensor.from_numpy(rhs.numpy()))
  except bt.BitsyError as exc: bitsy_raised, message = True, str(exc)
  return {"name": case.name, "op": case.op, "dtype": case.dtype,
          "expect_error": True, "torch_raised": torch_raised,
          "bitsy_raised": bitsy_raised, "bitsy_message": message,
          "passed": torch_raised and bitsy_raised}

# *** runner ***

def run(config:ParityConfig = CONFIG) -> int:
  results, skipped = [], []
  for case in CASES:
    if case.op not in config.ops: continue
    if case.device == "vulkan" and not bt.has_vulkan():
      skipped.append(case.name)
      continue
    if case.device == "vulkan" and not torch.cuda.is_available():
      skipped.append(case.name)
      continue
    result = error_case(case, config) if case.expect_error else success_case(case, config)
    results.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {case.name}")
    if not result["passed"] and config.fail_fast: break

  passed = sum(r["passed"] for r in results)
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": environment(),
    "config": {"ops": list(config.ops), "backends": list(config.backends),
               "include_cuda": config.include_cuda, "seed": config.seed},
    "summary": {"total": len(results), "passed": passed,
                "failed": len(results) - passed, "skipped": skipped},
    "cases": results,
  }
  config.report_path.parent.mkdir(parents=True, exist_ok=True)
  config.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"\n{passed}/{len(results)} parity cases passed. "
        f"{len(skipped)} skipped. Report: {config.report_path}")
  return 0 if passed == len(results) else 1

if __name__ == "__main__": raise SystemExit(run())
