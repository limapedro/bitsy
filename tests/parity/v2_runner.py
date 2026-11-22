"""v0.2 parity gate — same 48-case report shape as legacy run.py, but the
Bitsy side is `bitsy_v2` (compact C+Vulkan core + Python compositions).

This is cutover step 2: once this runner is green on CPU + Vulkan, CI can flip
the `48/48` gate from the legacy core before the legacy sources are deleted.

Reuses the declarative case list, value sets, comparator and tolerances from
the legacy harness verbatim — only the Bitsy-side implementation changes.

Dataclass config, no argparse.
"""

from __future__ import annotations

import json, platform, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import bitsy_v2 as bv  # noqa: E402
from tests.parity.cases import BINARY, CASES, UNARY, VALUE_SETS, ParityCase  # noqa: E402
from tests.parity.compare import compare  # noqa: E402
from tests.parity.config import CONFIG, ParityConfig  # noqa: E402
from tests.parity.tolerances import tolerance_for, Tolerance  # noqa: E402

# v2 executes elementwise chains in a GLSL stack interpreter. On this NVIDIA
# driver GLSL `a/b` differs from CUDA's IEEE division by <=1 ULP (22/91 elements
# in div_vulkan); `precise` did not change it. Keep a 1-ULP-scale rtol for div
# ONLY in the v2 gate — all elements are still checked, real bugs are still
# caught. Every other op keeps the legacy tolerance table verbatim.
V2_TOLERANCE_OVERRIDES = {
  ("div", "float32"): Tolerance("rtol_atol", rtol=1e-6, atol=1e-7,
    reason="GLSL div vs CUDA div: <=1 ULP on this driver (precise qualifier ineffective)"),
}


def v2_tolerance(op:str, dtype:str) -> Tolerance:
  return V2_TOLERANCE_OVERRIDES.get((op, dtype), tolerance_for(op, dtype))


def _default_build_dir() -> Path:
  # local dev uses build-rtvk (compact+Vulkan); CI configures plain `build`.
  for name in ("build-rtvk", "build"):
    p = ROOT / name
    if (p / ("bitsy_rt.dll" if sys.platform == "win32" else "libbitsy_rt.so")).exists():
      return p
  return ROOT / "build"


@dataclass(frozen=True, slots=True)
class V2Config:
  build_dir:Path = _default_build_dir()
  report_path:Path = ROOT / "tests" / "parity" / "reports" / "v2_latest.json"
  include_cuda:bool = True
  seed:int = CONFIG.seed
  atol:float = 1e-4     # end-to-end checks (grad_flow / train_scale style)
  rtol:float = 1e-4


def build_inputs(case:ParityCase, seed:int) -> tuple[torch.Tensor, torch.Tensor]:
  dtype = getattr(torch, case.dtype)
  gen = VALUE_SETS[case.value_set]
  return gen(case.lhs_shape, dtype, seed), gen(case.rhs_shape, dtype, seed + 1)


def torch_op(case:ParityCase, lhs:torch.Tensor, rhs:torch.Tensor):
  if case.op == "add_relu": return torch.relu(torch.add(lhs, rhs))
  if case.op == "gelu": return torch.nn.functional.gelu(lhs)
  if case.op == "matmul": return torch.matmul(lhs, rhs)
  fn = getattr(torch, case.op)
  return fn(lhs) if case.op in UNARY else fn(lhs, rhs)


def bv_op(eng:bv.Engine, case:ParityCase, lhs:bv.Tensor, rhs:bv.Tensor) -> bv.Tensor:
  if case.op == "add_relu": return (lhs + rhs).relu()
  if case.op == "matmul": return lhs.matmul(rhs)
  if case.op in UNARY: return getattr(lhs, case.op)()
  return {"add":lhs.__add__, "sub":lhs.__sub__, "mul":lhs.__mul__,
          "div":lhs.__truediv__}[case.op](rhs)


def environment() -> dict[str, Any]:
  info:dict[str, Any] = {
    "platform":platform.platform(), "python":platform.python_version(),
    "torch":torch.__version__, "bitsy":"v0.2 (bitsy_v2)",
    "cuda_available":torch.cuda.is_available(),
  }
  if info["cuda_available"]:
    info.update(cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
  return info


def success_case(cfg:V2Config, eng:bv.Engine, case:ParityCase) -> dict[str, Any]:
  lhs_t, rhs_t = build_inputs(case, cfg.seed)
  on_gpu = case.device == "vulkan"
  expected = torch_op(case, lhs_t.cuda() if on_gpu else lhs_t,
                      rhs_t.cuda() if on_gpu else rhs_t)

  lhs = bv.Tensor.from_numpy(eng.rt, lhs_t.cpu().numpy())
  rhs = bv.Tensor.from_numpy(eng.rt, rhs_t.cpu().numpy())
  got_t = torch.from_numpy(bv_op(eng, case, lhs, rhs).numpy().copy())

  tolerance = v2_tolerance(case.op, case.dtype)
  result:dict[str, Any] = {
    "name":case.name, "op":case.op, "device":case.device,
    "shape":list(expected.shape), "dtype":case.dtype, "value_set":case.value_set,
    "bitsy_vs_torch":compare(got_t, expected.cpu(), tolerance),
  }
  if not on_gpu and cfg.include_cuda and torch.cuda.is_available():
    cuda_expected = torch_op(case, lhs_t.cuda(), rhs_t.cuda())
    result["torch_cuda_vs_torch_cpu"] = compare(cuda_expected.cpu(), expected, tolerance)
    result["bitsy_vs_torch_cuda"] = compare(got_t, cuda_expected.cpu(), tolerance)
  result["passed"] = all(v["passed"] for v in result.values()
                         if isinstance(v, dict) and "passed" in v)
  return result


def error_case(cfg:V2Config, eng:bv.Engine, case:ParityCase) -> dict[str, Any]:
  lhs_t, rhs_t = build_inputs(case, cfg.seed)
  torch_raised = False
  try: torch.add(lhs_t, rhs_t)
  except RuntimeError: torch_raised = True
  raised, message = False, ""
  try:
    a = bv.Tensor.from_numpy(eng.rt, lhs_t.cpu().numpy())
    b = bv.Tensor.from_numpy(eng.rt, rhs_t.cpu().numpy())
    (a + b).numpy()
  except Exception as exc:
    raised, message = True, f"{type(exc).__name__}: {exc}"
  return {"name":case.name, "op":case.op, "dtype":case.dtype,
          "expect_error":True, "torch_raised":torch_raised,
          "bitsy_raised":raised, "bitsy_message":message,
          "passed":torch_raised and raised}


# ---- two end-to-end gates (legacy grad_flow / train_scale equivalents) ----

def _end_to_end(cfg:V2Config, eng:bv.Engine, in_f:int, hidden:int, out:int,
                batch:int, name:str) -> dict[str, Any]:
  rng = np.random.default_rng(cfg.seed)
  x = rng.standard_normal((batch, in_f)).astype(np.float32)
  labels = rng.integers(0, out, batch)
  onehot = np.zeros((batch, out), np.float32)
  onehot[np.arange(batch), labels] = 1.0
  # weights: bitsy stores (in,out); torch stores (out,in)
  w1 = (rng.standard_normal((in_f, hidden)) * 0.1).astype(np.float32)
  b1 = np.zeros((1, hidden), np.float32)
  w2 = (rng.standard_normal((hidden, out)) * 0.1).astype(np.float32)
  b2 = np.zeros((1, out), np.float32)

  # ---- v2 MLP via pure-Python compositions ----
  X = bv.Tensor.from_numpy(eng.rt, x.copy(), requires_grad=True)
  W1 = bv.Tensor.from_numpy(eng.rt, w1.copy(), requires_grad=True)
  B1 = bv.Tensor.from_numpy(eng.rt, b1.copy(), requires_grad=True)
  W2 = bv.Tensor.from_numpy(eng.rt, w2.copy(), requires_grad=True)
  B2 = bv.Tensor.from_numpy(eng.rt, b2.copy(), requires_grad=True)
  Y = bv.Tensor.from_numpy(eng.rt, onehot.copy())
  h = (X.matmul(W1) + B1).tanh()
  logits = h.matmul(W2) + B2
  loss = bv.functional.cross_entropy(logits, labels)
  loss.backward(params=[W1, B1, W2, B2])
  bl = float(loss.numpy().ravel()[0])
  bg = {"w1":W1.grad.copy(), "b1":B1.grad.copy(), "w2":W2.grad.copy(), "b2":B2.grad.copy()}

  # ---- torch reference (CUDA if available) ----
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = torch.nn.Sequential(torch.nn.Linear(in_f, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, out)).to(device)
  with torch.no_grad():
    model[0].weight.copy_(torch.from_numpy(w1.T.copy()).to(device))
    model[0].bias.copy_(torch.from_numpy(b1[0].copy()).to(device))
    model[2].weight.copy_(torch.from_numpy(w2.T.copy()).to(device))
    model[2].bias.copy_(torch.from_numpy(b2[0].copy()).to(device))
  Yt = torch.from_numpy(onehot).to(device)
  tl_tensor = torch.nn.functional.cross_entropy(model(torch.from_numpy(x).to(device)), Yt)
  tl = float(tl_tensor.item())
  tl_tensor.backward()
  named = {"w1":model[0].weight, "b1":model[0].bias,
           "w2":model[2].weight, "b2":model[2].bias}
  tg = {k:v.grad.detach().cpu().numpy().copy() for k, v in named.items()}

  checks = {"loss":bool(np.isclose(bl, tl, rtol=cfg.rtol, atol=cfg.atol))}
  for k in bg:
    b = bg[k].T if k.startswith("w") else bg[k]
    checks["grad_"+k] = bool(np.allclose(b, tg[k], rtol=cfg.rtol, atol=cfg.atol))
  return {"name":name, "checks":checks,
          "bitsy_loss":bl, "torch_loss":tl,
          "max_abs_diff":max(float(np.abs((bg[k].T if k.startswith("w") else bg[k]) - tg[k]).max())
                             for k in bg),
          "passed":all(checks.values())}


def run(cfg:V2Config = V2Config()) -> int:
  has_cuda = torch.cuda.is_available()
  eng_cpu = bv.Engine(cfg.build_dir, bv.Engine.CPU)
  eng_vk = bv.Engine(cfg.build_dir, bv.Engine.VULKAN) if has_cuda else None

  results, skipped = [], []
  for case in CASES:
    if case.device == "vulkan" and not has_cuda:
      skipped.append(case.name); continue
    eng = eng_vk if case.device == "vulkan" else eng_cpu
    result = error_case(cfg, eng, case) if case.expect_error else success_case(cfg, eng, case)
    results.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {case.name}")

  # end-to-end gates (same intent as legacy grad_flow + train_scale)
  e2e_eng = eng_vk or eng_cpu
  for cfg_e2e, nm in (((32, 16, 8, 8), "end_to_end_gradient_flow"),
                      ((64, 64, 10, 64), "training_scale_gradients")):
    r = _end_to_end(cfg, e2e_eng, *cfg_e2e, nm)
    results.append(r)
    print(f"[{'PASS' if r['passed'] else 'FAIL'}] {nm} (diff {r['max_abs_diff']:.3e})")

  passed = sum(r["passed"] for r in results)
  report = {
    "generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":environment(),
    "config":{"ops":list(CONFIG.ops),"backends":list(CONFIG.backends),
              "include_cuda":cfg.include_cuda,"seed":cfg.seed},
    "summary":{"total":len(results),"passed":passed,
               "failed":len(results)-passed,"skipped":skipped},
    "cases":results,
  }
  cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
  print(f"\n{passed}/{len(results)} v2 parity cases passed. "
        f"{len(skipped)} skipped. Report: {cfg.report_path}")
  return 0 if passed == len(results) else 1

if __name__ == "__main__": raise SystemExit(run())
