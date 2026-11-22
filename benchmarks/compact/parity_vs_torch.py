"""Compact runtime (C+Vulkan) numerics vs real PyTorch CUDA, fp32.

Every case: build the same graph in bitsy_rt and torch, realize on both, compare
max_abs_diff against documented tolerance. Reports pass/miss honestly.
Dataclass config, no argparse.
"""
from __future__ import annotations

import json, sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "compact"))
from rt_ctypes import load, RT

@dataclass(frozen=True, slots=True)
class ParityConfig:
  build_dir:Path = ROOT / "build-rtvk"
  device:int = RT.VULKAN
  seed:int = 1234
  tol_elementwise:float = 1e-5
  tol_matmul:float = 1e-4
  result_path:Path = ROOT / "benchmarks" / "results" / "compact_rt_parity.json"

CONFIG = ParityConfig()

def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())

def run(cfg: ParityConfig = CONFIG) -> int:
  if not torch.cuda.is_available():
    print("CUDA not available; skipping (parity oracle requires PyTorch CUDA)")
    return 0
  lib = load(cfg.build_dir)
  rt = RT(lib, cfg.device)
  rng = np.random.default_rng(cfg.seed)
  results = []

  def record(name, got, want, tol):
    d = maxdiff(got, want)
    results.append({"name": name, "max_abs_diff": d, "tol": tol, "passed": d <= tol})
    print(f"[{'PASS' if d<=tol else 'FAIL'}] {name:24s} diff={d:.3e} tol={tol:.0e}")

  # --- elementwise (fused program on GPU) ---
  a = rng.standard_normal((4, 6)).astype(np.float32)
  b = rng.standard_normal((4, 6)).astype(np.float32)
  ta, tb = rt.tensor(a), rt.tensor(b)
  for name, op in (("add","add"),("sub","sub"),("mul","mul"),("div","div"),
                   ("pow","pow"),("maximum","maximum"),("minimum","minimum"),
                   ("gt","gt"),("ge","ge"),("lt","lt"),("le","le")):
    if name == "pow":
      aa = rng.uniform(0.5, 3, (4,6)).astype(np.float32); bb = rng.uniform(0.5,3,(4,6)).astype(np.float32)
      t1, t2 = rt.tensor(aa), rt.tensor(bb)
      got = rt.get(rt.binary(op, t1, t2), a.shape)
      want = (torch.from_numpy(aa).cuda() ** torch.from_numpy(bb).cuda()).cpu().numpy()
    else:
      got = rt.get(rt.binary(op, ta, tb), a.shape)
      x, y = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
      want = getattr(torch, {"maximum":"max","minimum":"min"}.get(op, op))(x, y).cpu().numpy() \
             if op in ("maximum","minimum") else {
               "gt": x>y, "ge": x>=y, "lt": x<y, "le": x<=y,
               "add": x+y, "sub": x-y, "mul": x*y, "div": x/y,
             }[op].float().cpu().numpy()
    record(f"elem_{name}", got, want, cfg.tol_elementwise)

  for name in ("exp","log","sqrt","tanh","sigm","relu","gelu","abs"):
    aa = rng.uniform(0.5, 3, (4,6)).astype(np.float32) if name in ("log","sqrt") else a
    t1 = rt.tensor(aa)
    got = rt.get(rt.unary(name, t1), aa.shape)
    x = torch.from_numpy(aa).cuda()
    want = {
      "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "tanh": torch.tanh,
      "sigm": torch.sigmoid, "relu": torch.relu, "abs": torch.abs,
    }.get(name, lambda z: torch.nn.functional.gelu(z))(x).cpu().numpy()
    record(f"unary_{name}", got, want, cfg.tol_elementwise)

  # where
  cond = (a > b)
  got = rt.get(rt.where(rt.binary("gt", ta, tb), ta, tb), a.shape)
  want = torch.from_numpy(np.where(cond, a, b).astype(np.float32))
  record("where", got, want.numpy(), cfg.tol_elementwise)

  # --- GEMM with bias + transpose flags ---
  A = rng.standard_normal((8, 16)).astype(np.float32)
  B = rng.standard_normal((16, 12)).astype(np.float32)
  bias = rng.standard_normal((12,)).astype(np.float32)
  got = rt.get(rt.gemm(rt.tensor(A), rt.tensor(B), rt.tensor(bias)), (8,12))
  want = (torch.from_numpy(A).cuda() @ torch.from_numpy(B).cuda() + torch.from_numpy(bias).cuda()).cpu().numpy()
  record("gemm_bias", got, want, cfg.tol_matmul)

  At = rng.standard_normal((16, 8)).astype(np.float32)  # A^T stored (16,8)
  got = rt.get(rt.gemm(rt.tensor(At), rt.tensor(B), None, 1, 0), (8,12))
  want = (torch.from_numpy(At).cuda().T @ torch.from_numpy(B).cuda()).cpu().numpy()
  record("gemm_transA", got, want, cfg.tol_matmul)

  # --- reduce variants ---
  v = rng.standard_normal((5, 7)).astype(np.float32)
  tv = rt.tensor(v)
  for kind, name, fn in ((0,"sum",torch.sum),(1,"mean",torch.mean),(2,"max",torch.max),(3,"min",torch.min)):
    got = rt.get(rt.reduce(tv, kind, 1, 0), (5,))
    want = fn(torch.from_numpy(v).cuda(), dim=1).values.cpu().numpy() if kind in (2,3) \
           else fn(torch.from_numpy(v).cuda(), dim=1).cpu().numpy()
    record(f"reduce_{name}", got, want, cfg.tol_elementwise)

  # fusion proof
  rt.reset_dispatch()
  z = rt.unary("relu", rt.binary("add", rt.binary("mul", ta, ta), ta))
  rt.get(z, a.shape)
  nd = rt.dispatch()
  results.append({"name":"fusion_dispatch_count","max_abs_diff":float(nd),"tol":1,"passed":nd==1})
  print(f"[{'PASS' if nd==1 else 'FAIL'}] fusion_dispatch_count = {nd} (want 1)")

  passed = sum(r["passed"] for r in results)
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "device": "vulkan" if cfg.device==RT.VULKAN else "cpu"},
    "config": {k: str(v) if isinstance(v, Path) else v for k,v in asdict(cfg).items()},
    "summary": {"total": len(results), "passed": passed, "failed": len(results)-passed},
    "cases": results,
  }
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
  print(f"\n{passed}/{len(results)} compact-runtime vs PyTorch CUDA cases passed")
  print(f"report: {cfg.result_path}")
  return 0 if passed == len(results) else 1

if __name__ == "__main__":
  raise SystemExit(run())
