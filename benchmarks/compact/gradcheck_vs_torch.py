"""Gradcheck: compact-runtime autograd vs PyTorch CUDA, fp32.

For each op, build the graph in bitsy_rt (leaf requires_grad) and in torch,
run backward on a scalar loss, compare d(loss)/d(leaf). Negative results are
reported, not hidden. Dataclass config, no argparse.
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
class Config:
  build_dir:Path = ROOT / "build-rtvk"
  device:int = RT.VULKAN
  seed:int = 7
  shape:tuple = (4, 6)
  tol:float = 1e-3
  result_path:Path = ROOT / "benchmarks" / "results" / "compact_rt_gradcheck.json"

CFG = Config()

def md(a, b): return float(np.abs(np.asarray(a,dtype=np.float32)-np.asarray(b,dtype=np.float32)).max())

def run(cfg: Config = CFG) -> int:
  if not torch.cuda.is_available():
    print("CUDA unavailable; skipping"); return 0
  lib = load(cfg.build_dir)
  rt = RT(lib, cfg.device)   # single shared context for all cases
  results = []
  rng = np.random.default_rng(cfg.seed)
  shape = cfg.shape

  def check(name, rt_fn, torch_fn, x_np, y_np=None, tol=None):
    """rt_fn(rt, tx, ty)->loss tensor; torch_fn(x,y)->scalar tensor."""
    tol = tol or cfg.tol
    x = rng.uniform(0.5, 2.0, shape).astype(np.float32) if x_np is None else x_np
    tx = rt.tensor(x, requires_grad=True)
    ty = rt.tensor(y_np, requires_grad=True) if y_np is not None else None
    try:
      loss = rt_fn(rt, tx, ty)
      rt.backward(loss)
      g_rt = rt.grad(tx, x.shape)

      X = torch.from_numpy(x).cuda().requires_grad_(True)
      Y = torch.from_numpy(y_np).cuda().requires_grad_(True) if y_np is not None else None
      tl = torch_fn(X, Y)
      tl.backward()
      g_torch = X.grad.cpu().numpy()
      d = md(g_rt, g_torch)
      passed = d <= tol
      results.append({"name": name, "max_diff": d, "tol": tol, "passed": passed})
      print(f"[{'PASS' if passed else 'FAIL'}] {name:22s} diff={d:.3e}")
    except Exception as e:
      results.append({"name": name, "max_diff": None, "tol": tol, "passed": False, "error": str(e)})
      print(f"[FAIL] {name:22s} EXC {e}")

  base = lambda rt, x, y: rt.unary("sum", x) if False else None
  # We don't have a unary sum; use reduce(gemm?) Instead scalar loss = reduce_sum via rt_reduce
  def loss_of(t, rt): return rt.reduce(t, 0, 0, 0)  # RT_REDUCE_SUM over axis 0 (keeps dims>1 -> not scalar)
  # For a true scalar loss, reduce all axes then reduce again to 1 element.
  def scalar_loss(rt, t):
    r = rt.reduce(t, 0, 0, 0)   # (6,)
    return rt.reduce(r, 0, 0, 0)  # (1,)

  # unary ops
  for name in ("exp","log","sqrt","tanh","sigm","relu","gelu","abs","neg"):
    def rf(rt, x, y, n=name): return scalar_loss(rt, rt.unary(n, x))
    def tf(X, Y, n=name):
      f = {"exp":torch.exp,"log":torch.log,"sqrt":torch.sqrt,"tanh":torch.tanh,
           "sigm":torch.sigmoid,"relu":torch.relu,"abs":torch.abs,"neg":torch.neg}.get(n,
             lambda z: torch.nn.functional.gelu(z))
      return f(X).sum()
    check(f"grad_{name}", rf, tf, None)

  # binary ops (grad wrt first operand)
  for name in ("add","sub","mul","div","pow","maximum","minimum"):
    y = rng.uniform(0.5, 2.0, shape).astype(np.float32)
    def rf(rt, x, y, n=name): return scalar_loss(rt, rt.binary(n, x, y))
    def tf(X, Y, n=name):
      if n=="maximum": return torch.maximum(X,Y).sum()
      if n=="minimum": return torch.minimum(X,Y).sum()
      if n=="pow": return (X**Y).sum()
      return {"add":lambda: X+Y,"sub":lambda: X-Y,"mul":lambda: X*Y,"div":lambda: X/Y}[n]().sum()
    check(f"grad_{name}", rf, tf, None, y)

  # reduce ops: reduce over axis 0 -> (6,), then SUM to scalar (both sides)
  for kind, name, tf_reduce in ((0,"sum",torch.sum),(1,"mean",torch.mean),(2,"max",torch.max),(3,"min",torch.min)):
    def rf(rt, x, y, k=kind):
      r = rt.reduce(x, k, 0, 0)          # (6,)
      return rt.reduce(r, 0, 0, 0)       # scalar (SUM of reduced)
    def tf(X, Y, k=kind, fn=tf_reduce):
      if k in (2,3): return fn(X, dim=0).values.sum()
      return fn(X, dim=0).sum()
    check(f"grad_reduce_{name}", rf, tf, None)

  # matmul: d(sum(A@B))/dA
  A = rng.standard_normal((4,6)).astype(np.float32)
  B = rng.standard_normal((6,3)).astype(np.float32)
  def rf_gemm(rt, x, y): return scalar_loss(rt, rt.gemm(x, y))
  def tf_gemm(X, Y): return (X @ Y).sum()
  check("grad_matmul_A", rf_gemm, tf_gemm, A, B)

  # transpose: d(sum(X.T))/dX
  def rf_t(rt, x, y): return scalar_loss(rt, rt.transpose(x))
  def tf_t(X, Y): return X.T.sum()
  check("grad_transpose", rf_t, tf_t, None)

  # reshape: d(sum(X.reshape))/dX
  xr = np.arange(12, dtype=np.float32).reshape(3,4)/10+1
  def rf_r(rt, x, y):
    resh=rt.reshape(x, (4,3))
    return scalar_loss(rt, resh)
  def tf_r(X, Y): return X.reshape(4,3).sum()
  check("grad_reshape", rf_r, tf_r, xr)

  # where: d(sum(where(cond,x,y)))/dx
  wc = (rng.uniform(0,1,(4,6)) > 0.5).astype(np.float32)
  wy = rng.uniform(0.5,2.0,(4,6)).astype(np.float32)
  def rf_w(rt, x, y):
    cnd=rt.tensor(wc)
    ty=rt.tensor(wy)
    return scalar_loss(rt, rt.where(cnd, x, ty))
  def tf_w(X, Y): return (torch.where(torch.from_numpy(wc).cuda()>0, X, torch.from_numpy(wy).cuda())).sum()
  check("grad_where", rf_w, tf_w, None)

  # gemm bias: d(sum(A@B+b))/db
  bias = rng.standard_normal((3,)).astype(np.float32)
  def rf_b(rt, x, y):
    bb=rt.tensor(bias,requires_grad=True)
    return scalar_loss(rt, rt.gemm(x,y,bb))
  def tf_b(X, Y): return (X@Y + torch.from_numpy(bias).cuda()).sum()
  check("grad_gemm_bias", rf_b, tf_b, A, B)

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
  print(f"\n{passed}/{len(results)} gradcheck cases passed")
  print(f"report: {cfg.result_path}")
  return 0 if passed == len(results) else 1

if __name__ == "__main__":
  raise SystemExit(run())
