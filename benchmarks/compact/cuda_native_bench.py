"""PR #50: CUDA-native pointwise execution (explicit opt-in) vs PyTorch CUDA.

Every single eager op, warmup/median/spread/clock. The opt-in is required
because mixed CUDA/Vulkan graphs are unsafe; this measures the narrow pointwise
fast path where single-call eager calls can beat PyTorch's add+cpu baseline.
"""
from __future__ import annotations
import json, statistics, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import bitsy_v2 as bv


def gpu_clock():
  try:
    o=subprocess.check_output(["nvidia-smi","--query-gpu=clocks.current.graphics",
      "--format=csv,noheader"],text=True,timeout=5)
    return float(o.strip().splitlines()[0].split()[0])
  except Exception: return 0.0


@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path=ROOT/"build-rtvk"
  n:int=1<<20
  warmup:int=8; iters:int=30; tol:float=1e-3
  result_path:Path=ROOT/"benchmarks"/"results"/"cuda_native.json"


def bench(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)),float(max(ts))]}


CASES = {
  "add":   (lambda X,Y: X+Y,            lambda a,b: a+b),
  "mul":   (lambda X,Y: X*Y,            lambda a,b: a*b),
  "sub":   (lambda X,Y: X-Y,            lambda a,b: a-b),
  "div":   (lambda X,Y: X/Y,            lambda a,b: a/b),
  "relu":  (lambda X,Y: X.relu(),       lambda a,b: torch.relu(a)),
  "sigmoid":(lambda X,Y: X.sigmoid(),   lambda a,b: torch.sigmoid(a)),
  "tanh":  (lambda X,Y: X.tanh(),       lambda a,b: torch.tanh(a)),
  "exp":   (lambda X,Y: X.exp(),        lambda a,b: torch.exp(a)),
  "fused": (lambda X,Y: (X+X)*Y+X,      lambda a,b: (a+a)*b+a),
}

def run(cfg=Config()):
  rng=np.random.default_rng(0)
  a=rng.standard_normal(cfg.n).astype(np.float32)
  b=(rng.standard_normal(cfg.n)*0.5+0.5).astype(np.float32)
  # Initialize CUDA through PyTorch first so bitsy adopts the same context
  # (a second CUDA context roughly doubles single-call wall time).
  tx,tb=torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda()
  eng=bv.Engine(cfg.build_dir,bv.Engine.VULKAN)
  if not eng.use_cuda_native(True):
    print("CUDA unavailable; skipping"); return 0
  X,Y=eng.tensor(a),eng.tensor(b)
  results=[]
  clk0=gpu_clock()
  for name,(bf,tf) in CASES.items():
    got=bf(X,Y).numpy(); want=tf(tx,tb).cpu().numpy()
    # keep torch reference synced like common_ops_bench

    d=float(np.abs(got-want).max()); ok=d<=cfg.tol
    bb=bench(lambda:bf(X,Y).numpy(),cfg.warmup,cfg.iters)
    tt=bench(lambda:tf(tx,tb).cpu().numpy(),cfg.warmup,cfg.iters)
    gap=bb["median_ms"]/tt["median_ms"]
    results.append({"name":name,"diff":d,"passed":ok,
                    "bitsy_ms":bb["median_ms"],"torch_ms":tt["median_ms"],
                    "gap":gap,"faster":gap<1.0})
    print(f"[{'PASS' if ok else 'FAIL'}] {name:9s} bitsy {bb['median_ms']:.3f} "
          f"torch {tt['median_ms']:.3f} ms | gap {gap:.2f}x "
          f"({'FASTER' if gap<1 else 'slower'})")
  clk1=gpu_clock()
  faster=sum(r["faster"] for r in results)
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz | faster {faster}/{len(results)}")
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "results":results,"faster":faster,"total":len(results)}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  return 0 if all(r["passed"] for r in results) else 1

if __name__=="__main__": raise SystemExit(run())
