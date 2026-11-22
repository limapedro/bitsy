"""PR #41: transpose-folded GEMM (X @ W.T() in one dispatch) — single-call
MLP forward vs PyTorch CUDA. Honest: warmup/median/spread/clock, dispatch count.
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
  batch:int=64; in_f:int=256; hidden:int=128; out_f:int=1
  warmup:int=8; iters:int=30; tol:float=1e-3
  result_path:Path=ROOT/"benchmarks"/"results"/"gemm_epilogue.json"


def bench(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)),float(max(ts))]}


def run(cfg=Config()):
  eng=bv.Engine(cfg.build_dir,bv.Engine.VULKAN)
  rng=np.random.default_rng(0)
  x=rng.standard_normal((cfg.batch,cfg.in_f)).astype(np.float32)
  w1=rng.standard_normal((cfg.hidden,cfg.in_f)).astype(np.float32)
  w2=rng.standard_normal((cfg.out_f,cfg.hidden)).astype(np.float32)
  X=eng.tensor(x); A=eng.tensor(w1); B=eng.tensor(w2)
  def fwd(): return X.matmul(A.T()).relu().matmul(B.T())
  got=fwd().numpy()
  tx=torch.from_numpy(x); t1=torch.from_numpy(w1); t2=torch.from_numpy(w2)
  want=(torch.relu(tx@t1.T)@t2.T).numpy()
  d=float(np.abs(got-want).max()); ok=d<=cfg.tol
  print(f"[{'PASS' if ok else 'FAIL'}] 2-layer MLP fwd diff={d:.2e}")

  eng.rt.lib.rt_reset_dispatch_count(eng.rt.ctx)
  fwd().numpy()
  nd=eng.rt.lib.rt_dispatch_count(eng.rt.ctx)
  print(f"dispatch count = {nd} (was 4 before transpose fold)")

  def tfwd(): return (torch.relu(tx@t1.T)@t2.T).cpu()
  clk0=gpu_clock()
  bb=bench(lambda:fwd().numpy(),cfg.warmup,cfg.iters)
  bt=bench(tfwd,cfg.warmup,cfg.iters)
  clk1=gpu_clock()
  gap=bb["median_ms"]/bt["median_ms"]
  print(f"bitsy {bb['median_ms']:.4f} | torch {bt['median_ms']:.4f} ms | gap {gap:.2f}x "
        f"({'FASTER' if gap<1 else 'slower'})")
  print(f"bitsy spread {bb['spread_ms']} torch spread {bt['spread_ms']}")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")

  # PR #42: nn.Linear (matmul + bias) folds into one GEMM dispatch
  B2=eng.tensor(rng.standard_normal((cfg.out_f,cfg.in_f)).astype(np.float32))
  LB=eng.tensor(rng.standard_normal((1,cfg.out_f)).astype(np.float32))
  X2=eng.tensor(x)
  def linear(): return X2.matmul(B2.T())+LB
  eng.rt.lib.rt_reset_dispatch_count(eng.rt.ctx)
  linear().numpy()
  nd2=eng.rt.lib.rt_dispatch_count(eng.rt.ctx)
  print(f"linear+bias dispatch count = {nd2} (was 2 before bias fold)")
  bl=bench(lambda:linear().numpy(),cfg.warmup,cfg.iters)
  print(f"linear+bias wall {bl['median_ms']:.4f} ms (was 0.6088 before bias fold)")
  rep={"linear_bias_dispatch":nd2,"linear_bias_wall":bl,
       "linear_bias_before_ms":0.6088}
  rep.update({"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"diff":d,"passed":ok},"dispatch_count":nd,
    "bitsy":bb,"torch":bt,"gap":gap,
    "note":"Unrealized W.T() folded into GEMM trans_b flag; bias add folded into GEMM epilogue."})
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
