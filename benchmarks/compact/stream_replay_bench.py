"""PR #38: TRUE streaming update loop — K captured graphs, each reading a
distinct CPU-mapped input buffer, submitted in ONE vkQueueSubmit2 per batch of
K fresh steps. End-to-end vs PyTorch SGD over streaming batches.
Honest: warmup/median/spread/clock.
"""
from __future__ import annotations
import ctypes as C
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
  lr:float=1e-4; graphs:int=6; steps:int=600
  warmup:int=2; iters:int=6; tol:float=1e-3
  result_path:Path=ROOT/"benchmarks"/"results"/"stream_replay.json"


def bench(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)),float(max(ts))]}


def run(cfg=Config()):
  eng=bv.Engine(cfg.build_dir,bv.Engine.VULKAN)
  K=cfg.graphs
  rng=np.random.default_rng(0)
  x0=rng.standard_normal((cfg.batch,cfg.in_f)).astype(np.float32)
  w1=rng.standard_normal((cfg.hidden,cfg.in_f)).astype(np.float32)
  w2=rng.standard_normal((cfg.out_f,cfg.hidden)).astype(np.float32)
  t=rng.standard_normal((cfg.batch,cfg.out_f)).astype(np.float32)
  nbatch=cfg.steps//K
  batches=[rng.standard_normal((K,cfg.batch,cfg.in_f)).astype(np.float32)
           for _ in range(nbatch)]

  A=eng.tensor(w1,requires_grad=True); B=eng.tensor(w2,requires_grad=True)
  T=eng.tensor(t)
  xs=[]; ptrs=[]
  for i in range(K):
    X=eng.tensor(x0); xs.append(X); ptrs.append(eng.map_input(X))
    eng.stream_begin(i)
    y=bv.functional.mse_loss(X.matmul(A.T()).relu().matmul(B.T()),T)
    y.backward(params=[A,B])
    eng.sgd_update(A.h,eng.grad_tensor(A),cfg.lr,0.0)
    eng.sgd_update(B.h,eng.grad_tensor(B),cfg.lr,0.0)
    eng.stream_end(i)
    eng.zero_grad(A); eng.zero_grad(B)
  def reset(): eng.upload(A,w1); eng.upload(B,w2)

  def feed(b):
    for i in range(K):
      C.memmove(ptrs[i],b[i].ctypes.data_as(C.POINTER(C.c_float)),b[i].nbytes)
      eng.flush_input(xs[i])
  def bitsy_loop(nbatches):
    reset()
    for b in batches[:nbatches]:
      feed(b); eng.stream_submit(K)

  tx=[[torch.from_numpy(b[i]) for i in range(K)] for b in batches]
  tw1=torch.from_numpy(w1).requires_grad_(True)
  tw2=torch.from_numpy(w2).requires_grad_(True); tt=torch.from_numpy(t)
  opt=torch.optim.SGD([tw1,tw2],lr=cfg.lr)
  def torch_loop(nbatches):
    for b in tx[:nbatches]:
      for i in range(K):
        opt.zero_grad()
        o=torch.nn.functional.mse_loss(torch.relu(b[i]@tw1.T)@tw2.T,tt)
        o.backward(); opt.step()

  # correctness: 2 batches (=2K steps)
  bitsy_loop(2)
  ga=eng.get(A.h,(cfg.hidden,cfg.in_f)); gb=eng.get(B.h,(cfg.out_f,cfg.hidden))
  with torch.no_grad(): tw1.copy_(torch.from_numpy(w1)); tw2.copy_(torch.from_numpy(w2))
  torch_loop(2)
  d1=float(np.abs(ga-tw1.detach().numpy()).max())
  d2=float(np.abs(gb-tw2.detach().numpy()).max())
  ok=d1<=cfg.tol and d2<=cfg.tol
  print(f"[{'PASS' if ok else 'FAIL'}] streaming fresh-data ({2*K} steps) diff w1={d1:.2e} w2={d2:.2e}")

  clk0=gpu_clock()
  bn=bench(lambda:bitsy_loop(nbatch),cfg.warmup,cfg.iters)
  with torch.no_grad(): tw1.copy_(torch.from_numpy(w1)); tw2.copy_(torch.from_numpy(w2))
  bt=bench(lambda:torch_loop(nbatch),cfg.warmup,cfg.iters)
  clk1=gpu_clock()
  pn=bn["median_ms"]/cfg.steps; pt=bt["median_ms"]/cfg.steps
  gap=pn/pt
  print(f"N={cfg.steps:4d} K={K} bitsy {pn:.4f} | torch {pt:.4f} ms/step | gap {gap:.2f}x "
        f"({'FASTER' if gap<1 else 'slower'})")
  print(f"bitsy spread {bn['spread_ms']} torch spread {bt['spread_ms']}")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"w1":d1,"w2":d2,"passed":ok},
    "bitsy_stream":bn,"torch_stream":bt,"per_step":{"bitsy":pn,"torch":pt},"gap":gap,
    "note":"K captured fwd+bwd+SGD graphs, distinct CPU-mapped fresh inputs, K graphs in one submit per batch."}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
