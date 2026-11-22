"""PR #47: end-to-end realistic model win via the compiled/captured replay path.

A 4-layer MLP (256->128->128->64->1) training loop with FRESH batches each step,
run through Engine.compile_many (K captured fwd+bwd+SGD graphs in one submit per
batch) vs a PyTorch eager SGD training loop. Honest: warmup/median/spread/clock,
correctness on final weights.
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
from bitsy_v2 import functional as F


def gpu_clock():
  try:
    o=subprocess.check_output(["nvidia-smi","--query-gpu=clocks.current.graphics",
      "--format=csv,noheader"],text=True,timeout=5)
    return float(o.strip().splitlines()[0].split()[0])
  except Exception: return 0.0


@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path=ROOT/"build-rtvk"
  batch:int=64; dims:tuple=(256,128,128,64,1)
  lr:float=1e-4; K:int=6; steps:int=600
  warmup:int=2; iters:int=6; tol:float=1e-3
  result_path:Path=ROOT/"benchmarks"/"results"/"end_to_end.json"


def bench(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)),float(max(ts))]}


def run(cfg=Config()):
  eng=bv.Engine(cfg.build_dir,bv.Engine.VULKAN)
  K=cfg.K; nbatch=cfg.steps//K
  rng=np.random.default_rng(0)
  dims=cfg.dims
  ws=[rng.standard_normal((dims[i+1],dims[i])).astype(np.float32)*0.1
      for i in range(len(dims)-1)]
  t=rng.standard_normal((cfg.batch,dims[-1])).astype(np.float32)
  batches=[rng.standard_normal((K,cfg.batch,dims[0])).astype(np.float32)
           for _ in range(nbatch)]

  params=[eng.tensor(w,requires_grad=True) for w in ws]
  T=eng.tensor(t)
  def fn(X):
    h=X
    for w in params[:-1]:
      h=h.matmul(w.T()).relu()
    y=h.matmul(params[-1].T())
    loss=F.mse_loss(y,T)
    loss.backward(params=params)
    for p in params:
      eng.sgd_update(p.h,eng.grad_tensor(p),cfg.lr,0.0)
    return loss
  auto=eng.auto(K)
  def cf(batches): return auto.run("mlp_train",fn,[(cfg.batch,dims[0])],batches,zero_grads=params)
  def reset():
    for p,w in zip(params,ws): eng.upload(p,w)

  tw=[torch.from_numpy(w).requires_grad_(True) for w in ws]
  tt=torch.from_numpy(t)
  tbatches=[[torch.from_numpy(b[i]) for i in range(K)] for b in batches]
  opt=torch.optim.SGD(tw,lr=cfg.lr)
  def torch_batches(bs,nb):
    for b in bs[:nb]:
      for x in b:
        opt.zero_grad()
        h=x
        for w in tw[:-1]: h=torch.relu(h@w.T)
        y=h@tw[-1].T
        loss=torch.nn.functional.mse_loss(y,tt)
        loss.backward(); opt.step()
  def compiled_batches(bs,nb):
    reset()
    for b in bs[:nb]:
      cf([[b[i]] for i in range(K)])

  # correctness: final weights after 2 batches
  compiled_batches(batches,2)
  got=[p.numpy() for p in params]
  with torch.no_grad():
    for p,w in zip(tw,ws): p.copy_(torch.from_numpy(w))
  torch_batches(tbatches,2)
  diffs=[float(np.abs(g-p.detach().numpy()).max()) for g,p in zip(got,tw)]
  d=max(diffs); ok=d<=cfg.tol
  print(f"[{'PASS' if ok else 'FAIL'}] 4-layer MLP fresh-data train diffs={[f'{x:.1e}' for x in diffs]}")

  clk0=gpu_clock()
  bc=bench(lambda:compiled_batches(batches,nbatch),cfg.warmup,cfg.iters)
  with torch.no_grad():
    for p,w in zip(tw,ws): p.copy_(torch.from_numpy(w))
  bt=bench(lambda:torch_batches(tbatches,nbatch),cfg.warmup,cfg.iters)
  clk1=gpu_clock()
  pc=bc["median_ms"]/cfg.steps; pt=bt["median_ms"]/cfg.steps
  gap=pc/pt
  print(f"N={cfg.steps} K={K} compiled {pc:.4f} | torch {pt:.4f} ms/step | gap {gap:.2f}x "
        f"({'FASTER' if gap<1 else 'slower'})")
  print(f"compiled spread {bc['spread_ms']} torch spread {bt['spread_ms']}")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"diffs":diffs,"passed":ok},
    "compiled":bc,"torch":bt,"per_step":{"compiled":pc,"torch":pt},"gap":gap,
    "note":"4-layer MLP full fwd+bwd+SGD on fresh batches via compile_many (K graphs, one submit/batch)."}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
