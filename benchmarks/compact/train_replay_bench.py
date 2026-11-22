"""PR #35: capture whole forward+backward steps, replay N times in ONE submit.
Benchmark vs PyTorch CUDA eager training loop. Honest: warmup/median/spread/clock.
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
  # out_f=1 so mse_loss's axis-0 mean is a true scalar (grad correctness).
  batch:int=64; in_f:int=256; hidden:int=128; out_f:int=1
  steps:tuple=(10,100,500)
  warmup:int=5; iters:int=20; tol:float=1e-3
  result_path:Path=ROOT/"benchmarks"/"results"/"train_replay.json"


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
  t=rng.standard_normal((cfg.batch,cfg.out_f)).astype(np.float32)
  X=eng.tensor(x); A=eng.tensor(w1,requires_grad=True)
  B=eng.tensor(w2,requires_grad=True); T=eng.tensor(t)
  def fwd():
    return bv.functional.mse_loss(X.matmul(A.T()).relu().matmul(B.T()),T)
  eng.capture_begin(); y=fwd(); y.backward(params=[A,B]); eng.capture_end()
  g1r=eng.rt.grad_refresh(A.h,(cfg.hidden,cfg.in_f))
  g2r=eng.rt.grad_refresh(B.h,(cfg.out_f,cfg.hidden))
  tx=torch.from_numpy(x).cuda(); tw1=torch.from_numpy(w1).cuda().requires_grad_(True)
  tw2=torch.from_numpy(w2).cuda().requires_grad_(True); tt=torch.from_numpy(t).cuda()
  def tfwd():
    return torch.nn.functional.mse_loss(torch.relu(tx@tw1.T)@tw2.T,tt)
  eng.replay_n(1); g1=eng.rt.grad_refresh(A.h,(cfg.hidden,cfg.in_f))
  g2=eng.rt.grad_refresh(B.h,(cfg.out_f,cfg.hidden))
  l=torch.nn.functional.mse_loss(torch.relu(tx@tw1.T)@tw2.T,tt)
  l.backward()
  d1=float(np.abs(g1-tw1.grad.cpu().numpy()).max())
  d2=float(np.abs(g2-tw2.grad.cpu().numpy()).max())
  ok=d1<=cfg.tol and d2<=cfg.tol
  print(f"[{'PASS' if ok else 'FAIL'}] train-step capture grads w1={d1:.2e} w2={d2:.2e}")

  def read_grads():
    eng.rt.grad_refresh(A.h,(cfg.hidden,cfg.in_f))
    eng.rt.grad_refresh(B.h,(cfg.out_f,cfg.hidden))
  def torch_step():
    o=torch.nn.functional.mse_loss(torch.relu(tx@tw1.T)@tw2.T,tt)
    o.backward(); tw1.grad=None; tw2.grad=None
    return o
  clk0=gpu_clock(); results=[]
  for N in cfg.steps:
    bn=bench(lambda:(eng.replay_n(N),read_grads()),cfg.warmup,cfg.iters)
    bl=bench(lambda:([eng.replay_submit_async() for _ in range(N)],read_grads()),cfg.warmup,cfg.iters)
    bt=bench(lambda:[torch_step() for _ in range(N)],cfg.warmup,cfg.iters)
    pn,pb,pt=(v["median_ms"]/N for v in (bn,bl,bt))
    results.append({"N":N,"batch_one_submit":bn,"n_submits":bl,"torch":bt,
      "per_step":{"batch":pn,"n_submits":pb,"torch":pt},
      "gap_batch_vs_torch":pn/pt,"speedup_vs_n_submits":pb/pn})
    print(f"N={N:4d} batch {pn:.4f} | N-submits {pb:.4f} | torch {pt:.4f} ms/step | "
          f"gap {pn/pt:.2f}x | speedup {pb/pn:.2f}x")
  clk1=gpu_clock()
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"w1":d1,"w2":d2,"passed":ok},"results":results,
    "note":"forward+backward captured into one command buffer; N steps in one submit."}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
