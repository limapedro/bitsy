"""PR #36: capture the ENTIRE update loop (forward+backward+device SGD) into
one Vulkan command buffer, replay N times in ONE submit. Benchmark end-to-end
vs PyTorch CUDA SGD. Honest: warmup/median/spread/clock.
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
  lr:float=0.01; steps:tuple=(10,100,500)
  warmup:int=5; iters:int=20; tol:float=1e-3
  result_path:Path=ROOT/"benchmarks"/"results"/"train_loop_replay.json"


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
  # capture the whole update: forward + backward + in-place device SGD
  eng.capture_begin()
  y=fwd(); y.backward(params=[A,B])
  eng.sgd_update(A.h,eng.grad_tensor(A),cfg.lr,0.0)
  eng.sgd_update(B.h,eng.grad_tensor(B),cfg.lr,0.0)
  eng.capture_end()

  # reset weights: capture-time execution applied one step already
  def reset():
    eng.upload(A,w1); eng.upload(B,w2)
  reset()
  eng.replay_n(1)
  ga=eng.get(A.h,(cfg.hidden,cfg.in_f)); gb=eng.get(B.h,(cfg.out_f,cfg.hidden))
  reset()

  tx=torch.from_numpy(x); tw1=torch.from_numpy(w1).requires_grad_(True)
  tw2=torch.from_numpy(w2).requires_grad_(True); tt=torch.from_numpy(t)
  opt=torch.optim.SGD([tw1,tw2],lr=cfg.lr)
  def torch_step():
    opt.zero_grad()
    o=torch.nn.functional.mse_loss(torch.relu(tx@tw1.T)@tw2.T,tt)
    o.backward(); opt.step()
    return o
  torch_step()
  d1=float(np.abs(ga-tw1.detach().numpy()).max())
  d2=float(np.abs(gb-tw2.detach().numpy()).max())
  ok=d1<=cfg.tol and d2<=cfg.tol
  print(f"[{'PASS' if ok else 'FAIL'}] 1-step capture vs torch SGD: w1={d1:.2e} w2={d2:.2e}")

  clk0=gpu_clock(); results=[]
  for N in cfg.steps:
    bn=bench(lambda:(reset(),eng.replay_n(N),eng.get(A.h,(cfg.hidden,cfg.in_f)),
                     eng.get(B.h,(cfg.out_f,cfg.hidden))),cfg.warmup,cfg.iters)
    bt=bench(lambda:[torch_step() for _ in range(N)],cfg.warmup,cfg.iters)
    pn,pt=(v["median_ms"]/N for v in (bn,bt))
    results.append({"N":N,"batch_one_submit":bn,"torch":bt,
      "per_step":{"batch":pn,"torch":pt},"gap_batch_vs_torch":pn/pt})
    print(f"N={N:4d} batch {pn:.4f} | torch {pt:.4f} ms/step | gap {pn/pt:.2f}x "
          f"({'FASTER' if pn<pt else 'slower'})")
  clk1=gpu_clock()
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"w1":d1,"w2":d2,"passed":ok},"results":results,
    "note":"Entire fwd+bwd+SGD update captured into one command buffer, N steps in one submit."}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
