"""PR #48: auto-capture adaptive path for inference/single ops vs PyTorch.

Eager forward-only model inference through Engine.auto() (which selects eager
execution for inference workloads, since captured replay was measured slower
there) vs PyTorch eager inference. Honest: warmup/median/spread/clock.
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
  batch:int=64; in_f:int=256; out_f:int=1
  steps:int=200; warmup:int=5; iters:int=15; tol:float=1e-3
  result_path:Path=ROOT/"benchmarks"/"results"/"inference.json"


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
  w1=rng.standard_normal((128,cfg.in_f)).astype(np.float32)*0.1
  w2=rng.standard_normal((64,128)).astype(np.float32)*0.1
  w3=rng.standard_normal((cfg.out_f,64)).astype(np.float32)*0.1
  xs=[rng.standard_normal((cfg.batch,cfg.in_f)).astype(np.float32)
      for _ in range(cfg.steps)]

  P=[eng.tensor(w) for w in (w1,w2,w3)]
  def fn(X):
    h=X.matmul(P[0].T()).relu().matmul(P[1].T()).relu().matmul(P[2].T())
    return h
  auto=eng.auto()
  # correctness on a fresh batch
  got=auto.infer("infer",fn,[(cfg.batch,cfg.in_f)],[xs[0]])[0][0].numpy()
  t1=torch.from_numpy(w1); t2=torch.from_numpy(w2); t3=torch.from_numpy(w3)
  want=(torch.relu(torch.relu(torch.from_numpy(xs[0])@t1.T)@t2.T)@t3.T).numpy()
  d=float(np.abs(got-want).max()); ok=d<=cfg.tol
  print(f"[{'PASS' if ok else 'FAIL'}] inference auto path diff={d:.2e}")

  def bloop(): auto.infer("infer",fn,[(cfg.batch,cfg.in_f)],xs)
  def tloop():
    for x in xs:
      tx=torch.from_numpy(x)
      torch.relu(torch.relu(tx@t1.T)@t2.T)@t3.T
  clk0=gpu_clock()
  bb=bench(bloop,cfg.warmup,cfg.iters)
  bt=bench(tloop,cfg.warmup,cfg.iters)
  clk1=gpu_clock()
  pb=bb["median_ms"]/cfg.steps; pt=bt["median_ms"]/cfg.steps
  print(f"N={cfg.steps} auto-eager {pb:.4f} | torch {pt:.4f} ms/step | gap {pb/pt:.2f}x "
        f"({'FASTER' if pb<pt else 'slower'})")
  print(f"auto spread {bb['spread_ms']} torch spread {bt['spread_ms']}")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"diff":d,"passed":ok},"auto":bb,"torch":bt,
    "per_step":{"auto":pb,"torch":pt},"gap":pb/pt}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
