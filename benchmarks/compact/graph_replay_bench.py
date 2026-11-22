"""Batched graph replay benchmark (PR #32): N steps in ONE submit vs N submits
vs PyTorch. Honest: warmup, median, spread, clock.

Uses the fused pointwise chain (the capture path proven correct by replay_bench).
Note: GEMM/MLP multi-dispatch capture is a known pre-existing limitation and is
documented in BENCHMARKS.md; it is not used here.
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
    o = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.current.graphics",
      "--format=csv,noheader"], text=True, timeout=5)
    return float(o.strip().splitlines()[0].split()[0])
  except Exception:
    return 0.0


@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path = ROOT / "build-rtvk"
  n:int = 1 << 20
  steps:tuple = (10, 100, 500)
  warmup:int = 5
  iters:int = 20
  tol:float = 1e-4
  result_path:Path = ROOT / "benchmarks" / "results" / "graph_replay.json"


def bench(fn, w, it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),
          "spread_ms":[float(min(ts)),float(max(ts))]}


def run(cfg=Config()):
  eng=bv.Engine(cfg.build_dir,bv.Engine.VULKAN)
  x=np.random.default_rng(0).standard_normal(cfg.n).astype(np.float32)
  X=eng.tensor(x)
  eng.capture_begin(); y=((X*X)+X).relu(); y.realize(); eng.capture_end()
  tx=torch.from_numpy(x).cuda()

  # correctness (max N)
  eng.replay_n(max(cfg.steps)); got=y.numpy()
  want=torch.relu(tx*tx+tx).cpu().numpy()
  diff=float(np.abs(got-want).max())
  ok=diff<=cfg.tol
  print(f"[{'PASS' if ok else 'FAIL'}] replay_n({max(cfg.steps)}) diff={diff:.3e}")

  clk0=gpu_clock()
  results=[]
  for N in cfg.steps:
    bn=bench(lambda:(eng.replay_n(N),y.numpy()),cfg.warmup,cfg.iters)
    bl=bench(lambda:([eng.replay_submit_async() for _ in range(N)],y.numpy()),
             cfg.warmup,cfg.iters)
    bt=bench(lambda:(lambda o:o.cpu().numpy())([torch.relu(tx*tx+tx) for _ in range(N)][-1]),
             cfg.warmup,cfg.iters)
    pn,pb,pt=(v["median_ms"]/N for v in (bn,bl,bt))
    results.append({"N":N,"batch_one_submit":bn,"n_submits":bl,"torch":bt,
                    "per_step":{"batch":pn,"n_submits":pb,"torch":pt},
                    "gap_batch_vs_torch":pn/pt,"speedup_batch_vs_nsubmits":pb/pn})
    print(f"N={N:4d} batch {pn:.4f} ms/step | N-submits {pb:.4f} | torch {pt:.4f} | "
          f"gap(batch) {pn/pt:.2f}x | batch speedup {pb/pn:.2f}x")
  clk1=gpu_clock()
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER","clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"diff":diff,"passed":ok},"results":results,
    "note":"gap>1 = bitsy slower. batch_one_submit amortises host submit/fence over N steps."}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
