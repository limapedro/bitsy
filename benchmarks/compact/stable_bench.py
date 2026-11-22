"""PR #64: stable single-op benchmark with rotation to cancel warmup bias.

Earlier harnesses measured cases in a fixed order and the first cases always
looked slower (add/mul) while identical-kernel later cases looked fast (sub/div)
— reversing the order flipped which cases were slow. The fix: rotate the case
order every round so each case is measured at every position, discard the first
(warmup) rotation, record the GPU clock each round and drop clock-changing
rounds, then report median-of-rounds with spread and win counts.
"""
from __future__ import annotations
import json, os, statistics, subprocess, sys, time
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
  except Exception:
    return 0.0


@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path=ROOT/"build-rtvk"
  n:int=1<<20
  rounds:int=7; warmup:int=5; iters:int=12
  result_path:Path=ROOT/"benchmarks"/"results"/"stable.json"


def one(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return statistics.median(ts)


def run(cfg=Config()):
  rng=np.random.default_rng(0)
  a=rng.standard_normal(cfg.n).astype(np.float32)
  b=(rng.standard_normal(cfg.n)*0.5+0.5).astype(np.float32)
  ta,tb=torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda()  # CUDA first
  eng=bv.Engine(cfg.build_dir,bv.Engine.VULKAN)
  X,Y=eng.tensor(a),eng.tensor(b)

  cases={
    "add":  (lambda:X+Y,                 lambda:ta+tb),
    "mul":  (lambda:X*Y,                 lambda:ta*tb),
    "sub":  (lambda:X-Y,                 lambda:ta-tb),
    "div":  (lambda:X/Y,                 lambda:ta/tb),
    "relu": (lambda:X.relu(),            lambda:torch.relu(ta)),
    "exp":  (lambda:X.exp(),             lambda:torch.exp(ta)),
    "sum":  (lambda:X.sum(axis=0),       lambda:ta.sum()),
    "mean": (lambda:X.mean(axis=0),      lambda:ta.mean()),
    "max":  (lambda:X.max(axis=0),       lambda:ta.max()),
    "fused":(lambda:(X+X)*Y+X,           lambda:(ta+ta)*tb+ta),
  }
  items=list(cases.items())
  samples={name:[] for name,_ in items}
  diffs={}

  def calls(bf,tf):
    def bcall():
      v=bf()
      return v.numpy() if hasattr(v,"numpy") else eng.get(v,(cfg.n,))
    def tcall():
      v=tf()
      return v.cpu().numpy() if hasattr(v,"cpu") else np.asarray(v)
    return bcall,tcall

  # rotation 0 = warmup only; rotations 1..cfg.rounds measured
  for r in range(cfg.rounds+1):
    off=r % len(items)
    order=items[off:]+items[:off]
    for name,(bf,tf) in order:
      bcall,tcall=calls(bf,tf)
      if name not in diffs:
        g=np.asarray(bcall()).ravel(); w=np.asarray(tcall()).ravel()
        diffs[name]=float(np.abs(g-w).max())
      if r==0:
        one(tcall,cfg.warmup,cfg.iters); one(bcall,cfg.warmup,cfg.iters)
        continue
      c0=gpu_clock()
      tm=one(tcall,cfg.warmup,cfg.iters)
      bm=one(bcall,cfg.warmup,cfg.iters)
      c1=gpu_clock()
      if c0 and c1 and c0!=c1:
        continue
      samples[name].append((bm,tm,bm/tm))

  out={}
  for name,_ in items:
    s=samples[name]
    if not s:
      out[name]={"diff":diffs[name],"kept":0}; print(f"{name:10s} no stable rounds"); continue
    ratios=[x[2] for x in s]; bb=[x[0] for x in s]; tt=[x[1] for x in s]
    med=statistics.median(ratios)
    out[name]={"diff":diffs[name],"kept":len(s),
      "bitsy_ms":statistics.median(bb),"torch_ms":statistics.median(tt),
      "bitsy_spread":[min(bb),max(bb)],"torch_spread":[min(tt),max(tt)],
      "ratio_median":med,"ratio_spread":[min(ratios),max(ratios)],
      "wins":sum(1 for x in ratios if x<1.0)}
    print(f"{name:10s} bitsy {out[name]['bitsy_ms']:.4f} torch {out[name]['torch_ms']:.4f} "
          f"ratio {med:.2f}x [{min(ratios):.2f},{max(ratios):.2f}] wins {out[name]['wins']}/{len(s)} diff {diffs[name]:.1e}")

  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER"},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "method":"rotated case order each round (each case measured at every position); warmup rotation discarded; clock recorded per round, clock-changing rounds dropped; median of round ratios",
    "cases":out}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  return 0

if __name__=="__main__": raise SystemExit(run())
