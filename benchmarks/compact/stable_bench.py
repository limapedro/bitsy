"""PR #63: stable single-op benchmark methodology.

Prior sweeps fluctuated 0.7x-1.5x because GPU clocks drift and the two
frameworks were measured in separate blocks. This harness:
  * interleaves bitsy and torch inside every round (cancels thermal drift)
  * records the live GPU clock before/after every round and discards rounds
    where the clock changed
  * repeats R rounds, each with W warmups and I timed calls
  * reports median-of-round-medians plus full spread, and the ratio distribution
Dataclass config, no argparse.
"""
from __future__ import annotations
import json, statistics, subprocess, sys
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
  rounds:int=7; warmup:int=5; iters:int=15
  result_path:Path=ROOT/"benchmarks"/"results"/"stable.json"


def one(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=__import__("time").perf_counter(); fn(); ts.append((__import__("time").perf_counter()-s)*1e3)
  return statistics.median(ts)


def run(cfg=Config()):
  # torch creates the CUDA context first (bitsy adopts it -> no order penalty)
  rng=np.random.default_rng(0)
  a=rng.standard_normal(cfg.n).astype(np.float32)
  b=(rng.standard_normal(cfg.n)*0.5+0.5).astype(np.float32)
  ta,tb=torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda()
  eng=bv.Engine(cfg.build_dir,bv.Engine.VULKAN)
  X,Y=eng.tensor(a),eng.tensor(b)

  cases={
    "add":  (lambda:X+Y,                 lambda:ta+tb),
    "mul":  (lambda:X*Y,                 lambda:ta*tb),
    "sub":  (lambda:X-Y,                 lambda:ta-tb),
    "div":  (lambda:X/Y,                 lambda:ta/tb),
    "relu": (lambda:X.relu(),            lambda:torch.relu(ta)),
    "exp":  (lambda:eng.rt.unary("exp",X.h) if False else X.exp(), lambda:torch.exp(ta)),
    "sum":  (lambda:X.sum(axis=0),       lambda:ta.sum()),
    "mean": (lambda:X.mean(axis=0),      lambda:ta.mean()),
    "max":  (lambda:X.max(axis=0),       lambda:ta.max()),
    "fused":(lambda:(X+X)*Y+X,           lambda:(ta+ta)*tb+ta),
  }
  out={}
  # global warmup: exercise every case once so CUDA/driver state is settled
  # before any timed round (otherwise the first cases pay context/module init).
  for _bf,_tf in cases.values():
    _v=_bf(); _v.numpy() if hasattr(_v,"numpy") else eng.get(_v,(cfg.n,))
    _t=_tf(); _t.cpu().numpy() if hasattr(_t,"cpu") else None
  # wait for clock to settle (busy work until 4 consecutive identical readings)
  hist=[]
  for _ in range(200):
    _v=X+Y; _v.numpy()
    hist.append(gpu_clock())
    if len(hist)>=4 and len(set(hist[-4:]))==1: break
  for name,(bf,tf) in cases.items():
    # wrap handles so unary cases return a Tensor with numpy()
    def bcall():
      v=bf()
      return v.numpy() if hasattr(v,"numpy") else eng.get(v, (cfg.n,))
    def tcall():
      v=tf()
      return v.cpu().numpy() if hasattr(v,"cpu") else np.asarray(v)
    # correctness once
    g=np.asarray(bcall()).ravel(); w=np.asarray(tcall()).ravel()
    diff=float(np.abs(g-w).max())
    ratios=[]; bb=[]; tt=[]; kept=0
    for r in range(cfg.rounds):
      c0=gpu_clock()
      tb_ms=one(tcall,cfg.warmup,cfg.iters)
      tb_bitsy=one(bcall,cfg.warmup,cfg.iters)
      c1=gpu_clock()
      if c0 and c1 and c0!=c1:   # discard clock-changing round
        continue
      kept+=1; bb.append(tb_bitsy); tt.append(tb_ms); ratios.append(tb_bitsy/tb_ms)
    if not kept:
      out[name]={"diff":diff,"kept":0}; print(f"{name:10s} no stable rounds"); continue
    med=statistics.median(ratios)
    out[name]={"diff":diff,"kept":kept,
      "bitsy_ms":statistics.median(bb),"torch_ms":statistics.median(tt),
      "bitsy_spread":[min(bb),max(bb)],"torch_spread":[min(tt),max(tt)],
      "ratio_median":med,"ratio_spread":[min(ratios),max(ratios)],
      "wins":sum(1 for x in ratios if x<1.0)}
    print(f"{name:10s} bitsy {out[name]['bitsy_ms']:.4f} torch {out[name]['torch_ms']:.4f} "
          f"ratio {med:.2f}x [{min(ratios):.2f},{max(ratios):.2f}] wins {out[name]['wins']}/{kept} diff {diff:.1e}")
  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":"GTX 1660 SUPER"},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "method":"interleaved bitsy/torch per round; clock recorded per round; clock-changing rounds discarded; median of round medians",
    "cases":out}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  return 0

if __name__=="__main__": raise SystemExit(run())
