"""Break down the replay hot path: submit-only vs download vs PyTorch GPU/host.

Evidence-driven profiling for PR #6 — find where the 194x gap comes from before
optimizing. Dataclass config, no argparse.
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

def clock():
  try:
    o=subprocess.check_output(["nvidia-smi","--query-gpu=clocks.current.graphics","--format=csv,noheader"],text=True,timeout=5)
    return float(o.strip().splitlines()[0].split()[0])
  except Exception: return 0.0

@dataclass(frozen=True,slots=True)
class Cfg:
  build_dir:Path=ROOT/"build-rtvk"; device:int=bv.Engine.VULKAN
  n:int=1<<20; warmup:int=5; iters:int=30
  result_path:Path=ROOT/"benchmarks"/"results"/"compact_profile.json"

def bench(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),"spread_ms":[float(min(ts)),float(max(ts))]}

def run(cfg=Cfg()):
  if not torch.cuda.is_available(): print("skip"); return 0
  eng=bv.Engine(cfg.build_dir,cfg.device)
  rng=np.random.default_rng(0)
  x=rng.standard_normal(cfg.n).astype(np.float32)
  X=eng.tensor(x)
  eng.capture_begin(); y=((X*X)+X).relu(); y.realize(); eng.capture_end()
  tx=torch.from_numpy(x).cuda()
  def t_gpu(): return torch.relu(tx*tx+tx)
  def t_host(): return torch.relu(tx*tx+tx).cpu().numpy()

  c0=clock()
  r_submit=bench(lambda:eng.replay_submit_only(),cfg.warmup,cfg.iters)
  r_full=bench(lambda:eng.replay(),cfg.warmup,cfg.iters)
  t_gpu_r=bench(t_gpu,cfg.warmup,cfg.iters)
  t_host_r=bench(t_host,cfg.warmup,cfg.iters)
  c1=clock()

  rep={"generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"clock":[c0,c1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "replay_submit_only":r_submit,"replay_full_with_download":r_full,
    "torch_gpu_only":t_gpu_r,"torch_with_download":t_host_r,
    "download_cost_ms":r_full["median_ms"]-r_submit["median_ms"],
    "gaps":{"torch_gpu_vs_submit":r_submit["median_ms"]/t_gpu_r["median_ms"],
            "torch_gpu_vs_full":r_full["median_ms"]/t_gpu_r["median_ms"]}}
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"replay submit-only {r_submit['median_ms']:.3f} ms")
  print(f"replay full+download {r_full['median_ms']:.3f} ms  (download ~{rep['download_cost_ms']:.3f} ms)")
  print(f"torch GPU-only {t_gpu_r['median_ms']:.3f} ms; torch+download {t_host_r['median_ms']:.3f} ms")
  print(f"gap: torch_gpu vs submit {rep['gaps']['torch_gpu_vs_submit']:.1f}x; vs full {rep['gaps']['torch_gpu_vs_full']:.1f}x")
  print(f"clock {c0:.0f}->{c1:.0f} MHz")
  return 0

if __name__=="__main__": raise SystemExit(run())
