"""Capture/replay correctness + speed benchmark for the v0.2 compact runtime.

Correctness: capture a fused pointwise chain, replay with new input data,
compare to PyTorch CUDA (both full replay and lazy submit+download).
Speed: two honest measurements —
  (a) single-step with forced host download (what a user reading every output pays),
  (b) N GPU-only steps then one final download (the realistic training/loop pattern,
      fair vs PyTorch GPU-only).
Warmup, median, spread and live GPU clock. Dataclass config, no argparse.
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

def gpu_clock_mhz():
  try:
    o=subprocess.check_output(["nvidia-smi","--query-gpu=clocks.current.graphics","--format=csv,noheader"],text=True,timeout=5)
    return float(o.strip().splitlines()[0].split()[0])
  except Exception: return 0.0

@dataclass(frozen=True,slots=True)
class Config:
  build_dir:Path=ROOT/"build-rtvk"; device:int=bv.Engine.VULKAN
  n:int=1<<20; warmup:int=5; iters:int=30; steps:int=10
  tol:float=1e-4
  result_path:Path=ROOT/"benchmarks"/"results"/"compact_replay.json"

def md(a,b): return float(np.abs(np.asarray(a,np.float32)-np.asarray(b,np.float32)).max())
def bench(fn,w,it):
  for _ in range(w): fn()
  ts=[]
  for _ in range(it):
    s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
  return {"median_ms":float(statistics.median(ts)),"spread_ms":[float(min(ts)),float(max(ts))]}

def run(cfg=Config()):
  if not torch.cuda.is_available(): print("skip"); return 0
  eng=bv.Engine(cfg.build_dir,cfg.device)
  rng=np.random.default_rng(0); x=rng.standard_normal(cfg.n).astype(np.float32)
  X=eng.tensor(x)
  eng.capture_begin(); y=((X*X)+X).relu(); y.realize(); eng.capture_end()

  tx=torch.from_numpy(x).cuda()
  def torch_chain(): return torch.relu(tx*tx+tx)

  # ---- correctness ----
  eng.replay(); got=y.numpy(); want=torch_chain().cpu().numpy()
  d1=md(got,want)
  x2=rng.standard_normal(cfg.n).astype(np.float32); eng.upload(X,x2); eng.replay()
  d2=md(y.numpy(), torch.relu(torch.from_numpy(x2).cuda()**2+torch.from_numpy(x2).cuda()).cpu().numpy())
  # lazy path: submit-only then read
  eng.upload(X,x); eng.replay_submit_only(); d3=md(y.numpy(), want)
  ok = all(d<=cfg.tol for d in (d1,d2,d3))
  print(f"[{'PASS' if d1<=cfg.tol else 'FAIL'}] replay diff={d1:.3e}")
  print(f"[{'PASS' if d2<=cfg.tol else 'FAIL'}] replay(new data) diff={d2:.3e}")
  print(f"[{'PASS' if d3<=cfg.tol else 'FAIL'}] submit-only+lazy-read diff={d3:.3e}")

  # ---- speed (a): single step with forced download ----
  clk0=gpu_clock_mhz()
  eager=bench(lambda:(((X*X)+X).relu().numpy()),cfg.warmup,cfg.iters)
  full=bench(lambda:eng.replay(),cfg.warmup,cfg.iters)
  t_full=bench(lambda:torch_chain().cpu().numpy(),cfg.warmup,cfg.iters)

  # ---- speed (b): N async GPU steps + one final read (fair vs PyTorch async) ----
  # Both frameworks submit N steps asynchronously and pay one final sync+read.
  # (replay_submit_only would add an artificial per-step sync that torch lacks.)
  def bitsy_loop():
    for _ in range(cfg.steps): eng.replay_submit_async()
    y.numpy()   # one final download (waits for all pending submits)
  def bitsy_loop_sync():
    for _ in range(cfg.steps): eng.replay_submit_only()
    y.numpy()
  def torch_loop():
    o=None
    for _ in range(cfg.steps): o=torch_chain()
    return o.cpu().numpy()
  loop=bench(bitsy_loop,cfg.warmup,cfg.iters)
  loop_sync=bench(bitsy_loop_sync,cfg.warmup,cfg.iters)
  t_loop=bench(torch_loop,cfg.warmup,cfg.iters)
  clk1=gpu_clock_mhz()

  per_step_bitsy=loop["median_ms"]/cfg.steps
  per_step_torch=t_loop["median_ms"]/cfg.steps
  rep={
    "generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"clock":[clk0,clk1]},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "correctness":{"replay":d1,"new_data":d2,"submit_lazy_read":d3,"passed":ok},
    "single_step_forced_download":{"bitsy_eager":eager,"bitsy_replay":full,"torch":t_full,
       "replay_vs_eager":eager["median_ms"]/full["median_ms"],
       "torch_vs_replay":full["median_ms"]/t_full["median_ms"]},
    "loop_gpu_only_plus_final_read":{"bitsy":loop,"bitsy_sync_submit":loop_sync,"torch":t_loop,
       "bitsy_per_step_ms":per_step_bitsy,"torch_per_step_ms":per_step_torch,
       "torch_vs_bitsy_per_step":per_step_bitsy/per_step_torch,
       "sync_vs_async_per_step":loop_sync["median_ms"]/loop["median_ms"]},
    "note":"ratios >1 mean first arg faster. Async loop is the fair comparison: both frameworks submit N steps without a per-step sync and pay one final read. bitsy_sync_submit shows the cost of the legacy per-step-wait API.",
  }
  cfg.result_path.parent.mkdir(parents=True,exist_ok=True)
  cfg.result_path.write_text(json.dumps(rep,indent=2)+"\n",encoding="utf-8")
  print(f"single-step forced download: eager {eager['median_ms']:.3f} replay {full['median_ms']:.3f} "
        f"({rep['single_step_forced_download']['replay_vs_eager']:.2f}x vs eager) torch {t_full['median_ms']:.3f} ms")
  print(f"loop ({cfg.steps} GPU steps + read): bitsy {loop['median_ms']:.3f} ms "
        f"({per_step_bitsy:.4f} ms/step) torch {t_loop['median_ms']:.3f} ms "
        f"({per_step_torch:.4f} ms/step) gap {rep['loop_gpu_only_plus_final_read']['torch_vs_bitsy_per_step']:.1f}x")
  print(f"clock {clk0:.0f}->{clk1:.0f} MHz")
  return 0 if ok else 1

if __name__=="__main__": raise SystemExit(run())
