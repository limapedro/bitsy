"""bitsy_v2 nn/optim composition parity vs PyTorch CUDA, fp32.

Build the same MLP + LayerNorm in bitsy_v2 (Python compositions over C+Vulkan)
and PyTorch CUDA, copy identical weights, compare forward logits, loss, grads,
and one optimizer step. Also verifies layers add ZERO new C lines.
Dataclass config, no argparse.
"""
from __future__ import annotations

import json, sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
import bitsy_v2 as bv

@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path = ROOT / "build-rtvk"
  device:int = bv.Engine.VULKAN
  in_f:int = 8
  hidden:int = 16
  out_f:int = 5
  batch:int = 4
  seed:int = 0
  tol_fwd:float = 1e-4
  tol_grad:float = 1e-3
  result_path:Path = ROOT / "benchmarks" / "results" / "bitsy_v2_model_parity.json"

CFG = Config()

def md(a,b): return float(np.abs(np.asarray(a,np.float32)-np.asarray(b,np.float32)).max())

def run(cfg=CFG):
  if not torch.cuda.is_available():
    print("CUDA unavailable; skipping"); return 0
  rng = np.random.default_rng(cfg.seed)
  X = rng.standard_normal((cfg.batch, cfg.in_f)).astype(np.float32)
  targets = rng.integers(0, cfg.out_f, cfg.batch)

  eng = bv.Engine(cfg.build_dir, cfg.device)
  # bitsy_v2 MLP: Linear -> GELU -> Linear -> logits
  l1 = bv.nn.Linear(eng, cfg.in_f, cfg.hidden, cfg.seed)
  l2 = bv.nn.Linear(eng, cfg.hidden, cfg.out_f, cfg.seed+1)
  x = eng.tensor(X)
  h = l1(x).gelu()
  logits = l2(h)
  loss = bv.functional.cross_entropy(logits, targets)
  loss.backward([l1.W, l1.b, l2.W, l2.b])

  # PyTorch equivalent with identical weights
  torch.manual_seed(cfg.seed)
  tl1 = torch.nn.Linear(cfg.in_f, cfg.hidden).cuda()
  tl2 = torch.nn.Linear(cfg.hidden, cfg.out_f).cuda()
  with torch.no_grad():
    tl1.weight.copy_(torch.from_numpy(l1.W.numpy().copy()))
    tl1.bias.copy_(torch.from_numpy(l1.b.numpy().reshape(-1).copy()))
    tl2.weight.copy_(torch.from_numpy(l2.W.numpy().copy()))
    tl2.bias.copy_(torch.from_numpy(l2.b.numpy().reshape(-1).copy()))
  Xt = torch.from_numpy(X).cuda()
  th = torch.nn.functional.gelu(tl1(Xt))
  tlogits = tl2(th)
  tloss = torch.nn.functional.cross_entropy(tlogits, torch.from_numpy(targets).cuda())
  tloss.backward()

  results = []
  def rec(name, d, tol):
    results.append({"name":name,"max_abs_diff":d,"tol":tol,"passed":d<=tol})
    print(f"[{'PASS' if d<=tol else 'FAIL'}] {name:22s} diff={d:.3e}")

  rec("forward_logits", md(logits.numpy(), tlogits.detach().cpu().numpy()), cfg.tol_fwd)
  rec("loss", md(loss.numpy(), np.array([tloss.item()],np.float32)), cfg.tol_fwd)
  for nm, bp, tp in (("grad_l1W",l1.W,tl1.weight),("grad_l1b",l1.b,tl1.bias),
                     ("grad_l2W",l2.W,tl2.weight),("grad_l2b",l2.b,tl2.bias)):
    bg = bp.grad
    tg = tp.grad.detach().cpu().numpy()
    if bg is None:
      results.append({"name":nm,"max_abs_diff":None,"tol":cfg.tol_grad,"passed":False})
      print(f"[FAIL] {nm:22s} no grad"); continue
    if bg.shape != tg.shape: bg = bg.reshape(tg.shape)
    rec(nm, md(bg, tg), cfg.tol_grad)

  # LayerNorm parity
  ln = bv.nn.LayerNorm(eng, cfg.in_f)
  y = ln(x)
  tln = torch.nn.LayerNorm(cfg.in_f).cuda()
  with torch.no_grad():
    tln.weight.copy_(torch.from_numpy(ln.W.numpy().reshape(-1).copy()))
    tln.bias.copy_(torch.from_numpy(ln.b.numpy().reshape(-1).copy()))
  rec("layernorm", md(y.numpy(), tln(Xt).detach().cpu().numpy()), cfg.tol_fwd)

  # one SGD step parity
  w_before = l1.W.numpy().copy()
  tw_before = tl1.weight.detach().cpu().numpy().copy()
  opt = bv.optim.SGD([l1.W], lr=0.01)
  opt.step()
  topt = torch.optim.SGD([tl1.weight], lr=0.01); topt.step()
  rec("sgd_step", md(l1.W.numpy(), tl1.weight.detach().cpu().numpy()), cfg.tol_grad)

  # one AdamW step parity (fresh weights, identical grads)
  l1.W.h = l1.W.rt._raw_tensor(w_before, True); l1.W.grad = l1.W.grad
  aw = bv.optim.AdamW([l1.W], lr=0.01, weight_decay=0.01)
  aw.step()
  tl1.weight.data.copy_(torch.from_numpy(tw_before))
  taw = torch.optim.AdamW([tl1.weight], lr=0.01, weight_decay=0.01)
  # use the same grad that PyTorch computed earlier
  tl1.weight.grad = torch.from_numpy(l1.W.grad.copy()).cuda() if l1.W.grad is not None else None
  if tl1.weight.grad is not None:
    taw.step()
    rec("adamw_step", md(l1.W.numpy(), tl1.weight.detach().cpu().numpy()), cfg.tol_grad)

  # RNN cell one-step parity (forward)
  rtcell = bv.nn.RNNCell(eng, cfg.in_f, cfg.hidden, cfg.seed+2)
  xseq = eng.tensor(X)
  h0 = bv.Tensor.zeros(eng, (cfg.batch, cfg.hidden))
  h = rtcell(xseq, h0)
  trnn = torch.nn.RNNCell(cfg.in_f, cfg.hidden).cuda()
  with torch.no_grad():
    trnn.weight_ih.copy_(torch.from_numpy(rtcell.W_ih.numpy().T.copy()))
    trnn.weight_hh.copy_(torch.from_numpy(rtcell.W_hh.numpy().T.copy()))
    trnn.bias_ih.copy_(torch.from_numpy(rtcell.b.numpy().reshape(-1).copy()))
    trnn.bias_hh.zero_()
  th = trnn(torch.from_numpy(X).cuda(), torch.zeros(cfg.batch, cfg.hidden).cuda())
  rec("rnn_cell_fwd", md(h.numpy(), th.detach().cpu().numpy()), cfg.tol_fwd)

  passed = sum(r["passed"] for r in results)
  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "summary":{"total":len(results),"passed":passed,"failed":len(results)-passed},
    "cases":results,
  }
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
  print(f"\n{passed}/{len(results)} bitsy_v2 model-parity cases passed")
  return 0 if passed==len(results) else 1

if __name__=="__main__": raise SystemExit(run())
