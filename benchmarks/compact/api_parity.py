"""bitsy_v2 v0.2 extra-API parity vs PyTorch CUDA (PR #9).

Tests the new pure-Python compositions over the frozen C primitive set:
activations (silu/mish/softplus/leaky_relu/elu/selu/hardtanh/hardswish),
losses (l1/smooth_l1/bce/bce_with_logits/nll), reductions (var/std/logsumexp/norm),
logical helpers and comparisons. All add ZERO new C lines.

Forward and (where differentiable) gradient parity. Dataclass config, no argparse.
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
from bitsy_v2 import functional as F, Tensor

@dataclass(frozen=True, slots=True)
class Config:
  build_dir:Path = ROOT / "build-rtvk"
  device:int = bv.Engine.VULKAN
  n:int = 128
  seed:int = 0
  tol_fwd:float = 1e-4
  tol_grad:float = 1e-3
  result_path:Path = ROOT / "benchmarks" / "results" / "bitsy_v2_api_parity.json"

CFG = Config()

def md(a,b): return float(np.abs(np.asarray(a,np.float32)-np.asarray(b,np.float32)).max())

def rec(results, name, diff, tol, print_it=True):
  ok = diff <= tol
  results.append({"name":name,"max_abs_diff":diff,"tol":tol,"passed":ok})
  if print_it: print(f"[{'PASS' if ok else 'FAIL'}] {name:28s} diff={diff:.3e}")
  return ok

def run(cfg=CFG):
  if not torch.cuda.is_available():
    print("CUDA unavailable; skipping"); return 0
  eng = bv.Engine(cfg.build_dir, cfg.device)
  rng = np.random.default_rng(cfg.seed)
  x = rng.standard_normal(cfg.n).astype(np.float32)
  ydata = rng.standard_normal(cfg.n).astype(np.float32)
  probs = (1/(1+np.exp(-rng.standard_normal(cfg.n).astype(np.float32)))).astype(np.float32)
  logits2d = rng.standard_normal((cfg.n//4, 4)).astype(np.float32)
  labels = rng.integers(0, 4, logits2d.shape[0])

  results = []
  rec_fwd = lambda nm, d: rec(results, nm, d, cfg.tol_fwd)
  rec_grad = lambda nm, d: rec(results, nm, d, cfg.tol_grad)

  # ---- unary activations (forward + grad) ----
  acts = [
    ("silu",      lambda t: t.silu(),                       lambda t: torch.nn.functional.silu(t)),
    ("softplus",  lambda t: t.softplus(),                   lambda t: torch.nn.functional.softplus(t)),
    ("mish",      lambda t: t.mish(),                       lambda t: torch.nn.functional.mish(t)),
    ("leaky_relu",lambda t: t.leaky_relu(0.01),             lambda t: torch.nn.functional.leaky_relu(t, 0.01)),
    ("elu",       lambda t: t.elu(1.0),                     lambda t: torch.nn.functional.elu(t, 1.0)),
    ("selu",      lambda t: t.selu(),                       lambda t: torch.nn.functional.selu(t)),
    ("hardtanh",  lambda t: t.hardtanh(-1.0, 1.0),          lambda t: torch.nn.functional.hardtanh(t, -1.0, 1.0)),
    ("hardswish", lambda t: t.hardswish(),                   lambda t: torch.nn.functional.hardswish(t)),
  ]
  for nm, bf, tf in acts:
    bx = bv.Tensor.from_numpy(eng.rt, x.copy(), requires_grad=True)
    tx = torch.tensor(x, device="cuda", requires_grad=True)
    bo, to = bf(bx), tf(tx)
    rec_fwd(nm+"_fwd", md(bo.numpy(), to.detach().cpu().numpy()))
    bo.backward(params=[bx]); to.backward(torch.ones_like(to))
    if bx.grad is not None:
      rec_grad(nm+"_grad", md(bx.grad, tx.grad.detach().cpu().numpy()))
    else:
      results.append({"name":nm+"_grad","max_abs_diff":None,"tol":cfg.tol_grad,"passed":False})
      print(f"[FAIL] {nm+'_grad':28s} no grad")

  # ---- reductions (forward) ----
  reds = [
    ("var",    lambda t: t.var(0, True),   lambda t: torch.var(t, dim=0, unbiased=True)),
    ("std",    lambda t: t.std(0, True),   lambda t: torch.std(t, dim=0, unbiased=True)),
    ("logsumexp", lambda t: t.logsumexp(0),lambda t: torch.logsumexp(t, dim=0)),
    ("norm2",  lambda t: t.norm(2.0, 0),   lambda t: torch.linalg.vector_norm(t, ord=2, dim=0)),
    ("norm1",  lambda t: t.norm(1.0, 0),   lambda t: torch.linalg.vector_norm(t, ord=1, dim=0)),
    ("square", lambda t: t.square(),       lambda t: torch.square(t)),
    ("reciprocal", lambda t: t.reciprocal(), lambda t: torch.reciprocal(t)),
    ("rsqrt",  lambda t: t.rsqrt(),        lambda t: torch.rsqrt(t)),
    ("exp2",   lambda t: t.exp2(),         lambda t: torch.exp2(t)),
    ("log2",   lambda t: t.log2(),         lambda t: torch.log2(t)),
    ("clamp",  lambda t: t.clamp(-0.5, 0.5), lambda t: torch.clamp(t, -0.5, 0.5)),
    ("relu6",      lambda t: t.relu6(),        lambda t: torch.nn.functional.relu6(t)),
    ("celu",       lambda t: t.celu(1.0),      lambda t: torch.nn.functional.celu(t, 1.0)),
    ("softsign",   lambda t: t.softsign(),     lambda t: torch.nn.functional.softsign(t)),
    ("tanhshrink", lambda t: t.tanhshrink(),   lambda t: torch.nn.functional.tanhshrink(t)),
    ("logsigmoid", lambda t: t.logsigmoid(),   lambda t: torch.nn.functional.logsigmoid(t)),
    ("softshrink", lambda t: t.softshrink(0.5),lambda t: torch.nn.functional.softshrink(t, 0.5)),
    ("threshold",  lambda t: t.threshold(0.1, 0.5), lambda t: torch.nn.functional.threshold(t, 0.5, 0.1)),
    ("sign",   lambda t: t.sign(),         lambda t: torch.sign(t)),
    ("clamp_min", lambda t: t.clamp_min(-0.2), lambda t: torch.clamp_min(t, -0.2)),
    ("clamp_max", lambda t: t.clamp_max(0.2),  lambda t: torch.clamp_max(t, 0.2)),
    ("sum_all", lambda t: t.sum(axis=None), lambda t: t.sum()),
    ("mean_all", lambda t: t.mean(axis=None), lambda t: t.mean()),
  ]
  # positive inputs: C sqrt clamps negatives at 0 (documented exp_037 deviation),
  # while PyTorch rsqrt/log2 return NaN — compare on the overlapping domain.
  xpos = np.abs(x) + 0.5
  for nm, bf, tf in reds:
    bx = bv.Tensor.from_numpy(eng.rt, xpos.copy())
    tx = torch.tensor(xpos, device="cuda")
    rec_fwd(nm, md(bf(bx).numpy(), tf(tx).detach().cpu().numpy()))

  # ---- view/shape ops (reshape-based compositions) ----
  views = [
    ("flatten",     lambda t: t.reshape(8,16).flatten(),      lambda t: t.reshape(8,16).flatten()),
    ("flatten_dim1", lambda t: t.reshape(8,4,4).flatten(1),   lambda t: t.reshape(8,4,4).flatten(1)),
    ("squeeze",     lambda t: t.reshape(8,1,16).squeeze(),    lambda t: t.reshape(8,1,16).squeeze()),
    ("squeeze_dim", lambda t: t.reshape(8,1,16).squeeze(1),  lambda t: t.reshape(8,1,16).squeeze(1)),
    ("unsqueeze",   lambda t: t.reshape(8,16).unsqueeze(1),   lambda t: t.reshape(8,16).unsqueeze(1)),
  ]
  for nm, bf, tf in views:
    bx = bv.Tensor.from_numpy(eng.rt, x.copy())
    tx = torch.tensor(x, device="cuda")
    rec_fwd(nm, md(bf(bx).numpy(), tf(tx).detach().cpu().numpy()))

  # ---- two-input losses (forward + grad on first input) ----
  losses = [
    ("l1_loss",      lambda a,b: F.l1_loss(a,b),            lambda a,b: torch.nn.functional.l1_loss(a,b)),
    ("smooth_l1_loss",lambda a,b: F.smooth_l1_loss(a,b,1.0),lambda a,b: torch.nn.functional.smooth_l1_loss(a,b,beta=1.0)),
    ("bce_loss",     lambda a,b: F.bce_loss(a,b),            lambda a,b: torch.nn.functional.binary_cross_entropy(a,b)),
    ("bce_with_logits",lambda a,b: F.bce_with_logits_loss(a,b),
                                                              lambda a,b: torch.nn.functional.binary_cross_entropy_with_logits(a,b)),
  ]
  for nm, bf, tf in losses:
    is_bce = nm.startswith("bce")
    xa = probs.copy() if nm == "bce_loss" else x.copy()
    if nm == "hinge_loss":
      tdata = np.where(ydata >= 0, 1.0, -1.0).astype(np.float32)
    elif nm in ("poisson_nll", "kl_div"):
      tdata = np.abs(ydata).astype(np.float32) + 0.5
    else:
      tdata = (ydata > 0).astype(np.float32) if is_bce else ydata
    bx = bv.Tensor.from_numpy(eng.rt, xa, requires_grad=True)
    by = bv.Tensor.from_numpy(eng.rt, tdata.copy())
    ta = torch.tensor(xa, device="cuda", requires_grad=True)
    ty = torch.tensor(tdata, device="cuda")
    bo, to = bf(bx,by), tf(ta,ty)
    rec_fwd(nm+"_fwd", md(bo.numpy(), np.array([to.item()],np.float32)))
    bo.backward(params=[bx]); to.backward()
    if bx.grad is not None:
      rec_grad(nm+"_grad", md(bx.grad, ta.grad.detach().cpu().numpy()))
    else:
      results.append({"name":nm+"_grad","max_abs_diff":None,"tol":cfg.tol_grad,"passed":False})
      print(f"[FAIL] {nm+'_grad':28s} no grad")

  # ---- nll_loss (host integer labels) ----
  bx = bv.Tensor.from_numpy(eng.rt, logits2d.copy(), requires_grad=True)
  bl = bx.log_softmax(axis=-1)
  bo = F.nll_loss(bl, labels)
  tl = torch.tensor(logits2d, device="cuda", requires_grad=True)
  to = torch.nn.functional.nll_loss(torch.log_softmax(tl, dim=-1),
        torch.tensor(labels, device="cuda"))
  rec_fwd("nll_loss_fwd", md(bo.numpy(), np.array([to.item()],np.float32)))
  bo.backward(params=[bx]); to.backward()
  if bx.grad is not None:
    rec_grad("nll_loss_grad", md(bx.grad, tl.grad.detach().cpu().numpy()))

  # ---- logical helpers + comparisons ----
  z1 = (x > 0).astype(np.float32)
  z2 = (ydata > 0).astype(np.float32)
  b1 = bv.Tensor.from_numpy(eng.rt, z1)
  b2 = bv.Tensor.from_numpy(eng.rt, z2)
  t1 = torch.tensor(z1, device="cuda")
  t2 = torch.tensor(z2, device="cuda")
  logical = [
    ("logical_not",  lambda a,b: F.logical_not(a),  lambda a,b: a.logical_not().float()),
    ("logical_and",  lambda a,b: F.logical_and(a,b),lambda a,b: torch.logical_and(a,b).float()),
    ("logical_or",   lambda a,b: F.logical_or(a,b), lambda a,b: torch.logical_or(a,b).float()),
    ("logical_xor",  lambda a,b: F.logical_xor(a,b),lambda a,b: torch.logical_xor(a,b).float()),
    ("cmp_gt",       lambda a,b: a.gt(b),            lambda a,b: (a>b).float()),
    ("cmp_le",       lambda a,b: a.le(b),            lambda a,b: (a<=b).float()),
    ("lerp",         lambda a,b: a.lerp(b, 0.25),    lambda a,b: torch.lerp(a,b,0.25)),
    ("all",          lambda a,b: a.gt(Tensor.scalar(a.rt,0.0)).all(),
                                                      lambda a,b: (a>0).all().float()),
    ("any",          lambda a,b: a.gt(Tensor.scalar(a.rt,0.0)).any(),
                                                      lambda a,b: (a>0).any().float()),
    ("isclose",      lambda a,b: F.isclose(a,b),      lambda a,b: torch.isclose(a,b).float()),
    ("allclose",     lambda a,b: F.allclose(a,b),     lambda a,b: torch.tensor([float(torch.allclose(a,b))], device=a.device)),
    ("dot",          lambda a,b: a.dot(b),            lambda a,b: torch.dot(a,b).reshape(1)),
    ("outer",        lambda a,b: a.outer(b),          lambda a,b: torch.outer(a,b)),
    ("softmax_bias", lambda a,b: a.softmax(axis=-1, bias=b),
                                                      lambda a,b: torch.softmax(a+b,dim=-1)),
    ("logsoftmax_bias", lambda a,b: a.log_softmax(axis=-1, bias=b),
                                                      lambda a,b: torch.log_softmax(a+b,dim=-1)),
    ("cosine_sim",   lambda a,b: F.cosine_similarity(a,b,dim=-1),
                                                      lambda a,b: torch.nn.functional.cosine_similarity(a,b,dim=-1)),
    ("pairwise_dist",lambda a,b: F.pairwise_distance(a,b),
                                                      lambda a,b: torch.nn.functional.pairwise_distance(a,b)),
    ("ln_residual",  lambda a,b: bv.nn.LayerNorm(eng, a.shape[-1])(a, residual=b),
                                                      lambda a,b: torch.nn.LayerNorm(a.shape[-1]).to(a.device)(a+b)),
    ("greater",      lambda a,b: F.greater(a,b),      lambda a,b: (a>b).float()),
    ("less",         lambda a,b: F.less(a,b),         lambda a,b: (a<b).float()),
    ("equal",        lambda a,b: F.equal(a,b),        lambda a,b: (a==b).float()),
    ("hypot",        lambda a,b: F.hypot(a,b),        lambda a,b: torch.hypot(a,b)),
    ("copysign",     lambda a,b: F.copysign(a,b),     lambda a,b: torch.copysign(a,b)),
    ("hinge_loss",   lambda a,b: F.hinge_loss(a,b),   lambda a,b: torch.clamp(1.0-a*b,min=0.0).mean()),
    ("poisson_nll",  lambda a,b: F.poisson_nll_loss(a,b,True), lambda a,b: torch.nn.functional.poisson_nll_loss(a,b,log_input=True,full=False,reduction='mean')),
    ("kl_div",       lambda a,b: F.kl_div(a,b.abs()+0.5), lambda a,b: torch.nn.functional.kl_div(a,b.abs()+0.5,reduction='mean')),
  ]
  for nm, bf, tf in logical:
    rec_fwd(nm, md(bf(b1,b2).numpy(), tf(t1,t2).detach().cpu().numpy()))

  # ---- float classification on special values ----
  special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -2.5] * 4, np.float32)
  bs = bv.Tensor.from_numpy(eng.rt, special.copy())
  ts = torch.tensor(special, device="cuda")
  special_cases = [
    ("isnan",      lambda t: F.isnan(t),          lambda t: torch.isnan(t).float()),
    ("isinf",      lambda t: F.isinf(t),          lambda t: torch.isinf(t).float()),
    ("isfinite",   lambda t: F.isfinite(t),       lambda t: torch.isfinite(t).float()),
    ("nan_to_num", lambda t: F.nan_to_num(t),     lambda t: torch.nan_to_num(t)),
  ]
  for nm, bf, tf in special_cases:
    rec_fwd(nm, md(bf(bs).numpy(), tf(ts).detach().cpu().numpy()))

  # ---- LayerNorm backward (residual + weight + bias), scalar mse loss ----
  # 1-D only: mse_loss's mean() over axis 0 is a true scalar only for 1-D inputs.
  for shp in ((4,),):
    xv=rng.standard_normal(shp).astype(np.float32)
    rv=rng.standard_normal(shp).astype(np.float32)
    e2=bv.Engine(device=bv.Engine.VULKAN)
    X2=bv.Tensor.from_numpy(e2.rt,xv,requires_grad=True)
    R2=bv.Tensor.from_numpy(e2.rt,rv,requires_grad=True)
    ln2=bv.nn.LayerNorm(e2,shp[-1])
    loss=F.mse_loss(ln2(X2,residual=R2), bv.Tensor.from_numpy(e2.rt,np.zeros(shp,np.float32)))
    loss.backward(params=[X2,R2,ln2.W,ln2.b])
    tx2=torch.tensor(xv,requires_grad=True); tr2=torch.tensor(rv,requires_grad=True)
    tl2=torch.nn.LayerNorm(shp[-1]);
    tloss=torch.nn.functional.mse_loss(tl2(tx2+tr2),torch.zeros(shp))
    tloss.backward()
    pairs=[("x",X2.grad,tx2.grad.detach().numpy()),("res",R2.grad,tr2.grad.detach().numpy()),
           ("w",ln2.W.grad,tl2.weight.grad.detach().numpy()),("b",ln2.b.grad,tl2.bias.grad.detach().numpy())]
    for nm,bg,tg in pairs:
      rec_fwd(f"ln_bwd_{shp}_{nm}", md(bg.reshape(tg.shape),tg))

  # ---- capture/replay correctness for multi-dispatch graphs (PR #33) ----
  for nm, build, ref in (
    ("capture_gemm",
      lambda e,x,w: (lambda X,W: (e.capture_begin(), X.matmul(W).realize(), e.capture_end(), X.matmul(W))[3])(e.tensor(x),e.tensor(w)),
      lambda x,w: x@w),
    ("capture_mlp",
      lambda e,x,w1,w2: (lambda X,W1,W2: (e.capture_begin(), X.matmul(W1.T()).relu().matmul(W2.T()).realize(), e.capture_end(), X.matmul(W1.T()).relu().matmul(W2.T()))[3])(e.tensor(x),e.tensor(w1),e.tensor(w2)),
      lambda x,w1,w2: np.maximum(x@w1.T,0)@w2.T),
  ):
    ce=bv.Engine(device=bv.Engine.VULKAN)
    cx=rng.standard_normal((8,16)).astype(np.float32)
    if nm=="capture_gemm":
      cw=rng.standard_normal((16,8)).astype(np.float32)
      yy=build(ce,cx,cw); ce.replay_n(3); want=ref(cx,cw)
    else:
      cw1=rng.standard_normal((12,16)).astype(np.float32)
      cw2=rng.standard_normal((4,12)).astype(np.float32)
      yy=build(ce,cx,cw1,cw2); ce.replay_n(3); want=ref(cx,cw1,cw2)
    rec_fwd(nm, md(yy.numpy(),want))

  # ---- PR #35: backward capture correctness (device transpose fix) ----
  bx=rng.standard_normal((8,16)).astype(np.float32)
  bw1=rng.standard_normal((12,16)).astype(np.float32)
  bw2=rng.standard_normal((4,12)).astype(np.float32)
  bt=rng.standard_normal((8,4)).astype(np.float32)
  be=bv.Engine(device=bv.Engine.VULKAN)
  BX=be.tensor(bx); BW1=be.tensor(bw1,requires_grad=True)
  BW2=be.tensor(bw2,requires_grad=True); BT=be.tensor(bt)
  by=bv.functional.mse_loss(BX.matmul(BW1.T()).relu().matmul(BW2.T()),BT)
  by.backward(params=[BW1,BW2]); br1,br2=BW1.grad.copy(),BW2.grad.copy()
  be2=bv.Engine(device=bv.Engine.VULKAN)
  BX2=be2.tensor(bx); BA=be2.tensor(bw1,requires_grad=True)
  BB=be2.tensor(bw2,requires_grad=True); BT2=be2.tensor(bt)
  be2.capture_begin()
  by2=bv.functional.mse_loss(BX2.matmul(BA.T()).relu().matmul(BB.T()),BT2)
  by2.backward(params=[BA,BB]); be2.capture_end()
  be2.replay_n(3)
  rec_fwd("capture_mlp_backward_w1", md(be2.rt.grad_refresh(BA.h,(12,16)),br1))
  rec_fwd("capture_mlp_backward_w2", md(be2.rt.grad_refresh(BB.h,(4,12)),br2))

  # ---- PR #36: device-side SGD + whole-update-loop capture ----
  sw=rng.standard_normal(128).astype(np.float32)
  sg=rng.standard_normal(128).astype(np.float32)
  se=bv.Engine(device=bv.Engine.VULKAN)
  SW=se.tensor(sw); SG=se.tensor(sg)
  se.sgd_update(SW.h,SG.h,0.1,0.0)
  rec_fwd("sgd_device_update", md(SW.numpy(),sw-0.1*sg))
  se2=bv.Engine(device=bv.Engine.VULKAN)
  TW=se2.tensor(sw); TG=se2.tensor(sg)
  se2.sgd_update(TW.h,TG.h,0.1,0.01)
  rec_fwd("sgd_device_weight_decay", md(TW.numpy(),sw-0.1*(sg+0.01*sw)))

  # full forward+backward+SGD captured, N steps in one submit == torch SGD steps
  sx=rng.standard_normal((8,16)).astype(np.float32)
  sw1=rng.standard_normal((12,16)).astype(np.float32)
  sw2=rng.standard_normal((1,12)).astype(np.float32)
  st=rng.standard_normal((8,1)).astype(np.float32)
  lr=0.01
  ue=bv.Engine(device=bv.Engine.VULKAN)
  UX=ue.tensor(sx); UA=ue.tensor(sw1,requires_grad=True)
  UB=ue.tensor(sw2,requires_grad=True); UT=ue.tensor(st)
  def ufwd():
    return bv.functional.mse_loss(UX.matmul(UA.T()).relu().matmul(UB.T()),UT)
  ue.capture_begin()
  uy=ufwd(); uy.backward(params=[UA,UB])
  ue.sgd_update(UA.h,ue.grad_tensor(UA),lr,0.0)
  ue.sgd_update(UB.h,ue.grad_tensor(UB),lr,0.0)
  ue.capture_end()
  ue.upload(UA,sw1); ue.upload(UB,sw2); ue.replay_n(4)
  ga=ue.get(UA.h,(12,16)); gb=ue.get(UB.h,(1,12))
  tsx=torch.from_numpy(sx); tw1=torch.from_numpy(sw1).requires_grad_(True)
  tw2=torch.from_numpy(sw2).requires_grad_(True); tst=torch.from_numpy(st)
  uopt=torch.optim.SGD([tw1,tw2],lr=lr)
  for _ in range(4):
    uopt.zero_grad()
    ul=torch.nn.functional.mse_loss(torch.relu(tsx@tw1.T)@tw2.T,tst)
    ul.backward(); uopt.step()
  rec_grad("capture_full_update_w1", md(ga,tw1.detach().numpy()))
  rec_grad("capture_full_update_w2", md(gb,tw2.detach().numpy()))

  # ---- PR #38: multi-graph streaming with FRESH per-step mapped inputs ----
  K=2; slr=1e-4
  sw1=rng.standard_normal((16,32)).astype(np.float32)
  sw2=rng.standard_normal((1,16)).astype(np.float32)
  st=rng.standard_normal((8,1)).astype(np.float32)
  sdata=[rng.standard_normal((8,32)).astype(np.float32) for _ in range(K)]
  se=bv.Engine(device=bv.Engine.VULKAN)
  sA=se.tensor(sw1,requires_grad=True); sB=se.tensor(sw2,requires_grad=True)
  sT=se.tensor(st); sxs=[]; sptrs=[]
  for i in range(K):
    sx=se.tensor(sdata[i]); sxs.append(sx); sptrs.append(se.map_input(sx))
    se.stream_begin(i)
    sy=bv.functional.mse_loss(sx.matmul(sA.T()).relu().matmul(sB.T()),sT)
    sy.backward(params=[sA,sB])
    se.sgd_update(sA.h,se.grad_tensor(sA),slr,0.0)
    se.sgd_update(sB.h,se.grad_tensor(sB),slr,0.0)
    se.stream_end(i); se.zero_grad(sA); se.zero_grad(sB)
  se.stream_submit(K)
  sga=se.get(sA.h,(16,32)); sgb=se.get(sB.h,(1,16))
  tsx=[torch.from_numpy(d) for d in sdata]
  stw1=torch.from_numpy(sw1).requires_grad_(True); stw2=torch.from_numpy(sw2).requires_grad_(True)
  stt=torch.from_numpy(st); sopt=torch.optim.SGD([stw1,stw2],lr=slr)
  for i in range(K):
    sopt.zero_grad()
    sl=torch.nn.functional.mse_loss(torch.relu(tsx[i]@stw1.T)@stw2.T,stt)
    sl.backward(); sopt.step()
  rec_grad("stream_fresh_w1", md(sga,stw1.detach().numpy()))
  rec_grad("stream_fresh_w2", md(sgb,stw2.detach().numpy()))

  # ---- PR #39: eager ring wrap (>RT_RING dispatches before readback) ----
  rx=rng.standard_normal(2048).astype(np.float32)
  re=bv.Engine(device=bv.Engine.VULKAN)
  RX=re.tensor(rx)
  rnode=RX
  for _ in range(12): rnode=rnode*1.0001+0.0001   # 24 fused dispatches > ring size 8
  rgot=rnode.numpy()
  rwant=rx.copy()
  for _ in range(12): rwant=rwant*np.float32(1.0001)+np.float32(0.0001)
  rec_fwd("eager_ring_wrap", md(rgot,rwant))

  # ---- PR #40: compiled callable API (capture as default path) ----
  cw1=rng.standard_normal((12,16)).astype(np.float32)
  cw2=rng.standard_normal((1,12)).astype(np.float32)
  cdata=[rng.standard_normal((8,16)).astype(np.float32) for _ in range(2)]
  ce=bv.Engine(device=bv.Engine.VULKAN)
  cA=ce.tensor(cw1,requires_grad=True); cB=ce.tensor(cw2,requires_grad=True)
  def cfn(X):
    y=bv.functional.mse_loss(X.matmul(cA.T()).relu().matmul(cB.T()),cT)
    y.backward(params=[cA,cB])
    ce.sgd_update(cA.h,ce.grad_tensor(cA),slr,0.0)
    ce.sgd_update(cB.h,ce.grad_tensor(cB),slr,0.0)
    return y
  cT=ce.tensor(np.zeros((8,1),np.float32))
  ccf=ce.compile_many(cfn,[(8,16)],2,zero_grads=[cA,cB])
  ce.upload(cA,cw1); ce.upload(cB,cw2)
  ccf([[cdata[0]],[cdata[1]]])
  cga=ce.get(cA.h,(12,16)); cgb=ce.get(cB.h,(1,12))
  ctx=[torch.from_numpy(d) for d in cdata]
  ct1=torch.from_numpy(cw1).requires_grad_(True); ct2=torch.from_numpy(cw2).requires_grad_(True)
  copt=torch.optim.SGD([ct1,ct2],lr=slr)
  for i in range(2):
    copt.zero_grad()
    cl=torch.nn.functional.mse_loss(torch.relu(ctx[i]@ct1.T)@ct2.T,torch.zeros((8,1)))
    cl.backward(); copt.step()
  rec_grad("compiled_api_w1", md(cga,ct1.detach().numpy()))
  rec_grad("compiled_api_w2", md(cgb,ct2.detach().numpy()))

  # ---- PR #41: transpose fold into GEMM (one dispatch, no separate transpose) ----
  fx=rng.standard_normal((8,16)).astype(np.float32)
  fw=rng.standard_normal((4,16)).astype(np.float32)
  fe=bv.Engine(device=bv.Engine.VULKAN)
  FX=fe.tensor(fx); FW=fe.tensor(fw,requires_grad=True)
  fe.rt.lib.rt_reset_dispatch_count(fe.rt.ctx)
  fy=FX.matmul(FW.T()); fy.realize()
  nd=int(fe.rt.lib.rt_dispatch_count(fe.rt.ctx))
  rec(results, "folded_gemm_dispatch", abs(float(nd)-1.0), 0.0)   # exactly 1 dispatch
  fl=fy.sum(axis=0).sum(axis=0); fl.backward(params=[FW])
  rec_fwd("folded_gemm_fwd", md(fy.numpy(),fx@fw.T))
  tfw=torch.from_numpy(fw).requires_grad_(True)
  tfl=(torch.from_numpy(fx)@tfw.T).sum(); tfl.backward()
  rec_grad("folded_gemm_grad", md(FW.grad,tfw.grad.numpy()))

  # ---- PR #42: folded GEMM+bias epilogue (nn.Linear in one dispatch) ----
  gx=rng.standard_normal((8,16)).astype(np.float32)
  gw=rng.standard_normal((4,16)).astype(np.float32)
  gb=rng.standard_normal((1,4)).astype(np.float32)
  ge=bv.Engine(device=bv.Engine.VULKAN)
  GX=ge.tensor(gx); GW=ge.tensor(gw,requires_grad=True); GB=ge.tensor(gb,requires_grad=True)
  ge.rt.lib.rt_reset_dispatch_count(ge.rt.ctx)
  gy=GX.matmul(GW.T())+GB; gy.numpy()
  nd2=int(ge.rt.lib.rt_dispatch_count(ge.rt.ctx))
  rec(results, "folded_gemm_bias_dispatch", abs(float(nd2)-1.0), 0.0)
  gfl=gy.sum(axis=0).sum(axis=0); gfl.backward(params=[GW,GB])
  tgw=torch.from_numpy(gw).requires_grad_(True); tgb=torch.from_numpy(gb).requires_grad_(True)
  tgy=(torch.from_numpy(gx)@tgw.T+tgb).sum(); tgy.backward()
  rec_fwd("folded_gemm_bias_fwd", md(gy.numpy(),(torch.from_numpy(gx)@torch.from_numpy(gw).T+torch.from_numpy(gb)).numpy()))
  rec_grad("folded_gemm_bias_wgrad", md(GW.grad,tgw.grad.numpy()))
  rec_grad("folded_gemm_bias_bgrad", md(GB.grad,tgb.grad.numpy()))

  # ---- PR #47: auto-capture (captured path as default for recurring shapes) ----
  aw1=rng.standard_normal((8,16)).astype(np.float32)
  aw2=rng.standard_normal((1,8)).astype(np.float32)
  adata=[rng.standard_normal((4,16)).astype(np.float32) for _ in range(2)]
  ae=bv.Engine(device=bv.Engine.VULKAN)
  aA=ae.tensor(aw1,requires_grad=True); aB=ae.tensor(aw2,requires_grad=True)
  aT=ae.tensor(np.zeros((4,1),np.float32))
  def afn(X):
    y=bv.functional.mse_loss(X.matmul(aA.T()).relu().matmul(aB.T()),aT)
    y.backward(params=[aA,aB])
    ae.sgd_update(aA.h,ae.grad_tensor(aA),slr,0.0)
    ae.sgd_update(aB.h,ae.grad_tensor(aB),slr,0.0)
    return y
  aauto=ae.auto(2)
  aout=aauto.run("case",afn,[(4,16)],[[d] for d in adata],zero_grads=[aA,aB])
  ae.upload(aA,aw1); ae.upload(aB,aw2)
  aauto.run("case",afn,[(4,16)],[[d] for d in adata],zero_grads=[aA,aB])
  aga=ae.get(aA.h,(8,16)); agb=ae.get(aB.h,(1,8))
  atw1=torch.from_numpy(aw1).requires_grad_(True); atw2=torch.from_numpy(aw2).requires_grad_(True)
  aopt=torch.optim.SGD([atw1,atw2],lr=slr)
  for i in range(2):
    aopt.zero_grad()
    al=torch.nn.functional.mse_loss(torch.relu(torch.from_numpy(adata[i])@atw1.T)@atw2.T,torch.zeros((4,1)))
    al.backward(); aopt.step()
  rec_grad("auto_capture_w1", md(aga,atw1.detach().numpy()))
  rec_grad("auto_capture_w2", md(agb,atw2.detach().numpy()))

  passed = sum(r["passed"] for r in results)
  report = {
    "generated_at":datetime.now(timezone.utc).isoformat(),
    "environment":{"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)},
    "config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
    "summary":{"total":len(results),"passed":passed,"failed":len(results)-passed},
    "cases":results,
  }
  cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
  cfg.result_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
  print(f"\n{passed}/{len(results)} bitsy_v2 extra-API parity cases passed")
  return 0 if passed==len(results) else 1

if __name__=="__main__": raise SystemExit(run())
