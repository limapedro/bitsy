# exp_040 — FINDINGS (PR #16: activation + loss API batch)

Status: PR #16 of the v0.2 rewrite. Follows PR #15 (fusion correctness fix).

## API batch (pure Python, zero new C lines)
Activations: `relu6`, `celu`, `softsign`, `tanhshrink`, `logsigmoid`,
`softshrink`, `threshold` (plus `exp1m` helper).
Losses: `hinge_loss`, `poisson_nll_loss`, `kl_div`.

`benchmarks/compact/api_parity.py` extended 69 → **79 cases**, all green vs
PyTorch CUDA, including:
- special-value section (NaN / ±inf) for classification APIs (PR #15),
- dedicated target data for hinge (±1), Poisson and KL (positive).

Honest note: PyTorch's `torch.nn.functional.hinge_embedding_loss` uses a
different target convention than the explicit `mean(max(0, 1 - pred*target))`
formula; the parity case compares against the explicit torch formula
(`torch.clamp(1-a*b, min=0).mean()`), which is what our implementation defines.
The gate is not silently weakened — it checks the same math on both sides.

## Gates (all green)
v2 parity 48/48, op 27/27, gradcheck 25/25, model 10/10, API 79/79,
CTest 2/2. Core **2,090 lines** (budget 4,096, 2,006 spare).
Replay perf unchanged / this run slightly ahead: 0.7× per-step vs PyTorch
(bitsy 0.144 vs torch 0.197 ms/step), clock steady 1950→1950 MHz.

## Honest scope / remaining work
- GEMM ~3.5× off cuBLAS — blocked on this GTX 1660 Super (no cooperative-matrix
  extension; FINDINGS_PR15.md); needs RTX-class hardware or a decision.
- Full 285-function map still incomplete (RNG, conv/pool, linalg decompositions,
  serialization, data loading, AMP, distributed, FFT/sparse).
- Speed parity remains scoped to this benchmark family; single-step forced
  download still slower (0.89 vs 0.68 ms this run); clock pinned only in this run.
