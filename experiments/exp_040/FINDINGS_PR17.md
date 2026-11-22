# exp_040 — FINDINGS (PR #17: common-operations speed verification + parallel reduce)

Status: PR #17. Requested task: verify current speeds vs PyTorch across common ops.

## Speed verification (new `benchmarks/compact/common_ops_bench.py`)
GTX 1660 SUPER, driver 621.1.0.0; 1M-element fp32; end-to-end (op + host read);
warmup 5, 20 iterations, median+spread; clock 1530→1950 MHz.

### BEFORE this PR (single eager op + read, bitsy vs torch)
```
add 3.10ms vs 0.72ms (4.3x)   mul 2.86/0.70 (4.1x)    relu 2.67/0.69 (3.9x)
sigmoid 3.05/0.65 (4.7x)      gelu 3.46/0.77 (4.5x)   silu 3.52/0.67 (5.2x)
sum 42.56ms vs 0.05ms (802x)  mean 42.53/0.04 (983x)  max 42.54/0.06 (768x)
fused chain 3.36/0.74 (4.5x)
10-step async loop: bitsy 0.144 ms/step vs torch 0.173 -> 0.83x (bitsy FASTER)
```
Two findings:
1. The published "parity" (0.9×) is **only** the multi-step async loop metric.
   A single eager operation is 4–5× slower — that is the honest headline for
   per-call latency.
2. Reductions were catastrophically slow (~800×), a previously unmeasured hole.

## Root cause + fix for reductions
`reduce.comp` launched `(outer*inner+255)/256` invocations but used one
invocation per output line. For a 1-D 1M reduce, `outer=1, inner=1` → **one GPU
thread looped over 1M elements** sequentially in global memory.

Rewrote the shader: one WORKGROUP (256 cooperating threads) per output line;
each thread accumulates a strided slice, then a shared-memory tree reduction
combines them. Dispatch now launches `total` workgroups (not ceil(total/256)
invocations).

### AFTER (same benchmark, median; all gates green)
```
sum  1.93ms vs 0.054ms (35.7x)   mean 1.89/0.068 (27.7x)   max 1.70/0.071 (23.8x)
```
**~800× → ~25×** for reductions (42.5ms → 1.9ms). Still far off torch; the
remaining ~1.9ms is the eager-path submit + staging download (profiling showed
kernel itself is a small fraction).

## Honest summary of current speed vs PyTorch CUDA
| workload | bitsy | torch | gap |
|---|---|---|---|
| 10-step async fused loop | 0.143 ms/step | 0.173 ms/step | **0.83× (bitsy faster)** |
| single fused op + read | 3.36 ms | 0.74 ms | 4.5× slower |
| elementwise (add/mul/relu/...) | 2.7–3.5 ms | 0.65–0.80 ms | 3.9–5.2× slower |
| reductions (sum/mean/max) | 1.7–1.9 ms | 0.05–0.07 ms | 24–36× slower |
| GEMM n=1024 (PR #15) | 6.32 ms | 1.80 ms | ~3.5× slower |

Single-call latency is dominated by the eager path's submit + staging download
(`build+realize` 2.37ms vs `build+numpy` 2.95ms; graph build itself 0.003ms).
Capture/replay with async submit remains the fast path and is faster than torch
on the loop metric.

## Gates (all green)
v2 parity 48/48, op 27/27, gradcheck 25/25, model 10/10, API 79/79,
CTest 2/2. Core 2,090 lines ≤ 4,096.

## Honest scope
- "Speed parity" is benchmark-scoped: only the async loop reaches/beats torch;
  single-op latency is 4–5× slower and reductions 24–36× slower. No universal
  parity claim.
- GEMM ~3.5× off cuBLAS, blocked on this GPU (no cooperative-matrix extension).
- Clock not pinned across runs (1530→1950 here); spread recorded in JSON.
- Numeric diffs: div 3.1e-2 absolute over 1M elements (GLSL 1-ULP division);
  log/sqrt show NaN diffs because both sides produce NaN on negative inputs —
  harness artifact, not a parity failure (gates use valid domains).
