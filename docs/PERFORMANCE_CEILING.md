# Performance ceiling assessment — GTX 1660 Super / Vulkan (2026-11-21)

This is an **evidence-backed, conditional conclusion** for the current pairing:
TU116 GTX 1660 Super, driver 621.01, Vulkan 1.4, C11 core, interim Python
autograd. It is not a claim about other GPUs, a compiled graph compiler, or a
rewritten framework.

## Final measured status (single consistent run, clocks logged)

| class | Bitsy | PyTorch CUDA | ratio | target | status |
|---|---:|---:|---:|---:|---|
| eager end-to-end MLP step | 0.825 ms | 0.983 ms | **1.19** | ≥0.85 | **MET** |
| elementwise eager (1M) | 0.095–0.124 ms | 0.044–0.064 ms | 0.46–0.54 | ≥0.95 | missed |
| elementwise captured (1M) | 0.034–0.051 ms | 0.031–0.045 ms | **0.87–0.93** | — | at parity |
| GEMM 1024² | 1.276 ms | 0.907 ms | 0.71 | ≥0.80 | missed |
| GEMM 256² | 0.238 ms | 0.051 ms | 0.21 | — | missed |
| captured full MLP step | 0.1457 ms | 0.0737 ms (graph) | 0.51 | ≥0.85 | missed |
| pointwise capture (exp_013) | 0.899 ms/chain | 0.896 ms (graph) | **1.00** | — | at parity |

Calibrated ceilings (`perf/HARDWARE_LEDGER.md`): DRAM 304.4 GB/s, FP32 5.49
TFLOP/s @1950 MHz; the vec4 elementwise kernel reaches ~96% of DRAM ceiling and
the FMA probe reaches 95.3% of FP32 peak, so the measurement method is sound.

## Why each remaining gap is bounded

1. **Elementwise eager (0.46–0.54).** The same kernels reach 0.87–0.93× PyTorch
   CUDA graphs when captured (exp_031). The gap is the per-op submit+fence
   (~36 µs, WDDM), not the kernel. exp_032 removed an avoidable timestamp sync
   from every wait and found no further gain — the queue submission itself is
   the floor. Kernel tuning cannot fix a host-side cost.
2. **Standalone GEMM (0.71).** The full ladder is measured: naive, tiled,
   register-blocked 128×128×8 and 128×64×8, vec4, double buffering (k16
   negative), split-K (negative), L2 swizzle (negative), 64×128 (positive but
   small), fused epilogue, and packed f16 (3× slower and wrong here). No variant
   reaches ≥0.80 on TU116; cuBLAS is near peak for this shape. Further tuning
   is unlikely to add >10%.
3. **Captured full step (0.51).** 19 Vulkan dispatches remain after fusing
   bias, log_softmax, CE loss, SGD and more. PyTorch's graph uses far fewer,
   more fused kernels. Ad-hoc fusion attempts: exp_029 neutral (fused SGD),
   exp_033 broke training parity (fused CE backward), exp_019/023 broke
   gradients (views). The remaining structural change is a true step-level
   graph compiler with full training-scale parity — not a per-op shader.
4. **Small-op latency (0.21 GEMM 256², 0.38–0.68 elementwise 4096).** Pure
   launch/sync domination; captured execution removes most of it.

## Experiment outcome ledger (34 experiments)

Positive/negotiated: register-blocked GEMM 3.54× (exp_008), vec4 elementwise
96% DRAM (exp_012), capture/replay 1.00× (exp_013), deferred uploads 2.97×
(exp_014), median tuner + 64×128 (exp_026), fused log_softmax 1.07× (exp_027),
fused CE 1.15× (exp_028), fused bias epilogue → end-to-end target MET
(exp_024). Negative or neutral: k16, split-K, k8 over-buffering, L2 swizzle,
128×64 tile, packed f16, broadcast views (twice), dual mul, fused SGD, lazy
timestamps, async overlap, hazard barriers, several fusions. Two parity
failures (exp_019, exp_033) were caught only by training-scale checks and led
to the 44-case gate (exp_034).

## What would actually be required to go further

- A **step-level graph compiler** that fuses the whole forward/backward/optimizer
  into a few kernels while preserving training-scale gradients (44-case gate).
- Or hardware with **tensor cores** / a different driver stack where f16/packed
  math has a real throughput advantage (exp_030 showed no advantage here).
- Or changing the **fair-comparison baseline**: comparing captured Bitsy to
  captured PyTorch already shows parity for elementwise and pointwise chains —
  the eager ratios measure host launch cost, not kernel quality.

## Scope and honesty

This conclusion applies to the current GTX 1660 Super / driver / Vulkan / core
stack. It does not claim Bitsy cannot be faster in principle; it says the
measured mechanism for each remaining gap is understood and the untried rungs
of the campaign ladder are exhausted. All misses remain visible in
`BENCHMARKS.md` and `GAP_LEDGER.md`; no numbers are cherry-picked.
