# exp_007 — Findings: profiler and gap ledger

**Date:** 2026-11-21
**Scope:** Phase 2 instrumentation only. **No framework kernel or algorithm
optimization** was attempted. Numeric parity re-verified after instrumentation:
42/42 parity, gradcheck 1.19e-07 / 5.96e-08, conv gradients ≤7.6e-06.

## What was built

- `hg.profiler` (opt-in, off by default): C API `bt_profiler_enable`,
  `reset`, `dump_json`, plus counters for dispatches, barriers, submits,
  allocations and cache hits. Python surface: `bt.profile()`, `profile_reset()`,
  `profile_dump(path)`.
- Per-dispatch Vulkan timestamp queries (separate query pool), CPU enqueue time
  for descriptor setup + recording, estimated bytes/FLOPs per op, and a Chrome
  trace JSON with one X event per dispatch plus a summary event giving achieved
  GB/s and GFLOP/s as percentages of the Phase 1 ceilings (304.4 GB/s,
  5.49 TFLOP/s).
- `benchmarks/ops/three_way.py` (dataclass config) and
  `benchmarks/ops/torch_kernel_profile.py`.
- `GAP_LEDGER.md` ranked by microseconds lost.

## Bug found and fixed during the experiment

The first profiled run deadlocked. Cause: `vkGetQueryPoolResults` was called
with `VK_QUERY_RESULT_WAIT_BIT` on timestamps that had been reset in a previous
flush but not rewritten in the current batch. Fixed by resetting
`profiler_query_count` per batch and mapping each query slot to an event index
(`bt_profiler_record` now returns the event slot), so only just-written queries
are read. This is recorded here rather than silently dropped.

## Three-way results

| Op | torch wall | Bitsy wall | Bitsy GPU/dispatch | gap | ratio |
|---|---:|---:|---:|---:|---:|
| add 1M | 0.0880 ms | 0.1009 ms | 48.4 µs | 12.9 µs | 0.87 |
| mul 1M | 0.0869 ms | 0.1396 ms | 52.2 µs | 52.8 µs | 0.62 |
| relu 1M | 0.0676 ms | 0.1047 ms | 48.9 µs | 37.0 µs | 0.65 |
| matmul 256² | 0.0631 ms | 0.2186 ms | 165.8 µs | 155.5 µs | 0.29 |

Profiler overhead over a 50-iteration median: +2.8 µs (add), −21 µs (mul,
noise), +4.2 µs (relu), +15.2 µs (matmul) — small compared with the gaps.

## PyTorch kernel parameters observed

- add/mul/relu: `at::native::vectorized_elementwise_kernel<4, ...>` (vec4),
  42–46 µs/call for 1M elements; relu 33 µs via `clamp_min`.
- matmul: `volta_sgemm_128x32_nn` (18 µs/call) + `splitKreduce_kernel<32,16>`
  (5.1 µs/call). Bitsy GEMM runs at 3.7% of FP32 peak.

## Predictions

- "At least one op where submit/wait is a larger share than GPU dispatch":
  **partially supported.** Per-iteration host overhead is ~52 µs versus 48–166 µs
  GPU time; it is a large share for elementwise ops but not the largest for GEMM.
- "Three-way timing isolates gaps into categories with microsecond amounts":
  **supported** — `GAP_LEDGER.md` ranks 155.5 / 52.8 / 37.0 / 12.9 µs.
- "PyTorch tile/register choices explain the GEMM gap": **supported** —
  cuBLAS uses a 128×32 tile and split-K; Bitsy uses naive/16×16 tiled SGEMM.

## Known method gap (negative finding)

The campaign asks for isolated kernel timing with 1000+ repeats in one command
buffer. Phase 2 used per-dispatch timestamps as a **kernel proxy**, not that
measurement; the standalone `perf/` probe only provides isolated numbers for
copy/FMA/memory. This is recorded in `GAP_LEDGER.md` rather than disguised.

## Next (Phase 3/4/6, not started)

Fix the top-ranked GEMM gap first with the register-blocked GEMM ladder and
split-K, each with its own `exp_NNN` and before/after measurements; then
vectorize elementwise kernels to vec4 and remove unconditional barriers.
