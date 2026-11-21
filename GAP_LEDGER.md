# GAP_LEDGER — ranked performance gaps

**Phase:** TOGISHI Phase 2 (profiling only; no framework optimization yet)
**Evidence:** `benchmarks/results/three_way.json`, `benchmarks/results/torch_kernels.json`,
per-op Chrome traces `benchmarks/results/trace_*.json`, `experiments/exp_007/`.

Method: 50 iterations, warm-up 5, median reported. PyTorch CUDA is given
`cudnn.benchmark = True`. Bitsy wall time includes `bt.flush() + bt.wait()`
per iteration; PyTorch wall time includes `torch.cuda.synchronize()`. Bitsy
GPU time is the sum of per-dispatch timestamp queries — a **kernel proxy**, not
a 1000-repeat isolated measurement (that distinction is preserved in the data).

Calibrated ceilings from Phase 1: DRAM **304.4 GB/s**, FP32 **5.49 TFLOP/s**
@1950 MHz, submit+fence **~36 µs**, allocate+free **~363 µs**, empty dispatch
**~3.1 µs**.

## Ranked ledger

| Rank | Op | Wall gap (µs) | Bitsy GPU/dispatch (µs) | PyTorch kernel (µs/call) | Classification | Evidence |
|---:|---|---:|---:|---:|---|---|
| 1 | matmul 256² | **155.5** | 165.8 | 18.0 (`volta_sgemm_128x32_nn`) + 5.1 splitK | compute/method + wave quantization | `trace_matmul_prof.json`, 3.7% of FP32 peak |
| 2 | mul 1M | **52.8** | 52.2 | 46.1 (`vectorized_elementwise_kernel<4>`) | bandwidth + launch/sync | 79.1% of DRAM ceiling; host overhead 52.8 µs |
| 3 | relu 1M | **37.0** | 48.9 | 33.3 (`vectorized_elementwise_kernel<4>`) | launch/sync + bandwidth | 84.6% of DRAM ceiling |
| 4 | add 1M | **12.9** | 48.4 | 42.3 (`vectorized_elementwise_kernel<4>`) | launch/sync | 85.5% of DRAM ceiling |

## Per-op profiler counters (50 iterations)

| Op | dispatches | barriers | submits | allocations | cache hits | profiler overhead (µs) |
|---|---:|---:|---:|---:|---:|---:|
| add | 50 | 50 | 50 | 0 | 50 | +2.8 |
| mul | 50 | 50 | 50 | 0 | 50 | −21.0 (noise) |
| relu | 50 | 50 | 50 | 0 | 50 | +4.2 |
| matmul | 50 | 50 | 50 | 0 | 50 | +15.2 |

The caching allocator is working: **0** `vkAllocateMemory` calls in 50 steps,
all 50 allocations served from the size-class cache. One submit + one barrier
per iteration remains because the benchmark explicitly waits each iteration.

## PyTorch kernel parameters to copy

| Op | PyTorch kernel | Notes |
|---|---|---|
| add / mul | `at::native::vectorized_elementwise_kernel<4, ...>` | vec4 loads/stores; Bitsy uses scalar float per lane |
| relu | `vectorized_elementwise_kernel<4>` via `clamp_min` | vec4 |
| matmul | `volta_sgemm_128x32_nn` + `splitKreduce_kernel<32,16>` | 128×32 tile, split-K reduction; Bitsy uses naive/16×16 tiled SGEMM at 3.7% of peak |

## Interpretation / next fixes (ranked)

1. **GEMM (155.5 µs):** implement the GEMM ladder from the campaign —
   register-blocked 128×128×8 / 128×64×8 tiles, vec4 + conflict-free transposed
   A in shared memory, double buffering, split-K, L2-friendly block order,
   autotune cached per device. This is the single largest gap.
2. **Elementwise vectorization (~50 µs combined):** use `vec4` loads/stores and
   remove the unconditional global barrier between independent dispatches;
   target ≥90% of measured DRAM BW (currently 79–86%).
3. **Submission overhead (~14–53 µs):** the benchmark forces a wait per
   iteration; the profiler shows one submit and one barrier per dispatch.
   Overlap host work across iterations (timeline semaphore already present) and
   use synchronization2 with hazard-tracked minimal barriers.
4. **Allocator:** already solved for the steady state (0 real allocations in
   50 steps); keep the cache, add stream-ordered reuse and memory-budget
   awareness in Phase 4.

## Honesty

- Timestamp GPU time is a proxy, not the 1000-repeat isolated kernel time the
  campaign asks for; the standalone probe in `perf/` provides isolated numbers
  only for copy/FMA/memory measurements, not for these framework ops. This is a
  known gap in the Phase 2 method and is recorded rather than hidden.
- Profiling overhead is small (≤15 µs over a 50-iteration median) and one
  measurement (mul) is within noise.
- No framework kernel or algorithm was changed in this phase.
