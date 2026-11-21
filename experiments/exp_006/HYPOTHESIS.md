# exp_006 — TOGISHI Phase 1: hardware calibration

**Date:** 2026-11-21
**Owner:** Bitsy performance campaign TOGISHI

## Hypothesis

The framework's 0.22–0.53 throughput ratios are currently uninterpretable
because we do not know this GPU's *measured* ceilings. Vendor theory says
~336 GB/s DRAM and ~5 TFLOP/s FP32, but clocks, WDDM, driver overhead and our
own measurement method can move those numbers substantially. We must calibrate
bare Vulkan kernels before attributing any gap to Hagane's code.

## Prediction

- A large vec4 copy reaches ≥ 90% of the measured best copy bandwidth, and
  measured DRAM bandwidth is within ~15% of the ~336 GB/s theoretical figure.
- A dense FMA loop reaches ≥ 95% of the clock-normalized FP32 FMA peak
  (1408 cores × 2 × boost clock).
- Empty-dispatch, barrier, submit and `vkAllocateMemory` costs are each
  quantified in microseconds, showing how much of the 0.06–0.07 ms per-op floor
  is unavoidable.
- Shared-memory bank conflicts (stride 1 vs 32) show a measurable ratio, and
  L2-resident bandwidth exceeds DRAM bandwidth.

## Measurement

1. Write `perf/HARDWARE_LEDGER.md` **before** touching any Hagane source.
2. Build standalone C + Vulkan microbenchmarks under `perf/` (no Hagane code):
   device limits, vec4 copy, FP32 FMA, packed f16vec2 FMA, shared memory with
   conflict patterns, L2 vs DRAM reads, empty dispatch/barrier/submit,
   `vkAllocateMemory`, and H2D/D2H staging copies.
3. Use GPU timestamp queries for device time; repeat each kernel many times in
   one command buffer; report median + spread; log `nvidia-smi` clocks and
   Vulkan/driver versions.
4. Record every number in `perf/HARDWARE_LEDGER.md` with the command that
   produced it and the calculation (bytes/FLOPs ÷ seconds).
5. Gate: if bare kernels miss the 90%/95% ceilings, fix the measurement method
   first (this is a finding, not a framework bug).

## Honesty

No PyTorch comparison, no Hagane code, and no framework optimization in Phase 1.
Negative and inconclusive results are recorded, not deleted.
