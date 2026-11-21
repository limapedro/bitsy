# exp_001 — Caching device buffers instead of allocating per tensor

**Date:** 2026-11-21
**Owner:** Bitsy M2

## Hypothesis

Every Vulkan op allocates its output with `vkCreateBuffer` + `vkAllocateMemory`
and destroys it on release. Those driver calls dominate the 0.38–0.73 ms wall
time measured in `benchmarks/results/elementwise.json`, even for 4096-element
tensors. Reusing device buffers from size-class free lists should remove most
of that fixed cost.

## Prediction

- 4096-element `add` wall time drops from ~0.385 ms to below 0.15 ms.
- 1M-element `add` drops from ~0.675 ms to below 0.35 ms.
- Ratio versus PyTorch CUDA improves from ~0.07 to at least 0.25.

## Measurement

1. Build with `-DBT_ENABLE_VULKAN=ON`, run `python benchmarks/ops/elementwise.py`
   on the GTX 1660 Super; keep `benchmarks/results/elementwise.json` as the
   before snapshot (`experiments/exp_001/before.json`).
2. Implement the cache in `src/backends/vulkan/vk_allocator.c`.
3. Re-run the exact same command; save the after snapshot and median numbers.
4. Keep the change only if the prediction is met; otherwise record the negative
   result and revert. No silent deletion of the experiment.
