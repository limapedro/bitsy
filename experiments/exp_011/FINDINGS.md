# exp_011 — Findings: clock logging, pipeline cache attempt, minimal barriers, async overlap

**Date:** 2026-11-21
**Scope:** Phase 5 infrastructure/measurement work. No kernel algorithm changed.
Parity: CTest 1/1 Vulkan and CPU-only, 42/42 cases, gradcheck/conv unchanged.

## What was implemented

- **Per-run GPU clock logging** in `benchmarks/ops/gemm.py` and
  `elementwise.py` via `benchmarks/common/clock.py`; each result now carries
  `clockMHz`.
- **On-disk pipeline cache load path**: a device-UUID-validated
  `VkPipelineCache` is created from `BITSY_PIPELINE_CACHE` (default
  `bitsy.pipeline_cache`) and used for all compute pipelines.
- **Hazard-tracked barriers**: the unconditional global barrier after each
  dispatch was replaced by per-buffer `VkBufferMemoryBarrier`s emitted only
  when a prior dispatch in the batch wrote/read the same buffer. Transfer
  barriers are unchanged. Descriptor sets and query slots still reset per batch.
- **`benchmarks/ops/async_chain.py`**: compares wait-after-each-chain with
  submit-N-chains-then-wait (dataclass config, no argparse).

## Blocker: pipeline cache cannot be persisted on driver 621.01

`vkGetPipelineCacheData` **crashes the driver when `pData == NULL`**, and with
a preallocated buffer it returns `VK_INCOMPLETE` for every capacity from 1 MiB
to 64 MiB. No cache data can be written, so persistence is disabled and
recorded as an environment blocker. Loading an externally supplied valid cache
file remains supported. This is a negative result, not a silent omission.

## Measurements (all at 1530 MHz, logged per run)

| op | n | bitsy ms | torch ms | ratio |
|---|---:|---:|---:|---:|
| add | 4096 | 0.0694 | 0.0255 | 0.37 |
| mul | 4096 | 0.0678 | 0.0364 | 0.54 |
| relu | 4096 | 0.0616 | 0.0279 | 0.45 |
| add | 1M | 0.1162 | 0.0572 | 0.49 |
| mul | 1M | 0.1156 | 0.0524 | 0.45 |
| relu | 1M | 0.1140 | 0.0385 | 0.34 |
| GEMM | 256² | 0.249 | 0.047 | 0.19 |
| GEMM | 1024² | 1.265 | 0.903 | 0.71 |

Clock logging confirms the previously observed 0.69–0.78 GEMM spread correlates
with 1530–1950 MHz clock changes; at a fixed 1530 MHz the ratio is 0.71.

### Barriers and async overlap

Profiled 20 dependent `add`s in one submit: **20 dispatches, 19 barriers**
(only the first dispatch avoids a barrier because there is no prior writer).
The chain is a true dependency chain, so hazard tracking cannot remove barriers
here; the benefit is narrower barriers, not fewer. This is a negative result
versus the hypothesis of a material count reduction.

Async chain (20 adds × 1M elements, batch 8): sync 1.214 ms/chain, async
1.107 ms/chain, **1.10×**, below the ≥1.2× prediction. Host overlap helps but
is bounded by the chain's GPU work; the host is no longer the dominant cost.

## Predictions

- "clock logging explains the spread": **supported** (0.71 at 1530 MHz vs
  0.78 at higher clocks).
- "on-disk pipeline cache reduces startup": **blocked** by driver (above).
- "minimal barriers reduce count in a 20-op chain": **not met** (19 vs 20).
- "async host overlap ≥1.2×": **not met** (1.10×).

All negative results are preserved here; parity stayed green throughout.

## Next

The remaining elementwise gap is a kernel/bandwidth issue (86% of measured DRAM
ceiling in the chain, ratio 0.34–0.54), not submission overhead. Next work
should target vec4 elementwise kernels and reduced global traffic, with a new
`exp_NNN`, rather than more submission tweaks.
