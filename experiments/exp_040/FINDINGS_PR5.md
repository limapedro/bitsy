# exp_040 — FINDINGS (PR #5: capture/replay executor + speed benchmarks)

Status: PR #5 of the v0.2 rewrite. Builds on PR #1-4.

## What landed
- Real capture/replay on the compact Vulkan runtime: prebuilt command buffer
  (`replay_cmd`) with per-dispatch params+output buffer slots (dynamic args),
  recorded once and resubmitted via `vkQueueSubmit2` + timeline semaphore
  (driver-bypass). Up to 64 captured dispatches.
- `rt_upload` to refresh leaf data in place; replay copies results back to host.
- Fixed a latent bug: the fused kernel wrote output via the fixed 64KB staging
  buffer, which overflowed for large tensors; now writes via device address.
- `benchmarks/compact/replay_bench.py`: correctness (replay + new data) and
  speed (eager vs replay vs PyTorch CUDA) with warmup, median, spread, clock.

## Evidence (Windows MSVC + Vulkan 1.4, GTX 1660 Super, 1M-element fused chain)
```
$ python benchmarks/compact/replay_bench.py
  [PASS] replay numerics diff = 0.0
  [PASS] replay(new data) diff = 0.0
  eager   16.04 ms (spread 14.3-18.7)
  replay  10.85 ms (spread 10.4-12.5)  -> 1.48x faster than eager
  torch    0.056 ms                    -> torch/replay = 194x (Bitsy far slower)
  clock 1530 MHz
```
- No regressions: op parity 27/27, gradcheck 25/25, model 10/10, legacy 48/48,
  ctest 3/3. Compact core **1,627 lines** (budget 4096, 2469 spare).

## Key results
1. **Driver bypass works**: replay is 1.48× faster than eager for a fused chain —
   eliminating per-dispatch command-buffer recording / submit / fence overhead
   is a real, measurable win, exactly as tinygrad's research predicted.
2. **Speed parity with PyTorch is NOT achieved** — Bitsy replay is ~194× slower
   on this 1M-element chain. PyTorch fuses the whole chain into one optimized
   kernel that stays on GPU; our replay still does a 4MB device→host copy per
   step (to_host contract) and our kernel is not bandwidth-optimized.

## Honesty / what is NOT in PR #5
- The replay measurement includes the device→host output copy; a pure-GPU
  submit-only number would be smaller but would not represent a usable step.
- Capture is limited to fused/gemm/reduce dispatches with static shapes;
  reshape/transpose (host-side copies) are not captured.
- No multi-step训练 loop replay yet; no Vulkan perf tuning (PR #6).
- Speed goal (≥1.0× PyTorch) remains unmet and unclaimed.

## Bugs found and fixed
- Fused output via fixed 64KB staging buffer overflowed for large tensors →
  now writes via device address.
- Capture flag wasn't exercised because graphs were realized after the capture
  window → benchmark now realizes inside capture.
- Capture slots needed per-dispatch output buffers sized to the output, and
  device addresses, to avoid replayed dispatches clobbering each other.
