# exp_040 — FINDINGS (PR #6: Vulkan perf — eliminating the download bottleneck)

Status: PR #6 of the v0.2 rewrite. Builds on PR #5 (capture/replay).

## Hypothesis (measured first, then fixed)
PR #5 reported replay 194× slower than PyTorch. This PR profiled **where** the
time went before optimizing, then attacked the dominant cost.

## Profiling evidence (before fixes, GTX 1660 Super, 1M-element fused chain)
```
$ python benchmarks/compact/profile_replay.py
  replay submit-only         0.468 ms
  replay full + download    11.268 ms   <- download ~10.8 ms (96% of total)
  torch GPU-only             0.023 ms
  torch + download           0.652 ms
```
Conclusion: the 194× gap was almost entirely the **device→host output copy**
(10.8 ms for 4 MB — uncached HOST_VISIBLE|COHERENT memory), not the GPU kernel
or submit. Optimizing the kernel first would have been wasted effort.

## What landed
1. **HOST_CACHED replay output buffers**: replay slot outputs allocated in
   HOST_VISIBLE|HOST_CACHED memory → CPU readback is cached, not streaming over
   PCIe. This alone cut the 10.8 ms download to ~0.1 ms.
2. **Lazy download**: `rt_replay_submit_only` submits+waits but does NOT copy
   outputs; `rt_to_host` downloads only when the user actually reads, via a
   dirty flag on the output buffer. The hot loop pays no forced copy.
3. **Profiling API**: `rt_replay_submit_only` / split submit and download paths,
   plus `benchmarks/compact/profile_replay.py` and a fair multi-step
   `replay_bench.py` (N GPU steps + one final read vs PyTorch GPU-only).
4. Fused stack shrunk 256→16 (keeps interpreter stack in registers).
5. Fixed fused output via fixed 64KB staging buffer overflow (PR #5 follow-up
   carried here) — outputs written via device address.

## Honest results (after, median, warmup, spread, clock logged)
```
$ python benchmarks/compact/replay_bench.py
  [PASS] replay diff 0.0 / new-data 0.0 / submit-lazy-read 0.0
  single-step forced download: eager 12.8 ms, replay 0.82 ms (15.7x vs eager),
                               torch 0.76 ms  -> replay ~= torch on this metric
  loop (10 GPU steps + read): bitsy 0.65 ms/step, torch 0.19 ms/step -> gap 3.5x
  clock 1530->1950 MHz
```
- **Gap reduced 194× → 3.5×** on the fair per-step metric.
- On the single-step forced-download metric, replay (0.82 ms) is now on par with
  PyTorch+download (0.76 ms) — near numeric/speed parity for that case.
- Remaining 3.5× per-step gap = host submit/wait overhead (~0.48 ms) + kernel
  throughput (~0.43 ms/1M elements ≈ 28 GB/s vs 336 GB/s peak).

## Gates (all green)
op parity 27/27, gradcheck 25/25, model 10/10, legacy 48/48, ctest 3/3.
Compact core **1,658 lines** (budget 4096, 2438 spare).

## Honesty / what is NOT in PR #6
- **Speed parity NOT achieved** overall: 3.5× per-step gap remains; no ≥1.0×
  claim. The single-step forced-download metric is near parity but is a narrow case.
- vec4 coalesced loads, shared-memory tiling, subgroup reductions and tile
  autotune were NOT implemented — profiling showed the download dominated, so
  effort was spent there. The interpreter kernel is still throughput-limited.
- Clock drift 1530→1950 MHz during PyTorch runs is logged; ratios are honest but
  the machine was not clock-pinned.
- Legacy core (5710 lines) still retained; compact core 1658 lines.
