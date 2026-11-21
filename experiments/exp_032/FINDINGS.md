# exp_032 — lazy device-time timestamp collection (neutral)

**Date:** 2026-11-21
**Scope:** Phase 5 submission overhead. Parity: 43/43 incl. gradient gate,
CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 3.278e-07, CIFAR 7.153e-07.

## Implementation

Gated device-time timestamp recording (`bt_vk_timestamp_start/end`) and the
`vkGetQueryPoolResults(..., WAIT_BIT)` + query-pool reset block in `bt_vk_wait()`
on a `timestamps_wanted` flag set by `bt_vk_device_time_ms()`. The profiler uses
a separate query pool and was left untouched. Callers that never ask for device
time no longer pay an extra GPU sync + reset on every wait.

## Measured

| metric | before (exp_031) | after | prediction | status |
|---|---:|---:|---|---|
| captured step @1740 MHz | 0.156 ms | **0.145–0.166 ms** (ratio 0.46–0.49) | ≤0.153 | not reliably met |
| eager step | 0.906–1.042 ms | **0.894–0.899 ms** (ratio 1.23–1.24) | ≥1.1× vs 0.906 | ~1.04×, missed |
| elementwise eager 1M | 0.42–0.47 | 0.39–0.52 | ≥1.1× | not met |
| elementwise captured (exp_031 methodology) | 0.88–1.03 | 0.86–1.06 | — | unchanged |
| GEMM 1024² | 0.68–0.76 | 0.68 | — | unchanged |

`elementwise.py`/`gemm.py` call `bt.device_time_ms()` each iteration, so the
flag is set after their first sample and the optimization cannot help them —
which explains why the eager elementwise ratios did not move. The captured-step
benchmark does not request device time, so it benefits (one run reached 0.145 ms
/ 0.49 ratio), but variance across runs (0.145–0.166) prevents claiming the
≥1.1× prediction.

**Behavior note:** the first `bt.device_time_ms()` call returns 0 because the
flag is set at call time, after preceding waits recorded no timestamps; the
next call returns the correct value (verified: 0.0 then 0.00272 ms). Median-based
benchmarks are unaffected; a one-shot user could be surprised.

## Predictions

- "eager ratios improve ≥1.1×": **not met**.
- "parity green": **met**; device time correct once requested (cold first sample).
- "no regression": **met** (training step slightly faster, GEMM/elementwise
  within variance).

Kept as a small architectural cleanup (avoiding unconditional work), recorded
as a neutral result because the speed prediction was not met.

## Interpretation / next

The eager submit/fence gap is dominated by WDDM queue submission itself, not by
device-time query collection — exp_031's classification stands. With elementwise
kernels at captured parity, GEMM ladder exhausted, and per-wait trim not helping,
the remaining structural gap is the eager API's per-op submit and the captured
step's ~19 dispatches. A step-level fused kernel would be the only high-upside
change; otherwise the honest conclusion is a practical ceiling on this GPU.
All changes must still pass the 43-case gate.

Raw data in `experiments/exp_032/{before_*,after_*}`.
