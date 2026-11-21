# exp_011 — Phase 5: clock logging, pipeline cache, minimal barriers, async overlap

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5

## Prerequisite

`GAP_LEDGER.md` and exp_010 show GEMM ratio 0.69–0.78 varying with GPU clock
(1530–1950 MHz), and elementwise ratios 0.23–0.87. The profiler records 50
barriers for 50 dispatches — one global barrier per op — and the benchmark
forces a host wait after every iteration.

## Hypothesis

Three concrete overheads remain: (1) measurements are not clock-normalized,
(2) pipeline creation cost is paid on every process start, (3) unconditional
global barriers and per-iteration host waits serialize work that is independent.

## Predictions

- Logging GPU clocks per run explains the 0.69–0.78 GEMM ratio spread.
- An on-disk pipeline cache reduces second-process startup time (measured by
  time from process start to first dispatch).
- Hazard-tracked per-buffer barriers reduce barrier count in a 20-op chain
  while preserving all 42 parity cases.
- Letting the CPU submit several chains before one wait reduces wall time per
  chain by ≥1.2× versus waiting after every chain.

## Measurement

1. Snapshot current elementwise/GEMM JSON into `before_*.json`.
2. Add clock sampling to `benchmarks/ops/gemm.py` and `elementwise.py`
   (dataclass config, no argparse) and record `clockMHz` per result.
3. Add `VkPipelineCache` load/store with a device-ID-validated header.
4. Replace the dispatch-global barrier with per-buffer hazard barriers; keep
   transfer barriers intact; verify with parity and profiler counts.
5. Add `benchmarks/ops/async_chain.py` comparing wait-every-chain with
   submit-N-then-wait.
6. Re-run elementwise/GEMM, parity, CTest; write FINDINGS including negatives.

## Honesty

No framework kernel algorithm changes in this experiment. If minimal barriers
or async overlap do not help, the negative result and profiler evidence are
recorded. All parity checks must stay green; a regression blocks the PR.
