# exp_015 — Findings: full-step capture and replay (CUDA-graph equivalent)

**Date:** 2026-11-21
**Scope:** Phase 5 submission. No kernel or tolerance change. Parity: 42/42,
CTest 1/1 Vulkan and CPU-only.

## Root cause discovered before implementation

A naive full-step capture **deadlocked**: the interim autograd's `_unbroadcast()`
does a CPU roundtrip (`g.numpy()`) while reducing bias gradients. During
capture, `bt_vk_flush()` is a no-op, so `bt_vk_wait()` in that readback tried to
collect timestamp queries with `VK_QUERY_RESULT_WAIT_BIT` from the not-yet-
submitted capture command buffer — an infinite wait. Instrumented evidence was
 captured by repeatedly tracing `numpy()` calls in backward (the stall occurred
at `numpy() called shape=(64, 10)` from `unbroadcast (64,10) -> (64,1)`).

## Implementation

- A capture is now a list of command-buffer **chunks**. At a host readback
  inside a capture, `bt_vk_capture_sync_chunk()` ends + submits + waits the
  current chunk (so the readback staging buffer is valid), then continues
  recording the rest of the step into a fresh chunk. `bt_vk_replay(N)`
  submits all chunks in order N times with one final host wait.
- Critically, chunk submission **does not recycle** staging slots, scratch
  buffers, descriptor sets or timestamp slots — a replayed chunk still
  references all of them. A first implementation that did recycle produced
  bit-wrong replays (captured loss 14.99 vs eager 2.44, diverging on further
  replays); removing the recycling fixed it (diff 0.0).
- **Limitation (documented in the public header):** after the completion wait,
  resources are recycled, so the captured chunks must not be replayed again
  without re-capturing. Re-capturing per trial crashed (staging/chunk
  accumulation), so the benchmark captures once and times replay+wait only;
  numerics are verified in a separate single-replay check.

## Measurements (MLP 784→64→10, batch 64; new `benchmarks/models/graph_step_bench.py`)

| path | ms/step | note |
|---|---:|---|
| Bitsy eager (exp_014, 1530 MHz) | 3.220 | default API |
| Bitsy captured+replayed | **0.294–0.355** | 100 replays/wait, 10 trials |
| PyTorch eager (exp_014) | 1.20 | `synchronize()` per step |
| PyTorch CUDA graph | **0.067–0.068** | 100 replays/sync, 10 trials |

- Captured step is **~9–11× faster than eager Bitsy** (clock differed: eager
  measured at 1530 MHz, graph runs at 1732–1935 MHz; even clock-normalised the
  speedup is large).
- **Graph throughput ratio PyTorch/Bitsy = 0.19–0.23**, far below the ≥0.85
  target. The ≥0.80 intermediate gate is also missed. Each replay submits two
  chunks (~36 µs submit+fence each) plus many small GPU dispatches, while
  PyTorch issues one graph launch with fused kernels.

## Predictions

- "≥0.80× PyTorch CUDA graph throughput": **not met** (0.19–0.23).
- "≥2× faster than eager Bitsy": **met** (~9–11×).
- "captured step numerically identical to eager": **met** —
  `eager_loss == captured_loss == 2.437650203704834`, abs diff **0.0**.
- "input refresh works": **not implemented** — Bitsy has no in-place copy API,
  so replay uses static input buffers. PyTorch CUDA graph benchmarks also use
  static inputs; this is stated in the report and is not hidden.

Regression check: `replay_chain.py` gave 0.888 ms/chain and 1.00× vs PyTorch
graph on rerun, matching exp_013 (an intermediate 1.133 ms sample at 1920 MHz
was clock/variance, not a regression). Elementwise 0.42–0.58 and GEMM 1024²
0.72 at 1530 MHz are unchanged; parity 42/42.

## Interpretation / next

Full-step capture is now functional and bit-exact, and it is the fastest
Bitsy execution path by a wide margin — but it is still ~4–5× slower than
PyTorch CUDA graphs. The dominant remaining costs are (a) one extra submit per
replay (two chunks because of the CPU roundtrip in `_unbroadcast`), and (b) the
many small dispatches from the interim Python autograd. The next experiment
should remove the CPU roundtrip by implementing device-side reductions for
`_unbroadcast` (single-chunk capture), and then measure again — with parity
re-verified, since reduction order changes fp32 rounding.

Raw data: `experiments/exp_015/{graph_step,after_elementwise,after_gemm,
replay_chain_regression}.json`, `before_*.json`.
