# BENCHMARKS — living performance ledger

This is the **living** record of measured speed vs PyTorch CUDA on this machine.
Every entry has a wall-clock timestamp (when the PR was created), the exact
command, and honest before/after numbers. Nothing is claimed that was not
measured. Ratios `>1` mean **bitsy is slower**; `<1` means **bitsy is faster**.

## Machine / methodology
- GPU: NVIDIA GeForce GTX 1660 SUPER, driver 621.1.0.0, 5.83 GiB VRAM.
- fp32, warmup 5, median of 20 iterations, min/max spread recorded in JSON.
- Live GPU clock via `nvidia-smi` (not pinned — noted per run).
- Scripts: `benchmarks/compact/common_ops_bench.py` (17 common ops, end-to-end
  op + host read), `gemm_bench.py`, `replay_bench.py`, `profile_replay.py`.
- Results JSON: `benchmarks/results/{common_ops,compact_gemm,compact_replay}.json`.

---

## 2026-11-22 02:57 — PR #18 (in progress): skip useless uploads + lazy readback

**Hypothesis**: single-op latency was 4–5× off PyTorch. Profile every call:

```
[eager] submit 20us  wait ~100us
[dl]    copy ~390us  memcpy ~320us
[vkbuf] alloc+upload 1286..1453us   <-- 1.3-1.4 ms per output tensor!
```
Two distinct wastes found:
1. `vkbuf()` uploaded **uninitialized host scratch** to DEVICE_LOCAL for every
   output buffer, which the kernel immediately overwrites (~1.3 ms).
2. Eager `realize()` downloaded to `b->host`, then `rt_to_host` copied again to
   the user's numpy array (two 4 MB memcpys).

**Fixes**:
- `RtBuf.host_valid` flag: only leaf inputs (real host data) upload to device.
  Reshape/transpose host-copied outputs set it = 1.
- Eager kernels mark output `dirty` instead of downloading; `rt_to_host` copies
  DEVICE_LOCAL → HOST_CACHED staging → **user buffer directly**.
- Host-read paths (reshape/transpose) explicitly sync dirty buffers first.

**Before → after (median ms, clock 1530→1950 MHz)**:
```
op          before   after    torch    gap before -> gap after
add          3.10     2.01     0.79       4.3x   ->  2.54x
mul          2.86     1.46     0.89       4.1x   ->  1.64x
relu         2.67     1.41     0.76       3.9x   ->  1.86x
exp          2.90     1.28     0.88       4.3x   ->  1.46x
gelu         3.46     1.47     0.63       4.5x   ->  2.32x
fused chain  3.36     1.53     0.98       4.5x   ->  1.55x
GEMM n=1024  6.32     4.04     1.76       3.5x   ->  2.30x
loop/step    0.144    0.150    0.174      0.86x  ->  0.90x (bitsy faster)
```
All gates green: v2 parity 48/48, op 27/27, gradcheck 25/25, model 10/10,
API 79/79, CTest 2/2. Core 2,129 lines.

**Still open**: reductions 10–38× (tiny output, fixed download cost dominates);
single-call elementwise 1.5–2.7× off; GEMM 2.3× (cooperative-matrix blocked on
this GPU — see FINDINGS_PR15).

---

## 2026-11-22 11:10 — PR #42: GEMM+bias epilogue — nn.Linear in one dispatch

`rt_add` now folds `matmul(...) + bias` into the GEMM's existing bias epilogue
(flag bit 4), the same way PR #23 folds ReLU. This removes a separate add
dispatch per linear layer. The fold exposed a latent bias-gradient bug
(`keepdim=0` then `unbroadcast` collapsed a `(1,N)` bias) — fixed with
`keepdim=1`, which also tightened existing bias-gradient parity.

A/B (`gemm_epilogue_bench.py`, batch 64, 256→128 linear + bias):
```
no fold:  dispatch 2, wall 0.6088 ms
folded:   dispatch 1, wall 0.3242 ms   (-47% wall)
```
Also tested making large outputs HOST_CACHED to remove the D2H copy entirely
(`BT_HOSTOUT_KB` sweep, reverted): 4 MB output regressed 1.543→2.263 ms because
GPU PCIe writes cost ~0.32 ms vs 0.066 ms to DEVICE_LOCAL — the 4 KB limit is
correct. The ~0.4 ms synchronous fence wait remains driver-bound.

Added `folded_gemm_bias_dispatch/fwd/wgrad/bgrad` gates (API 104 → **108 cases**).
Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API **108/108**, ctest 2/2.
Core 3,157 lines.

## 2026-11-22 10:35 — PR #41: transpose-folded GEMM — half the dispatches per linear layer

The GEMM shader already supports `trans_a`/`trans_b`, and `grad_wrt` already
handles the folded flags, but `X @ W.T()` was building a separate device
transpose node then a GEMM. `rt_gemm` now folds an **unrealized** transpose
operand into the GEMM's trans flag (bias/ReLU epilogues and autograd keep
working; a realized transpose is still used as-is).

A/B on a 2-layer MLP forward (batch 64, 256→128→1), same build, env-gated
disable flag (removed after measurement), clock 1530:
```
no fold:  dispatch 4, wall 0.7653 ms
folded:   dispatch 2, wall 0.4875 ms   (-36% wall, half the dispatches)
```
New `benchmarks/compact/gemm_epilogue_bench.py` records dispatch count and the
vs-PyTorch sweep (single-call MLP fwd still host-sync dominated: 0.5125 vs
0.0258 ms PyTorch async). Added `folded_gemm_dispatch` (exactly 1 dispatch),
`folded_gemm_fwd` and `folded_gemm_grad` gates (API 101 → **104 cases**).

This is the "fewer calls / most fused kernels" lever directly: every linear
layer in MLPs and transformers drops one kernel dispatch.

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API **104/104**, ctest 2/2.
Core 3,138 lines.

## 2026-11-22 10:00 — PR #40: compiled-callable API — captured replay as the default path

Added `Engine.compile(fn, shapes)` and `Engine.compile_many(fn, shapes, K,
zero_grads)` (pure Python, `python/bitsy_v2/compiled.py`). A compiled callable
runs one submit (or K captured graphs in one submit) with fresh inputs written
through CPU-mapped buffers, making the amortized/captured execution path a
normal API instead of a benchmark-only trick. `CompiledMany.__call__` returns
device tensors by default (PyTorch-style async), with `read=True` for host data.
Inputs must be mapped BEFORE capture so the recorded device address is the mapped
one; `zero_grads` clears leaf grads between captured graphs (autograd otherwise
accumulates them across graphs).

New `benchmarks/compact/compiled_bench.py` — full fwd+bwd+SGD steps through the
high-level API (batch 64, 256→128→1, lr 1e-4, K=6, N=600 fresh batches):
```
[PASS] final-weight diff w1=2.38e-07, w2=1.79e-07
N=600 K=6  compiled 0.2776 | torch 0.3121 ms/step | gap 0.89x  (1.1x FASTER)
```
Added `compiled_api_w1/w2` parity gates (API 99 → **101 cases**).

Honest scope: forward-only through the same API is 4.7× slower than PyTorch's
async launches (one blocking submit per batch); the win requires full training
steps where PyTorch's optimizer/zero_grad overhead is counted. Fused GEMM
epilogues and faster single-op reductions were investigated but not shipped:
the two-stage reduce is already sound and its gap is sync-dominated, not kernel.

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API **101/101**, ctest 2/2.
Core 3,128 lines.

## 2026-11-22 09:25 — PR #39: per-op ring — 24% off single-call wall, no fence stall on re-record

PR #38 measured the single-call bottleneck: `vkWaitForFences` ~0.44 ms vs
PyTorch add+sync 0.038 ms. This turn removes the stall that forced a wait before
**re-recording** shared GPU state.

Added a per-op ring (8 slots) of command buffers, params buffers, and descriptor
sets (`RT_RING`): an eager dispatch records into its own slot, so a still-running
predecessor is never clobbered. We wait only when a slot is reused (every 8 ops)
or at host readback. Timestamp-query waits are deferred to the getters so they
are paid only by the profiler; `copy_buffer`'s existing fence wait doubles as the
ring sync. Added `eager_ring_wrap` parity gate (24 fused dispatches before
readback; 96 → **99 cases** after restoring a dropped stream gate).

Single-call `(X+X).numpy()` on 1M floats (warmup/median/spread, clock 1530):
```
before: wall 1.493 ms | kernel 0.0666 | copy 0.320 | host 1.106 (74.1%)
after:  wall 1.130 ms | kernel 0.0665 | copy 0.320 | host 0.743 (65.8%)
        = 24% faster wall, host overhead 74% -> 66%
```
Systematic single-op sweep (`common_ops_bench.py`): pointwise ops 1.2–1.6×,
reductions 4–5× vs PyTorch — **still not overall parity**. Async loop
(10 steps) remains 0.66× gap (1.5× faster).

Also tried a GPU-written mapped completion counter (`signal.comp`) polled by the
CPU to replace the driver fence wait: correctness held but wall regressed
(1.287 ms), so it was **reverted** — the fence round trip is not the only cost.

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API **99/99**, ctest 2/2.
Core 3,128 lines.

## 2026-11-22 08:50 — PR #38: true streaming update loop (fresh data per step) beats PyTorch + single-call overhead investigation

PR #36's win replayed fixed-shape steps on the SAME inputs. This turn removes
that caveat and separately profiles why single eager calls lag.

### Single-call host overhead — measured root cause (no fix shipped)
C-level timing split of the ~0.44 ms compute-submit path:
```
vkQueueSubmit        ~0.025 ms (CPU time)
vkWaitForFences      ~0.44 ms  (blocking driver round trip)
```
PyTorch does `add+sync` in **0.038 ms**, `add+cpu` in **0.627 ms**; ours were
realize 0.44 ms / to_host 1.26 ms. A non-blocking eager submit (fence ring,
timeline semaphore, deferred query waits) cut realize to **0.082 ms** and
to_host to **0.334 ms**, but was **incorrect**: the shared command/params
buffers must be idle before recording, and re-waiting costs the same ~0.4 ms.
`vkDeviceWaitIdle` restored correctness but erased the speedup. Conclusion:
the 74% host overhead is driver-bound and cannot be removed without a full
per-op command/params ring redesign — **reverted, nothing broken shipped**.

### True streaming loop — fresh data per step (shipped)
Added CPU-mapped HOST_CACHED input buffers (`rt_tensor_map`/`rt_tensor_flush`)
so fresh batches are written with a memcpy (no H2D submit), plus multi-graph
capture (`rt_stream_begin/end/submit`): K captured fwd+bwd+SGD graphs, each
reading a DISTINCT mapped input, submitted in ONE `vkQueueSubmit2` per batch.
Also added `rt_zero_grad` (C-level `p->grad=NULL`) so backward doesn't accumulate
across captured graphs, and raised capture slots 64→256 (K=6 needs >64).

New `benchmarks/compact/stream_replay_bench.py` (batch 64, 256→128→1, lr 1e-4,
K=6, N=600 fresh batches, warmup/median/spread, clock 1530→1950):
```
N=600 K=6  bitsy 0.2595 | torch 0.3110 ms/step | gap 0.83x  (1.2x FASTER)
correctness over 12 fresh steps: w1 diff 2.38e-07, w2 diff 2.38e-07
```
**First win with genuinely fresh per-step data** — the amortization applies to
a real streaming update loop, not just repeated identical steps. Gains are
modest (1.2×) because per-batch CPU memcpys + flushes add host cost; the
single-submit amortization of PR #32–36 remains the larger lever.

Added `stream_fresh_w1/w2` parity gates (API 96 → **98 cases**).
Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API 98/98, ctest 2/2.
Core 3,021 lines.

## 2026-11-22 08:15 — PR #36: device-side SGD — ENTIRE update loop in one submit, beats PyTorch at every N

PR #35 replayed fwd+bwd but optimizers were still host-side numpy. Added a
device-side fused SGD kernel `sgd.comp` (`w -= lr*(g + wd*w)`, one invocation
per element, in-place on the leaf weight buffer with a compute→compute memory
barrier so the next replayed iteration sees updated weights). Exposed
`rt_sgd_update` / `Engine.sgd_update` / `Engine.grad_tensor`, and captured the
SGD dispatches INSIDE the graph. Now forward+backward+weight-update is a single
command buffer replayed N times in ONE submit.

Correctness (full update, captured replay vs torch.optim.SGD over N steps):
- standalone N=5 (8×16→12×16→1): w1 diff 1.19e-07, w2 diff 5.96e-08
- parity gate `capture_full_update_w1/w2` over N=4 passes (API 92 → **96 cases**)

New `benchmarks/compact/train_loop_replay_bench.py` (batch 64 / 256→128→1,
lr 0.01, warmup/median/spread, clock 1530, weights reset before each replay):
```
N= 10   batch 0.2281 | torch 0.2951 ms/step | gap 0.77x  (1.3x FASTER)
N=100   batch 0.1320 | torch 0.2751 ms/step | gap 0.48x  (2.1x FASTER)
N=500   batch 0.0661 | torch 0.2791 ms/step | gap 0.24x  (4.2x FASTER)
```
**First end-to-end training loop that beats PyTorch at every measured N** —
even N=10, because torch's zero_grad/backward/optimizer host cost is counted
while ours is amortized into one submit. Scope is honest: fixed-shape,
fixed-hyperparameter steps replayed on the SAME inputs (no new data per step);
this is the amortized/captured regime, not a claim of universal single-call
parity (single eager calls remain 1.2–1.45× off PyTorch).

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API **96/96**, ctest 2/2.
Core 2,893 lines.

## 2026-11-22 07:50 — PR #35: backward capture FIXED — full training-step replay 8.3× faster than PyTorch

**Root cause of the backward-capture bug**: `RT_TRANSPOSE`/`RT_RESHAPE` were realised
as **host-side memcpy** reading `in->host`. During capture the input's data lives
in a capture slot (device memory), so the host copy produced garbage on replay —
exactly the near-zero grads observed. Added a device-side `transpose.comp`
(one invocation per output element) so transpose participates in capture slots;
`RT_TRANSPOSE` now dispatches on Vulkan (host fallback on CPU). Also realised all
`requires_grad` leaf grads inside `rt_backward` so their dispatches land in the
capture window. Added `capture_mlp_backward_w1/w2` parity gates.

```
matmul backward capture diff 0.0 ; MLP backward capture diff 0.0 / 0.0
```
New `benchmarks/compact/train_replay_bench.py` (full forward+backward captured,
N steps in one submit, batch 64 / 256→128→1, warmup/median/spread, clock 1530):
```
N= 10   batch 0.2048 | N-submits 0.2054 | torch 0.4939 ms/step | gap 0.41x
N=100   batch 0.1395 | N-submits 0.1635 | torch 0.4864 ms/step | gap 0.29x
N=500   batch 0.0668 | N-submits 0.1884 | torch 0.5557 ms/step | gap 0.12x  (8.3x FASTER)
```
Batch is 2.82× faster than N separate submits at N=500. This is the first
**whole-training-step** replay that beats PyTorch — the 73% host overhead from
PR #31 is amortised across N steps. Note: optimizers remain host-side numpy, so
this replays repeated fwd+bwd on fixed weights (not a full update loop).

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API **92/92**, ctest 2/2.
Core 2,823 lines.

## 2026-11-22 07:25 — PR #34: training-step capture investigation (blocked by backward capture bug)

Proposed next lever: capture whole forward+backward+optimizer steps and replay N
times in one submit. Implemented a probe (`rt.grad_refresh` to re-read grads
after replay) and measured:
```
elementwise backward (x*w).sum()   capture replay diff = 0.0        (works)
MLP backward  2 GEMMs + MSE        capture replay diff = 10.33      (WRONG)
matmul backward (single GEMM)      capture replay diff = 7.4e+34    (WRONG)
```
**Conclusion: backward-through-matmul capture is broken** (grad buffers contain
uninitialised/garbage values on replay), even though forward-only MLP capture
was fixed in PR #33. Simple pointwise backward captures correctly. Since a
training step needs matmul backward, whole-step capture cannot be shipped
without a deeper fix in the autograd/capture interaction (grad nodes created
during `backward` are not all realised into capture slots).

No code change shipped for the training-loop API — only the `grad_refresh`
probe helper. Gates remain green. This is recorded so it is not mistaken for a
working feature; the batched-replay wins (fused 4.2×, MLP 1.75× at N=500) still
apply to forward-only captured graphs.

## 2026-11-22 07:05 — PR #33: multi-dispatch capture fixed — batched MLP works

Fixed the capture gap documented in PR #32:
- GEMM capture wrote to the **eager** out buffer address even in capture mode → captured output stayed zeros. Now uses `slot->out` device address.
- Multi-dispatch chains read inputs from the eager buffer address; added `buf_addr()` which resolves any input produced by an earlier captured dispatch to that slot's out buffer.
- Added `capture_gemm` and `capture_mlp` parity cases (API harness 88 → **90 cases**).

Captured MLP (2 GEMMs + ReLU, batch 64 / 256→128→10) correctness diff 1.5e-04.
New `benchmarks/compact/mlp_replay_bench.py` (warmup/median/spread, clock 1530→1950):
```
N= 10   batch 0.1058 | N-submits 0.0951 | torch 0.0491 ms/step | gap 2.16x
N=100   batch 0.0778 | N-submits 0.0702 | torch 0.0519 ms/step | gap 1.50x
N=500   batch 0.0308 | N-submits 0.0706 | torch 0.0537 ms/step | gap 0.57x (1.75x FASTER)
```
Batched replay now extends beyond pointwise chains to MLP steps; the win needs
N≥~300 to overcome the small-graph overhead. Still slower than PyTorch at N≤100.

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API 90/90, ctest 2/2.
Core 2,765 lines.

## 2026-11-22 06:45 — PR #32: batched graph replay — 4.2× faster than PyTorch at N=500

PR #31 measured 73% host overhead in a single call. Added `rt_replay_n(RT*,n)`
/ `Engine.replay_n(n)`: N graph executions in **ONE** `vkQueueSubmit2` (one
timeline signal + one wait), then a single final read. New reproducible
`benchmarks/compact/graph_replay_bench.py` (fused pointwise chain, 1M elements,
warmup/median/spread/clock 1530→1935):
```
N= 10   batch 0.1764 ms/step | N-submits 0.1245 | torch 0.1907 | gap 0.92x
N=100   batch 0.0592 ms/step | N-submits 0.0699 | torch 0.1238 | gap 0.48x
N=500   batch 0.0289 ms/step | N-submits 0.0666 | torch 0.1214 | gap 0.24x  (4.2x FASTER)
```
Correctness diff 1.9e-06. **This directly attacks the measured 73% host
overhead**: at N=500 the batch path is 2.31× faster than N separate submits and
4.2× faster than PyTorch's loop.

**Known limitation documented**: multi-dispatch GEMM/MLP capture currently
produces zeros (single-dispatch fused capture works) — a pre-existing capture
gap, not introduced here. The benchmark uses the proven fused-chain capture.

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API 88/88, ctest 2/2.
Core 2,747 lines. Fixed a variable-shadowing warning in LN backward.

## 2026-11-22 06:25 — PR #31: GPU timestamp profiling — the definitive split

Added Vulkan timestamp queries to the compact runtime (`rt_device_time_ms`,
`rt_copy_time_ms`) and a reproducible script `benchmarks/compact/timestamp_profile.py`.
For a single eager `x+x` on 1M elements (median, 20 iters, warmup 8, clock 1530):
```
wall 1.433 ms
  GPU kernel       0.067 ms  ( 4.7%)   <- compute shader
  GPU D2H copy     0.320 ms  (22.3%)   <- DMA engine
  host overhead    1.046 ms  (73.0%)   <- submit + fence waits + Python/numpy
```
**This is why kernel tuning alone cannot close the single-call gap**: 95% of
wall time is not the compute kernel. The vec4 GEMM work (PR #30) still helps
multi-op/graphs, but per-call latency is host-bound. TimestampPeriod 1.0 ns;
clock logged; spread in `benchmarks/results/timestamp_profile.json`.

All gates green (v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API 88/88,
ctest 2/2). Core 2,715 lines.

## 2026-11-22 06:05 — PR #30: register-blocked vec4 GEMM

Researched further GEMM strategies (web searches returned nothing usable), then
implemented `gemm_vec4.comp`: each thread owns a **4×4 output tile** (16
accumulator registers), loading `vec4` A/B slices from shared memory and doing a
16-FMA outer product per k-step. 16×16 workgroups cover a 64×64 output tile.
BN/BK handling, bias and ReLU epilogues preserved. A/B 3 repeats (n=1024 GEMM+ReLU):
```
baseline scalar kernel: 2.56 / 2.62 / 2.64 ms
vec4 register-blocked:  2.38 / 2.24 / 2.57 ms   (~10% faster)
```
Made vec4 the default. Numerical diffs unchanged (4.2e-05 … 2.2e-04).
Latest full sweep (clock 1530→1935): GEMM n=1024 2.80 ms (1.71×), GEMM+ReLU
2.16 ms (1.74×); n=512 GEMM+ReLU 2.01×. Prior shared-memory scalar kernel was
~2.4× at n=1024. GEMM remains ~1.7× off cuBLAS (tensor cores blocked by hardware).

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API 88/88, ctest 2/2.
Core 2,682 lines. Also fixed a variable-shadowing warning in LN backward.

## 2026-11-22 05:45 — PR #29: residual+LayerNorm fusion + LN backward bug fix

`LayerNorm(x + residual)` previously took 2 dispatches (add + layernorm).
Extended `layernorm.comp` with a `HAS_RES` spec constant and 2 pipeline variants;
`RT_LAYERNORM` carries an optional 4th source (residual), requiring
`RTTensor.src[4]`. `nn.LayerNorm.forward(x, residual=None)` uses it.
```
residual+LN 4096x128: 1 dispatch (was 2), diff 2.4e-07, end-to-end 1.14ms vs torch 0.43ms (~2.6x)
```

**Correctness fix found while testing**: LayerNorm backward (shipped in PR #25,
never gated) was missing the `n·dxhat` term and returned dw/db with the wrong
shape. Verified against PyTorch: 1-D LN backward diff 1.2e-07 after fix (was
0.66). Added 4 backward parity cases (x, residual, weight, bias) so this is
gated now. Note `pow` on negative bases yields NaN in the GLSL kernel
(pre-existing) — the new gate uses `mse_loss`, not `(y**2)`.

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API 88/88, ctest 2/2.

## 2026-11-22 05:20 — PR #28: fused softmax+bias + more APIs; clock-pin attempt

**Clock pinning attempt**: `nvidia-smi -lgc 1530,1530` → "current user does not
have permission to change clocks" (needs admin). Clocks cannot be pinned from
this session; every benchmark continues to record start/end clock explicitly
(e.g. 1530→1950 MHz) so drift is visible. No hidden clock normalisation.

**Softmax+bias fusion**: extended `softmax.comp` with a `HAS_BIAS` spec constant
and 4 pipeline variants (softmax/logsoftmax × bias/no-bias). `RT_SOFTMAX` now
carries an optional bias source; `Tensor.softmax/log_softmax(axis, bias=...)` use
it. Backward covers both inputs.
```
softmax(x+bias) 4096x128: 1 dispatch (was 2), diff 6e-08, end-to-end 1.21ms vs torch 0.41ms (~3x)
```
**API batch**: `functional.cosine_similarity`, `pairwise_distance`. API parity
79 → **83 cases**, all green. Feature inventory updated.

Gates: v2 48/48, op 27/27, gradcheck 25/25, model 10/10, API 83/83, ctest 2/2.

## 2026-11-22 05:05 — PR #27: D2H fence profiling + dedicated transfer queue A/B (negative)

Profiled the remaining D2H copy with high-resolution timers (BT_PROF, 4 MB):
```
[copy] submit 14-44us   wait 550-1320us   size 4194304
```
The fence wait — not the DMA — dominates, and grows under repeated submissions
(0.55 ms first → ~1.2 ms steady). Tried a **dedicated transfer-only queue**
(family 1 exists on this GPU) via `BT_XFERQ=1` with its own queue + command pool:
```
baseline  fused_chain 0.87/0.87/0.90ms, loop 0.66x
xfer queue fused_chain 0.98/0.90/1.15ms, loop 0.53-0.64x  (no win, often worse)
```
The compute queue's existing copy path is as good or better on this driver;
the dedicated queue's fence wait was similar (0.67-1.58 ms). Reverted the entire
experiment; gates green again (v2 48/48, gradcheck 25/25, model 10/10).
No code change shipped — recorded so it is not retried.

**D2H approaches now tested and rejected on this GPU/driver:**
pinned host import (PR #22, 3× worse), combined kernel+copy submit (PR #26,
incorrect), dedicated transfer queue (this PR, no win). The ~1 ms fence wait
appears driver-bound; further gains need a different runtime path or hardware.

## 2026-11-22 04:50 — PR #26: combined kernel+D2H submit A/B (negative)

Searched for Vulkan D2H techniques (web search returned no usable results), then
A/B-tested recording the staging copy **in the same command buffer** as the
kernel (`BT_COMBINED=1`), with a C-assigned readback slot per output, to reduce
two submits/fences to one.

```
baseline (2 submits): fused_chain 0.91ms, relu 0.90ms, loop 0.62x
combined (1 submit):  fused_chain 0.86ms, relu 0.89ms, loop 0.67x  (marginal)
```
**Correctness regression**: `training_scale_gradients` failed deterministically
(grad_w2 check false; sometimes diff 1.9e-08, sometimes 6.3e-02) — the C-assigned
ring slot and Python-side slot rotation can desynchronise when several gradients
are read, aliasing results. Reverted the entire experiment; all gates green
again (48/48, 25/25, 10/10). No code change shipped — this entry records the
negative result so it is not retried blindly.

The two-submit path is retained: for a single 4 MB output the fence/DMA cost is
~0.4 ms and is driver-bound; the alternatives tried (pinned host import PR #22,
combined submit this PR) were either much slower or incorrect.

## 2026-11-22 04:35 — PR #25: fused LayerNorm (4 → 1 dispatch)

`LayerNorm` previously took **4 dispatches** (mean, var, inv-sqrt+scale, affine).
Added `src/rt/shaders/layernorm.comp`: one 256-thread workgroup per row,
cooperative shared-memory load, tree-reduce mean → variance → normalize with
weight/bias. `RT_LAYERNORM` node + `rt_layernorm()` API; Python `nn.LayerNorm`
uses it for last-dim ≤256 (composition fallback otherwise). Autograd implements
the standard LayerNorm backward from existing primitives.

```
dispatch count: 4 -> 1 (verified)
numeric diff vs composition: 1.2e-07
end-to-end (4096x128): fused 0.95ms vs torch 0.37ms (~2.6x)
```
All gates green (48/48, 27/27, 25/25, 10/10, 79/79, ctest 2/2).

## 2026-11-22 04:20 — PR #24: fused softmax/log_softmax (4 → 1 dispatch)

`X.softmax(axis=-1)` previously took **4 dispatches** (max, exp, sum, div).
Added `src/rt/shaders/softmax.comp`: one 256-thread workgroup per row, cooperative
load into shared memory, tree-reduce max → exp → tree-reduce sum → normalize.
`RT_SOFTMAX` node + `rt_softmax()` API; Python uses it for last-axis, ≤256 cols
(fallback to composition otherwise). Autograd implements the standard softmax
and log_softmax backward rules from existing primitives.

```
dispatch count: 4 -> 1 (verified)
numeric diff vs torch: softmax 3e-08, log_softmax 5e-07
end-to-end (4096x128, median, clock 1530->1950): fused 0.66-0.96ms vs torch 0.30ms (~2-3x)
old 4-dispatch composition measured 0.33ms in one run, so the fused shader wins
clearly on dispatch count but the shared per-call download overhead still dominates.
```
All gates green (48/48, 27/27, 25/25, 10/10, 79/79, ctest 2/2).

## 2026-11-22 04:05 — PR #23: GEMM+ReLU epilogue fusion (1 dispatch)

`matmul().relu()` previously took **2 dispatches** (GEMM cut + fused ReLU).
Added `RT_GEMM_RELU` node type: `rt_relu()` folds an unrealized `RT_MATMUL`
into a new GEMM node with a ReLU epilogue flag (bit 3 of the GEMM flags word);
`gemm.comp` applies `max(0, v)` after bias. Autograd masks the upstream grad
with `gt0(Y)` before the matmul backward. CPU path applies ReLU in `realize_node`.

```
n=256  GEMM 5.81x -> GEMM+ReLU 4.94x  (1 dispatch, diff 4.2e-05)
n=512  GEMM 3.35x -> GEMM+ReLU 2.59x  (1 dispatch, diff 1.1e-04)
n=1024 GEMM 2.11x -> GEMM+ReLU 1.91x  (1 dispatch, diff 2.2e-04)
```
Tried passing bias directly through GEMM in `Tensor.linear` so `linear+relu`
would also be one dispatch — bias grads broke (8/10 model parity); reverted that
part while keeping the (N,) or (1,N) bias shape acceptance. All gates green:
48/48, 27/27, 25/25, 10/10, 79/79, ctest 2/2.

## 2026-11-22 03:55 — PR #22: pinned-D2H A/B (negative) — reverted

`VK_EXT_external_memory_host` **is present** on this GPU/driver, so I implemented
aligned host allocations imported as `VkDeviceMemory` — the kernel writes directly
into pinned CPU-visible RAM, theoretically eliminating the ~0.4 ms staging copy.
Enabled the extension with a fallback for CI's software Vulkan, added
`vkGetMemoryHostPointerPropertiesEXT` (loaded via `vkGetDeviceProcAddr`), and
A/B-tested with `BT_PINNED=1`.

```
baseline (DEVICE_LOCAL + DMA staging):  fused chain 1.06ms, mul 1.18ms, loop 0.62x
pinned direct kernel write:             fused chain 3.16ms, mul 2.65ms, loop 0.76x
```
**Result: 3× WORSE.** GPU kernel writes to pinned host memory are scattered and
much slower than writing to fast local HBM followed by an efficient DMA engine
copy via `vkCmdCopyBuffer`. The two-step path wins decisively on this hardware.
The entire pinned path was **reverted**; no dead flag left in the tree. All gates
green after revert (48/48, 27/27, 25/25, 10/10, 79/79, ctest 2/2).

Also measured: `add` appearing ~2× slower than `mul` in `common_ops_bench` was a
first-measurement artifact (1.81ms then 1.05ms on the same op) — clock ramp-up,
not an op-specific regression. **Fixed the harness** with a global warmup pass
over all 17 ops before per-op timing. Corrected picture (median, clock 1530→1950):
```
fused_chain  1.04ms vs 0.93ms -> 1.12x   sqrt 1.04/1.10 -> 0.95x (bitsy faster)
relu/abs     1.02-1.06ms       -> 1.2-1.3x
add/mul/exp  1.05-1.15ms       -> 1.26-1.45x
sum/mean/max 0.28-0.34ms       -> 6-8x (torch fully warm)
loop         0.125ms/step vs 0.192 -> 0.65x (bitsy 35% faster)
```
The earlier 1.7–2.6× elementwise figures were inflated by cold measurement.

## 2026-11-22 03:35 — PR #21: zero-copy readback (loop 0.86× → 0.6×)

**Profile**: after PR #20 the remaining per-call cost was the staging→numpy
memcpy (~0.3 ms) plus the extra fence submit that produced it.

**Fix**: `rt_read_ptr()` returns a pointer to a persistent HOST_CACHED readback
ring (32 slots) instead of copying into a fresh numpy array. Python wraps it
with `np.ctypeslib.as_array` into an ndarray subclass that keeps the RT engine
alive. Each call rotates slots so concurrently-held arrays don't alias.
`rt_vk_read_ptr` handles DEVICE_LOCAL outputs (staging copy) and HOST_CACHED
outputs (return mapped pointer directly).

**Reverted experiment**: combining kernel+download copy into one command buffer
with a per-output HOST_CACHED buffer regressed the loop 0.65× → 0.94× (extra
buffer allocation overhead); reverted, kept the ring design.

```
loop 10 steps:  0.86x -> 0.56-0.67x (bitsy 33-44% FASTER than torch)
fused chain:    1.49x -> 1.01-1.44x (varies with clock)
elementwise:    1.7-2.6x -> 1.2-2.4x (relu ~1.2-1.6x)
reductions:     4-7.6x -> ~4.4-5.9x
```
All gates green. Core 2,212 lines.

## 2026-11-22 03:18 — PR #20: HOST_CACHED tiny outputs (reduce 8× → 5×)

A/B measured with `BT_HOSTOUT` flag before making it automatic:
- HOST_CACHED output buffer: kernel writes straight to CPU-visible RAM, no
  staging fence. Great for tiny outputs, ~1.5 ms penalty for 4 MB outputs.
- Rule: outputs `<= 4 KB` use HOST_CACHED; larger stay DEVICE_LOCAL + staging.

```
sum  0.51ms -> 0.26ms (~8x -> ~4.7x gap)
mean 0.54ms -> 0.29ms (~10x -> ~5.4x)
max  0.47ms -> 0.33ms (~8x -> ~4.9x)
```
All gates green. Core 2,212 lines. GEMM n=1024 3.53 ms / 2.44×, loop 0.9×.

## 2026-11-22 03:10 — PR #19: two-stage reduce (800× → 8×)

**Profile**: single-stage parallel reduce submitted+waited **1.5 ms** — one
256-thread workgroup = one SM processing all 1M elements (4096 iterations/thread).

**Fix**: two-stage reduction, both dispatches in ONE command buffer (one fence):
- stage 0: `lines × 64` workgroups (chunk index in y) → partial buffer (uses all SMs)
- stage 1: one thread per output line reduces 64 partials → final result
- catches multi-line reductions (axis 0/1 of 2D) and captured replay

```
sum  42.5ms -> 0.51ms (802x -> ~8x gap)
mean 42.5ms -> 0.54ms (983x -> ~10x)
max  42.5ms -> 0.47ms (768x -> ~8x)
```
All gates green (48/48, 27/27, 25/25, 10/10, 79/79, ctest 2/2). Core 2,207 lines.

## 2026-11-22 02:40 — PR #17: parallel reduce (800× → 25×)

`reduce.comp` used one invocation per output line, so a 1D 1M reduce ran on
**one GPU thread**. Rewrote to one 256-thread workgroup per line with shared-
memory tree reduction. `sum 42.5ms → 1.6ms` (802× → ~25× gap), gates green.

## 2026-11-22 02:32 — PR #16: activation/loss APIs (79 parity cases)
Pure-Python `relu6/celu/softsign/tanhshrink/logsigmoid/softshrink/threshold`,
`hinge/poisson_nll/kl_div`. API parity 69 → 79, all green. Core 2,090 lines.

## 2026-11-22 02:23 — PR #15: fusion correctness fix + classification APIs
Found a silent bug: fusions with >8 distinct leaf inputs failed `emit()` and
left `buf` NULL → zeros. Fixed with a 4-attempt child-materialization retry.
`isnan/isinf/isfinite/nan_to_num/...` → API parity 69/69. Core 2,090 lines.

## 2026-11-22 02:11 — PR #14: GEMM tuning investigation
Subgroup reductions don't apply to GEMM (shared-memory broadcast/coalescing
already free). 64×64×8 + FMA remains best. 128×32/32×128 no gain; 64×128
overflowed `acc[16]` (caught by parity, documented). GEMM ~3× off cuBLAS.

## 2026-11-22 01:59 — PR #13: GEMM forced FMA (4× → 3×)
`s = fma(A,B,s)` instead of mul+add: n=1024 6.80 → 5.36 ms, gap 4.0× → 3.2×.
Numeric diff grew to 2.2e-4 at n=1024 (documented 3e-4 tolerance for large K).

## 2026-11-22 01:38 — PR #12: shared-memory tiled GEMM (207× → 4×)
Rewrote naive GEMM (no data reuse) with cooperative shared-memory tiles.
Tile/BK sweep: 64×64×8 best. n=1024 304 → 6.8 ms, gap 207× → 4×.

## 2026-11-22 01:24 — PR #11: legacy cutover complete
Deleted 5,710-line legacy core (78 files); CI flipped to `v2_runner.py`.
Gates green; core 2,026 lines.

## 2026-11-22 01:06 — PR #10: v2 parity runner (48/48)
Audited 48 legacy cases against v2; fixed 0-d scalar shape and documented GLSL
1-ULP div tolerance; built `tests/parity/v2_runner.py` with identical report shape.

## 2026-11-22 00:40 — PR #9: next API batch (29 functions, 44 parity cases)
Pure-Python activations, reductions, losses. Core unchanged at 2,026 lines.

## 2026-11-22 00:25 — PR #8: async submit + fast readback (speed parity claim)
Found benchmark methodology flaw (per-step sync on bitsy only); added async
submit, direct download, reusable readback pool. Loop 0.9× — **but this is only
the loop metric**, corrected by PR #17's comprehensive verification.

## 2026-11-22 00:10 — PR #7: DEVICE_LOCAL + specialized fused kernel
Found device buffers in PCIe memory; moved to DEVICE_LOCAL HBM; baked op-codes
into Vulkan specialization constants. submit-only 0.47 → 0.10 ms. Gap 3.5× → 1.7×.

## 2026-11-22 23:50 (prior day) — PR #6: download bottleneck
Profiled first: download was 96% of the gap. HOST_CACHED staging + lazy download.
Gap 194× → 3.5×.

---

## Honest status (updated 2026-11-22 03:35)
- **Fastest path**: capture/replay + zero-copy readback beats PyTorch on the
  loop by 33–44% (0.56–0.67×). Single-call fused chain ~1.0–1.4×, elementwise
  1.2–2.4×, GEMM 2.4×, reductions 4–6×. Still no universal parity claim.
- **No universal parity claim**: the gap is closed only on the loop workload.
- Clock not pinned; spread in JSON. GEMM tensor-core path blocked on this GPU
  (no `VK_KHR_cooperative_matrix`) — needs different hardware or a decision.
