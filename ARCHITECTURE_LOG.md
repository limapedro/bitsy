# Architecture Log

Decisions are recorded as dated entries. Each entry has **Context**, **Options
considered**, **Decision**, and **Consequences**. A locked decision is changed
only by a later entry that supersedes it and explains why.

---

## 2026-11-21 — Core language is C11, with no C++ in the core

**Context.** Bitsy must be a small, auditable framework whose internals a C
reader can follow. PyTorch's ATen is C++, but the brief requires a C core.

**Options considered.** (1) C11 core with a thin C ABI, (2) C++ core with a C
ABI shim, (3) Rust core.

**Decision.** The core library (`src/core`, `src/ops`, `src/autograd`, the
backends, and the C binding) is C11. No C++ source is compiled into the core.

**Consequences.** We mirror PyTorch's *design* rather than its code structure
verbatim. Opaque structs and explicit functions replace methods and templates.
MSVC is invoked with `/std:c11`; gcc/clang with `-std=c11`. Thread-local state
uses a compiler abstraction macro because MSVC's C11 support differs from
gcc's.

---

## 2026-11-21 — Refcounted storage/tensor split with explicit ownership

**Context.** PyTorch views share storage; Bitsy must reproduce view semantics
and detect invalid in-place edits during backward.

**Options considered.** (1) One object with embedded data and copied views,
(2) refcounted storage plus a lightweight tensor view, (3) arena with manual
lifetimes.

**Decision.** `bt_storage_t` is a refcounted buffer owning bytes and a version
counter. `bt_tensor_t` is a view holding `(storage, offset, sizes, strides,
dtype, device, requires_grad)`. Both expose `retain`/`release`. No hidden
global ownership.

**Consequences.** Views are cheap and share in-place writes. Every operation
that mutates a tensor bumps the storage version; autograd records version
counters to detect invalid in-place modification. Refcount misuse is a
programming error surfaced by sanitizers in CI.

---

## 2026-11-21 — Dispatcher keyed by (op, backend, dtype)

**Context.** Each operator must run on `cpu_ref` and `vulkan`, and backends may
specialize individual dtypes.

**Options considered.** (1) A global switch in every op, (2) a dispatcher
table keyed by `(op, backend, dtype)`, (3) virtual function tables in C.

**Decision.** A dispatcher table keyed by `(op id, backend id, dtype id)` maps
to a backend kernel function pointer, with a generic fallback and a clear
`BT_ERR_NOT_IMPLEMENTED` result when no entry exists.

**Consequences.** Adding a backend kernel is a registration, not an edit to
the operator body. The table is the single place to audit coverage.

---

## 2026-11-21 — Error model: status codes plus a thread-local message

**Context.** Library code must never call `exit()` or `abort()` for ordinary
errors, and Python bindings need to raise exceptions with PyTorch-like types
and intent.

**Options considered.** (1) Exceptions across the C ABI, (2) `bt_status_t`
returns plus a thread-local message buffer, (3) global error singleton.

**Decision.** All fallible API functions return `bt_status_t`. A thread-local
message buffer is readable with `bt_get_last_error()`. Only proven-unrecoverable
internal corruption may abort, and each such site must be documented at the
call.

**Consequences.** Error messages survive across the C/Python boundary. Tests
assert both the status code and the message intent. Thread-local storage avoids
cross-thread contamination.

---

## 2026-11-21 — Python binding strategy: ctypes bring-up, C extension at M6

**Context.** M0 needs Python to call the C core so the parity harness can
compare `add` against PyTorch. The final binding is a hand-written CPython C
extension (milestone M6).

**Options considered.** (1) Write the full CPython extension now, (2) a
`ctypes`/`cffi` shim for early bring-up and replace it at M6, (3) implement the
core in Python first.

**Decision.** Use a narrow, explicit `ctypes` shim (`python/bitsy/_ctypes.py`)
that loads the built shared library and wraps only the bring-up API. It is
marked as transitional in `DEVIATIONS.md` and in its docstring, and is replaced
by the C extension in M6. No third-party runtime dependency is introduced.

**Consequences.** M0 can verify numeric parity before the binding surface is
complete. The shim is not the public `Tensor` API; it is test tooling.

---

## 2026-11-21 — Licensing: BSD-3-Clause with PyTorch attribution

**Context.** Bitsy studies PyTorch's API and source for semantics. PyTorch is
BSD-3-Clause.

**Options considered.** (1) BSD-3-Clause with a NOTICE file, (2) MIT, (3)
Apache-2.0.

**Decision.** License Bitsy under BSD-3-Clause and include `NOTICE` crediting
the PyTorch project and reproducing its license text.

**Consequences.** Derived semantics and algorithms are attributed. PyTorch is
an optional test/benchmark dependency only and is never imported by the runtime
package.

---

## 2026-11-21 — Backend order: `cpu_ref` is the first oracle

**Context.** A Vulkan bug must be distinguishable from a misunderstanding of
PyTorch semantics.

**Options considered.** (1) Implement Vulkan first and debug against PyTorch,
(2) implement naive C first, cross-check it against PyTorch, then use it as a
second oracle for Vulkan.

**Decision.** Implement and verify `cpu_ref` against PyTorch first. Only then
compare Vulkan against both PyTorch and `cpu_ref`.

**Consequences.** M1 (core on `cpu_ref`) precedes M2 (Vulkan bring-up). CI on
GPU-less runners can run the full correctness suite with `cpu_ref` and software
Vulkan.

---

## 2026-11-21 — Build system: CMake + Ninja, SPIR-V compiled at build time

**Context.** Windows is primary, Linux is required in CI; shaders must be
compiled from GLSL to SPIR-V reproducibly.

**Options considered.** (1) Hand-written Makefiles, (2) CMake + Ninja with
`glslc` at build time, (3) runtime GLSL compilation.

**Decision.** CMake 3.20+ with Ninja. A `bt_add_shader()` helper compiles each
`.comp` to `.spv` with `glslc` from the Vulkan SDK. Vulkan headers are 1.3, but
the targeted feature set is Vulkan 1.2 as exposed by Turing; optional
extensions are always queried, never assumed.

**Consequences.** A single build command works on both platforms. Missing
`glslc` is a configuration error only when shaders are present. The shared
library produced by the build is what the ctypes shim loads.

---

## 2026-11-21 — Platform and TDR constraints

**Context.** The target GPU is a GTX 1660 Super (TU116, 6 GB, no Tensor Cores).
Windows WDDM resets a GPU dispatch lasting roughly two seconds.

**Options considered.** (1) Single large dispatches, (2) tiled, bounded
submissions with explicit synchronization.

**Decision.** Every kernel and benchmark is tiled so no single submission
approaches the TDR budget. Tests and benchmarks enforce timeouts and keep
allocations well below 6 GB. FP32 is the primary compute path; FP16 is stored
as packed 16-bit and accumulated in FP32; BF16 is stored as `uint16` with
explicit rounding. No cooperative-matrix path is used on this hardware.

**Consequences.** Some workloads need multiple submissions, which adds launch
overhead that the small-op latency benchmark must account for. Performance
results must not be extrapolated to GPUs with matrix hardware.

---

## 2026-11-21 — Numeric parity harness and tolerance policy

**Context.** "Verified" means a command was run comparing both implementations.

**Options considered.** (1) Per-op ad-hoc scripts, (2) one declarative,
dataclass-configured harness with a single tolerance table.

**Decision.** `tests/parity/` contains a declarative case list (op name,
argument generators, dtypes, shapes, differentiability flag), a single
tolerance table in `tolerances.py`, and a runner that compares PyTorch CPU,
PyTorch CUDA (when present), `cpu_ref`, and `vulkan`. Exact comparison is used
for integer/bool ops, data movement, comparisons, `where`, min/max selection,
and deterministic correctly-rounded fp32 add/sub/mul/div/sqrt. Transcendentals
use ULP distributions measured against a float64 reference. Reductions, GEMM,
convolution and normalization use rtol/atol against both PyTorch and a float64
reference.

**Consequences.** Adding an operator means adding one declarative entry.
Loosening a tolerance requires a new entry here with numeric justification and
must never be done to hide a red test.

---

## 2026-11-21 — Experiment and configuration conventions

**Context.** Optimization work must be reproducible and must preserve negative
results.

**Options considered.** (1) Ad-hoc notebooks and scripts, (2) numbered
`experiments/exp_NNN/` directories with `HYPOTHESIS.md` written first and
`FINDINGS.md` written after, (3) no record.

**Decision.** Any uncertain optimization gets an `exp_NNN` directory containing
the hypothesis, prediction, measurement method, code, raw results and findings
(including failures). Python configuration uses `@dataclass`, never
`argparse`. Subsystems and milestones use memorable human names.

**Consequences.** Failed experiments remain in the tree. Benchmark runners and
test generators are instantiated from dataclass config modules.

---

## 2026-11-21 — Precision strategy for reduced-precision dtypes

**Context.** Turing has no tensor cores and no native BF16. FP16 and BF16 must
still be useful and numerically comparable to PyTorch.

**Options considered.** (1) Native 16-bit ALU math, (2) store FP16/BF16 packed
in 16 bits, convert to FP32 for arithmetic and accumulate in FP32, (3) promote
everything to FP64.

**Decision.** Store FP16 and BF16 as packed 16-bit values; perform arithmetic in
FP32 with FP32 accumulators for FP16 inputs. BF16 uses explicit round-to-nearest-
even conversions. FP64 is used only where `shaderFloat64` is available and is
expected to be slow on this GPU.

**Consequences.** Reduced-precision GEMM will not match PyTorch's tensor-core
throughput, but it matches the honest regular-core baseline and keeps accuracy
no worse than PyTorch's FP32-accumulate path.

---

## 2026-11-21 — Project rename: Hagane is now Bitsy (supersedes all prior naming)

**Context.** The user requested the project be called Bitsy. Every user-visible
name, document, header, symbol prefix and Python import must change; the
architecture and milestone history stay intact.

**Options considered.** (1) Alias `bitsy` while keeping `hagane` internally,
(2) full rename of repo, documents, paths, C ABI prefix (`hg_` → `bt_`) and
Python package (`import bitsy as bt`), (3) rename only the repository.

**Decision.** Full rename. GitHub repo `limapedro/hagane` → `limapedro/bitsy`;
`include/hagane` → `include/bitsy`; `hg_*.h` → `bt_*.h`; `hagane.h` →
`bitsy.h`; `python/hagane` → `python/bitsy`; C symbols `hg_*`/`HG_*` →
`bt_*`/`BT_*`; shared library `bitsy.dll` / `libbitsy.so`; Python import
convention `import bitsy as bt`. Version `v0.1.0-m0` stays published and a new
`v0.2.0-bitsy` tag marks the rename.

**Consequences.** No compatibility shim for the old `hg_` ABI is provided: this
is a pre-1.0 rename and every call site in the tree is updated atomically.
PyTorch API parity is unaffected — only the Hagane-specific prefix changed.

---

## 2026-11-21 — GitHub name collision: preserve the old `bitsy` repository

**Context.** `limapedro/bitsy` already existed: a 11 MB deep-learning framework
last updated 2024-05-15 with real commits (`add CNN layers`, `conv2d`, MNIST
training). It blocked the requested name.

**Options considered.** (1) Delete it, (2) rename it and keep all history,
(3) leave it and use a different name for this project.

**Decision.** Rename the old repo to `limapedro/bitsy-legacy-2023`. Nothing was
deleted; its full history and data remain available. The other two similarly
named repos (`bitsy_2`, `bitsy_alpha`) do not collide with the exact name and
were left untouched.

**Consequences.** The exact `bitsy` name is now owned by this project. The
legacy repo is preserved read-mostly as a historical artifact.

---

## 2026-11-21 — Code style follows tinygrad / geohot: small, dense, commented

**Context.** The user asked for geohot's style and pointed at tinygrad. I read
the actual source rather than guessing:
`https://raw.githubusercontent.com/tinygrad/tinygrad/master/tinygrad/helpers.py`,
`.../tinygrad/tensor.py`, `.../tinygrad/nn/optim.py`. Observed rules there:
2-space indentation; many one-line functions that return directly; walrus
assignment inside conditions; `@dataclass(frozen=True)` for configs;
`# *** section ***` banners instead of structural boilerplate; short `NOTE:`
and explanatory comments that say *why* ("Python has a non moving GC, so this
should be okay", "a self assign is a NOOP", "classic momentum does post
learning rate update"); `assert` for internal invariants and `RuntimeError`
with context for user errors; tiny helpers that do one thing.

**Options considered.** (1) Keep the verbose, heavily-separated style of M0,
(2) adopt the tinygrad conventions above in both C and Python, (3) adopt only
the Python conventions.

**Decision.** Adopt the conventions in both languages, adapted to C11:
2-space indentation; small functions with early returns; no wrapper layers
that add no behaviour; section banners; comments on non-obvious lines that
explain *why*; explicit invariants at API boundaries; keep the public header
comments required by the quality bar. Python uses 2-space indentation, one-line
methods where readable, walrus assignment, `__slots__`, frozen dataclasses and
lazy imports.

**Consequences.** Files get shorter and denser. Behaviour must not change:
every refactor is re-verified with CTest and the 13-case PyTorch parity suite.
This is a style change only, not a semantic deviation from PyTorch.

---

## 2026-11-21 — Vulkan backend: one device, one command buffer, one shader

**Context.** M2 needs GPU compute on the GTX 1660 Super with Vulkan 1.2 features
only. WDDM TDR makes long submissions dangerous, and descriptor churn is the
usual first source of overhead.

**Options considered.** (1) Per-dispatch command buffers, descriptor sets and
pipelines, (2) one persistent command buffer, descriptor set and pipeline with
a fence wait per dispatch, (3) timeline semaphores and multiple in-flight
submissions from day one.

**Decision.** Option 2 for bring-up. One `VkCommandBuffer`, one
`VkDescriptorSet`, one compute pipeline from an embedded SPIR-V module, one
fence. A single GLSL shader handles all eight fp32 elementwise ops via a push
constant; the op code is uniform across invocations, so there is no branch
divergence. Push constants carry the broadcast shape/strides, so no shape
buffers are needed. Each dispatch is at most `ceil(n/256)` workgroups, bounded
by tensor size and far below the TDR budget.

**Consequences.** Correct and easy to audit; small-op latency is bounded by
submit + fence wait (~0.06 ms) rather than kernel time. Asynchronous streams,
timestamp queries and validation-layer integration are deliberately deferred
and tracked as follow-up work, not claimed as done.

---

## 2026-11-21 — Device buffers are cached by size class (exp_001)

**Context.** The first Vulkan benchmark showed 0.38–0.73 ms per elementwise op,
mostly driver allocation calls, with a ratio of only ~0.07 versus PyTorch CUDA.

**Options considered.** (1) Keep allocating per tensor, (2) cache whole device
buffers by power-of-two size class, (3) build a byte-granular caching
suballocator immediately.

**Decision.** Option 2, as a hypothesis-first experiment (`experiments/exp_001`:
`HYPOTHESIS.md`, before/after JSON, `FINDINGS.md`). It met its predictions:
5–6x lower wall time and ratios of 0.20–0.45. Negative result preserved: the
original numbers remain in `before.json`.

**Consequences.** The §7.2 bandwidth target is still missed; `BENCHMARKS.md`
says so plainly. A finer suballocator and fewer submissions are future
experiments, not silent tweaks. Cached buffers are released at shutdown.

---

## 2026-11-21 — Vulkan fp32 division uses a float64 intermediate

**Context.** The expanded 33-case parity suite found `div_vulkan` mismatching
PyTorch CUDA at one ULP in 22 of 91 elements (e.g. -0.789525568 vs
-0.789525628). PyTorch CPU and CUDA agree bit-for-bit, and `cpu_ref` div is
exact, so the discrepancy is the GLSL division path, not a semantics error.
The tolerance policy requires exact fp32 div; loosening it would hide a real
precision difference.

**Options considered.** (1) Switch div to rtol/atol, (2) Newton/fma refinement
of the approximate quotient, (3) compute `float(double(a) / double(b))` using
the optional `shaderFloat64` feature.

**Decision.** Option 3, with the feature queried from
`vkGetPhysicalDeviceFeatures` before enabling it. Turing exposes
`shaderFloat64 = true`; the double intermediate is then rounded once to fp32,
giving the correctly rounded result. `precise` alone (tried first) did not fix
the mismatch.

**Consequences.** All 33 parity cases now pass, including div on Vulkan. FP64
division is slower than the fp32 path, so it should be replaced later by a
documented correctly-rounded fp32 sequence if an experiment proves one exists
on this driver. No tolerance was loosened.

---

## 2026-11-21 — GEMM and transpose: naive first, optimization later

**Context.** Training an MLP needs `matmul` and its transpose-based backward.
cuBLAS is heavily tuned, so a parity-first kernel is more valuable than an
unverified fast one.

**Options considered.** (1) Tiled shared-memory SGEMM immediately, (2) naive
one-lane-per-output dot products with correct strides, then optimize with an
experiment, (3) CPU fallback for matmul.

**Decision.** Option 2. `bt_matmul` and `bt_transpose` are registered on both
backends; the Vulkan GEMM shader uses an 8x8 workgroup and an fp32 accumulator.
Transpose reuses the elementwise shader with swapped strides (op code 8), so no
extra pipeline is needed. Tolerance for matmul is rtol/atol 1e-5 because
accumulation order legitimately differs from cuBLAS.

**Consequences.** 40 parity cases now pass, including 3 matmul cases with zero
mismatches. Performance is not claimed; tiling/register blocking becomes its own
`exp_NNN` with before/after numbers, following the M5 plan.

---

## 2026-11-21 — Interim Python-side autograd, nn and optim

**Context.** End-to-end training parity needs gradients, parameters and an
optimizer. The full C autograd engine (M4) and CPython extension (M6) are still
ahead of this work.

**Options considered.** (1) Build the C autograd engine first, (2) implement a
small dynamic graph in the Python front end that delegates every forward and
backward *compute* op to the C/Vulkan dispatcher, (3) skip training until M6.

**Decision.** Option 2, explicitly interim and recorded in `DEVIATIONS.md`.
`bitsy.autograd.Tensor` records `(backward, parents)` and runs `add/sub/mul/div/
matmul/transpose/exp/log/tanh/sigmoid/relu` through the C core; `bitsy.nn`
provides Parameter, Linear, activations, Sequential, MSE and cross-entropy;
`bitsy.optim.SGD` updates parameters with C ops. Gradients are Bitsy tensors,
so all compute stays on the selected backend. `_unbroadcast` currently uses a
CPU roundtrip for gradient shape alignment; this is a correctness-first stopgap.

**Consequences.** Training parity can be verified now. The graph engine must
eventually move into C (M4/M6), and the CPU roundtrip must be replaced by a
reduction kernel. No PyTorch is imported by the runtime package; torch appears
only in the parity/training scripts.

---

## 2026-11-21 — Conv2d on both backends, and a large-batch eval instability

**Context.** CIFAR-10 needs convolution and its backward. cuDNN is the PyTorch
oracle. During the first CIFAR run, per-epoch losses matched PyTorch CUDA to
~1e-7, but evaluating all 512 test samples in one Vulkan call intermittently
produced a handful of completely wrong logits (one sample: 4.54 vs -0.32).
Evaluating the same sample alone, or in 64-sample chunks, matched PyTorch.

**Options considered.** (1) Report the divergence as a model bug, (2) treat it
as a suspected large-batch Vulkan hazard, chunk evaluation, and record the
observation without claiming the root cause is fixed, (3) loosen accuracy
tolerance and hide it.

**Decision.** Option 2. `Conv2d` and its three backward kernels (dX, dW, dB)
run on both cpu_ref and Vulkan through a separate conv dispatcher; stride and
padding are push constants. The CIFAR script evaluates in 64-sample chunks,
matching the training batch size. `tests/parity/conv_check.py` records forward
and gradient differences (< 1e-5) on both backends. The large-batch anomaly is
recorded here as an open correctness issue, not silently ignored.

**Consequences.** CIFAR-10 loss curves, accuracy and final weights now match
PyTorch CUDA (`benchmarks/results/cifar_cnn_parity.json`). The intermittent
full-batch eval anomaly still needs a dedicated Vulkan synchronization/race
investigation; only the chunked path is claimed correct. No tolerance was
loosened.

---

## 2026-11-21 — Batched Vulkan submissions and timestamp queries (exp_003)

**Context.** Elementwise and GEMM ratios were 0.17–0.36 and 0.22 of PyTorch
CUDA. exp_001 removed allocation overhead; the remaining fixed cost was one
`vkQueueSubmit` + fence wait per operator. Single-op benchmarks could not show
the cost of a whole training step, so the gap needed to be measured directly.

**Options considered.** (1) Keep submit-per-op, (2) record many dispatches into
one command buffer and flush only on host readback or an explicit API call,
(3) go straight to timeline semaphores and multiple in-flight frames.

**Decision.** Option 2, driven by `experiments/exp_003/HYPOTHESIS.md`. Each
dispatch gets its own descriptor set from a rotating pool (descriptor contents
are host-written, so one shared set cannot represent a batch). A new public
`bt_vulkan_flush()` submits and waits; host readback also flushes. Vulkan
timestamp queries measure each dispatch when the device limit
`timestampComputeAndGraphics` is true, and `bt_vulkan_device_time_ms()` reports
the last batch. `BT_VK_NO_BATCH=1` restores the old path for A/B measurement.

**Consequences.** A 20-op pointwise chain is 2.17x faster (2.773 → 1.277 ms),
meeting the experiment's prediction; timestamps show 0.942 ms of GPU work in
that batch. Single-op wall time is unchanged because one op still means one
submit, and the headline ratios remain below target. All correctness checks
stayed green (CTest 1/1, parity 40/40, gradcheck and conv_check unchanged).

---

## 2026-11-21 — Timeline semaphore + first fused pointwise kernel (exp_004)

**Context.** exp_003 still waited on a fence at every flush, and pointwise
chains issued one dispatch per op. Both were candidates for the remaining gap.

**Options considered.** (1) Keep fence waits and separate pointwise kernels,
(2) signal a timeline semaphore on submit and add an explicit host wait, plus a
fused `relu(a+b)` shader, (3) implement a full graph-level fusion pass first.

**Decision.** Option 2, recorded in `experiments/exp_004/HYPOTHESIS.md`.
`bt_vulkan_flush()` submits asynchronously and signals the timeline; the new
`bt_vulkan_wait()` waits for the last value, collects timestamp results and
recycles descriptor/query slots. Upload/readback wait where the host needs the
data. A fused `bt_add_relu` kernel (GLSL + cpu_ref) replaces two dispatches with
one. A general fusion pass remains future work.

**Consequences.** Fusion is a clear win: 20 add+relu pairs drop from 2.365 ms
to 1.200 ms (1.97x). The async prediction was **not** met: the chain already
had a single submission, so timeline submission improved it only 1.277 → 1.251
ms (~2%). We kept it because it removes the CPU fence stall and exposes an
explicit completion point, but we do not claim a speedup from it. Parity grew
to 42/42 including exact fused cases; no tolerance was loosened. Performance
targets remain missed and are reported in `BENCHMARKS.md`.

---

## 2026-11-21 — GEMM ladder and size-based autotune (exp_008)

**Context.** `GAP_LEDGER.md` ranked GEMM first at 155.5 µs / ratio 0.29. The
tiled kernel reached 496 GFLOPS (9% of FP32 peak); PyTorch profile showed
`volta_sgemm_128x32_nn` plus a split-K reduction.

**Options considered.** (1) Keep the 16×16 tiled kernel, (2) register-blocked
128×128×8 with conflict-free shared memory, (3) add split-K, (4) autotune by
shape from measured timings.

**Decision.** All four variants are selectable with `BT_GEMM_VARIANT=0..3`;
the default autotunes by shape: tiled for ≤512, register-blocked above. Split-K
is kept but not selected automatically because it was slower (1.558 ms vs
1.211 ms at 1024²). Matmul push constants gained `k_start`/`k_len` so split-K
really partitions K. Scratch buffers use deferred free after the timeline wait.

**Consequences.** At 1024², 4.329 → 1.224 ms (3.54×), 496 → 1754 GFLOPS,
ratio 0.22 → 0.77. The ≥0.80 target is still missed by 0.03. Parity remains
42/42 and the unit test caught an earlier split-K bug that wrote 4× the correct
result. The autotune is a hard-coded threshold from these measurements, not a
runtime cache; both facts are recorded in `experiments/exp_008/FINDINGS.md`.

---

## 2026-11-21 — vec4 + double-buffered GEMM (exp_009)

**Context.** exp_008 missed the GEMM target by 0.03 and listed vec4 loads and
double buffering as the next rungs.

**Options considered.** (1) Stop tuning and move to elementwise, (2) add a
vec4/double-buffered specialization of the register-blocked kernel with a
stride check and safe fallback, (3) rewrite the GEMM in inline PTX (not
available for Vulkan).

**Decision.** Option 2, recorded in `experiments/exp_009/HYPOTHESIS.md`. The new
variant is selectable with `BT_GEMM_VARIANT=4` and is auto-selected for large
aligned matrices; misaligned inputs fall back to the scalar register-blocked
kernel with identical numerics.

**Consequences.** 1024² improves 1.224 → 1.178 ms and 1754 → 1823 GFLOPS
(ratio 0.77 → 0.78). The ≥1.08× prediction and the ≥0.80 target are **both
missed** and recorded as such in `experiments/exp_009/FINDINGS.md`. An initial
incorrect vec4 mapping produced up to 216 absolute error; it was caught by a
direct PyTorch comparison and fixed before any commit. Parity stays 42/42.

---

## 2026-11-21 — On-disk autotune cache; 16-deep K step rejected (exp_010)

**Context.** exp_009 left the GEMM target missed and listed larger K steps and
a runtime autotune cache as the next rungs.

**Options considered.** (1) Keep the hard-coded size threshold, (2) write an
on-disk per-shape cache from measured benchmarks, (3) add a 16-deep K-step
variant, (4) both 2 and 3.

**Decision.** Option 4, recorded in `experiments/exp_010/HYPOTHESIS.md`.
`benchmarks/ops/gemm.py` variants are measured by `gemm_tune.py`, which writes
`benchmarks/results/gemm_autotune.txt` (`M N K variant`). The C runtime reads
it on each matmul; a miss falls back to the measured size threshold. Variant 5
(16-deep K, 32 KB shared memory) is selectable but not auto-selected.

**Consequences.** The cache works: profiler traces show `gemm_tiled` at 256²
and `gemm_rb_vec4` at 1024². The k16 variant is **1.59× slower** than the
8-deep kernel (1.562 vs 0.985 ms in the tuning run) and is recorded as a
negative result. Final ratio is 0.69–0.78 depending on GPU clock
(1530–1950 MHz sampled), still below the 0.80 target. Two bugs found while
building the cache (static env caching, GPU name with spaces in the key) were
fixed before commit. Parity remains 42/42.

---

## 2026-11-21 — Clock logging, minimal barriers, async overlap (exp_011)

**Context.** Phase 5 remained: no clock normalization, one global barrier per
dispatch, and a host wait after every benchmark iteration.

**Options considered.** (1) Keep the global barrier and sync-per-iteration,
(2) add per-run clock logging, hazard-tracked buffer barriers, async host
overlap, and an on-disk pipeline cache.

**Decision.** Option 2, recorded in `experiments/exp_011/HYPOTHESIS.md`. No
kernel algorithm changed.

**Consequences.** Clock logging explains the GEMM ratio spread (0.71 at a fixed
1530 MHz). Hazard barriers reduce a 20-add chain from 20 global barriers to 19
buffer-specific ones — a true dependency chain still needs them, so the
material benefit is barrier scope, not count. Async batching gives 1.10×,
below the 1.2× prediction; host overhead is no longer dominant. Pipeline cache
persistence is blocked by driver 621.01 (`vkGetPipelineCacheData` crashes with
NULL pData and returns `VK_INCOMPLETE` up to 64 MiB), so only the validated
load path remains. All negative results are in `experiments/exp_011/FINDINGS.md`;
parity stays 42/42.

---

## 2026-11-21 — vec4 aligned elementwise fast path (exp_012)

**Context.** exp_011 showed elementwise wall ratios 0.34–0.54 and identified
memory-transaction inefficiency and runtime div/mod indexing as likely causes.
PyTorch's profiler showed `vectorized_elementwise_kernel<4>`.

**Options considered.** (1) Keep the scalar shader, (2) add a vec4 shader with
linear indexing enabled only for contiguous equal-shaped aligned tensors,
falling back to the scalar path otherwise.

**Decision.** Option 2, recorded in `experiments/exp_012/HYPOTHESIS.md`. The
fast path preserves the existing numeric semantics, including the fp64 division
intermediate and transcendental functions. Broadcast, strided, scalar, and
odd-sized cases keep the proven scalar kernel.

**Consequences.** Large-tensor device-time bandwidth is now 269–284 GB/s, and a
20-add chain reaches 293 GB/s = 96% of the measured DRAM ceiling. Wall ratios
only improve to 0.43–0.49 because ~36 µs/op submit+fence dominates. The ≥1.3×
speed prediction was not met overall (1.06–1.29×), recorded in
`experiments/exp_012/FINDINGS.md`. Parity stays 42/42. This locks in the
decision that further elementwise work should not chase bandwidth; the next
target is submission overhead.

---

## 2026-11-21 — Record-once capture and timeline replay (exp_013)

**Context.** exp_012 showed kernels at ~96% of DRAM ceiling while eager wall
ratios stayed 0.43–0.49; ~36 µs/op of submit+fence remained.

**Options considered.** (1) Accept the overhead, (2) add a dedicated reusable
command buffer that is recorded once and submitted many times with timeline
signals, keeping one host wait.

**Decision.** Option 2, recorded in `experiments/exp_013/HYPOTHESIS.md`. New API
`bt_vulkan_capture_begin/end/replay` flips recording into a reusable command
buffer, submits it N times, and uses the existing timeline wait as the single
completion point. Input lifetime during replay is documented publicly.

**Consequences.** A 20-add chain drops 1.080 → 0.899 ms/chain (1.20×), and
Bitsy replay versus PyTorch CUDA graph replay is **1.00×** — the first measured
performance-parity result, limited to captured execution. The ≥1.5× prediction
was not met because GPU work is now the floor (recorded as a negative result).
Parity stays 42/42 and captured kernels are bit-identical. Default eager ratios
are unchanged; the next step is to make deferred submission the default for
chains and measure end-to-end training steps.

---

## 2026-11-21 — Deferred transfers as the default (exp_014)

**Context.** exp_013's 1.00× result required explicit capture. The ordinary API
still paid two forced flush+waits per training step from `from_numpy()`, and a
fresh staging buffer allocation on every upload/readback. No end-to-end
training wall-time measurement existed.

**Options considered.** (1) Keep synchronous uploads, (2) pool host-visible
staging buffers, enqueue the copy into the current command buffer, and release
the slot only after the submitted timeline value is waited on — making the
default API stream-like (block only on readback).

**Decision.** Option 2, recorded in `experiments/exp_014/HYPOTHESIS.md`. Staging
slots transition used → submitted (on flush) → pooled (on wait), which prevents
reuse while the GPU can still access a copy. `BITSY_DEFERRED_UPLOAD=0` keeps the
old path as an A/B escape hatch. The default was flipped to deferred because
the speed prediction was met.

**Consequences.** Training-step wall time drops 9.575 → 3.220 ms/step at matched
1530 MHz (2.97×), with 0 staging allocations during 200 timed steps. The ≥0.85
end-to-end throughput target is **missed** (0.34–0.37) — the remainder is
per-op Python/ctypes dispatch and autograd construction, recorded as a negative
result. Numeric parity holds: 42/42, MNIST report identical, CIFAR Bitsy losses
bit-identical between modes; the small torch-side variation is cuDNN benchmark
nondeterminism. Next: full-step capture/replay against PyTorch CUDA graphs.

---

## 2026-11-21 — Full-step capture with readback chunking (exp_015)

**Context.** exp_014 left end-to-end throughput at 0.34–0.37, dominated by
per-op dispatch. A pointwise capture had already matched PyTorch graphs 1.00×
(exp_013), but a whole training step had never been captured.

**Options considered.** (1) Declare full-step capture impossible because the
interim autograd reads bias gradients back to the CPU during backward,
(2) split the capture into command-buffer chunks at each readback: submit+wait
the chunk so the readback is valid, then continue recording.

**Decision.** Option 2, recorded in `experiments/exp_015/HYPOTHESIS.md`. Chunk
submission must **not** recycle staging, scratch, descriptor or timestamp
resources, because replayed chunks reference them; a version that did recycle
produced bit-wrong replays and was rejected. Replaying after the final wait is
documented as invalid without re-capture.

**Consequences.** A captured step is bit-identical to eager (abs diff 0.0) and
runs in 0.294–0.355 ms/step versus 3.220 ms eager (~9–11×). Against PyTorch
CUDA graphs (0.067 ms/step) the ratio is only **0.19–0.23**, missing both
end-to-end targets; recorded as a negative result. The remaining causes are the
extra submit per replay imposed by the CPU readback and the many small
dispatches from the Python autograd. Next: device-side reductions to remove the
readback and obtain a single-chunk capture.

---

## 2026-11-21 — Device-side unbroadcast reductions (exp_016)

**Context.** exp_015's capture needed two chunks because `_unbroadcast()` read
bias gradients back to the CPU mid-backward, forcing an extra submit per replay.

**Options considered.** (1) Keep the chunked capture, (2) replace the 2D
unbroadcast CPU roundtrip with device matmul/view reductions, keeping a CPU
fallback for >2D.

**Decision.** Option 2, recorded in `experiments/exp_016/HYPOTHESIS.md`. No
kernel, tolerance or benchmark-method change. The 1.07e-7 relative fp32 drift
from changing reduction order is measured and accepted inside the 1e-5 gate.

**Consequences.** Capture-phase submit count drops 2 → 1, confirming a
single-chunk MLP capture. The eager step improves 3.220 → 2.626–2.911 ms
(1.1–1.2×; ratio 0.37–0.41), but **captured graph throughput does not improve**
(0.20–0.23 vs PyTorch CUDA graph); the ≥1.3× / ≥0.28 graph prediction is
recorded as a miss. The dominant remaining cost is now clearly the ~66 small
dispatches and barriers per step, not submit count. Next: fuse backward/
optimizer chains to cut dispatch count. Parity stays 42/42; MNIST identical,
CIFAR unchanged.

---

## 2026-11-21 — Profiler-guided dispatch fusion (exp_017)

**Context.** exp_016 identified ~66 Vulkan dispatches per step as the limiting
cost. A profile showed 16 of them were materialised transposes for matmul
backward, and the SGD update used two ops plus an lr upload per parameter.

**Options considered.** (1) Accept the dispatch count, (2) turn transposes into
zero-dispatch strided views (GEMM already consumes strides) and fuse the SGD
update into a dedicated shader with lr as a push constant.

**Decision.** Option 2, recorded in `experiments/exp_017/HYPOTHESIS.md`. The
fused kernel keeps the fp32 multiply-then-subtract order; the CPU-only build
uses the existing ops with the same order. No tolerance change.

**Consequences.** Dispatch count 66 → 46 (30.3%, target met) and barriers
66 → 52. Captured step 0.296–0.334 → **0.257 ms** (1.15–1.30×, centrally
~1.23×) and graph ratio 0.20–0.23 → 0.259; the ≥1.3× / ≥0.28 stretch goals
were not reliably met and are recorded as misses. Eager step 2.626 → 2.46 ms,
ratio 0.41–0.47 — the best eager result so far. Parity holds (42/42, CTest
both builds, MNIST/CIFAR drift < 2e-7). The remaining ~20 tiled-GEMM
dispatches (many tiny reduction matmuls) are the next target.
