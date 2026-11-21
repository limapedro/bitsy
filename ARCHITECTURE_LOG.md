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
