# exp_040 — FINDINGS (PR #7: Vulkan kernel throughput + DEVICE_LOCAL memory)

Status: PR #7 of the v0.2 rewrite. Builds on PR #6 (download bottleneck fixed).
Honest goal: attack the remaining **3.5× per-step gap** vs PyTorch CUDA.

## Measured starting point (PR #6)
```
replay submit-only 0.468 ms   (host submit + kernel + wait)
kernel ~28 GB/s effective     (vs 336 GB/s peak on GTX 1660 Super)
per-step gap 3.5x vs torch (bitsy 0.65 ms/step vs torch 0.19 ms/step)
```

## Root cause found by reading the allocator (not guessing)
All device buffers were allocated as **HOST_VISIBLE|HOST_COHERENT** (system memory
over PCIe, ~30 GB/s). The GPU kernel read its inputs over PCIe — that is why the
kernel was 12× off peak. Two independent bottlenecks were then measured:

| Lever | Measurement | Fix |
|---|---|---|
| Input/output memory in PCIe system RAM | generic kernel 0.46 ms/1M | DEVICE_LOCAL buffers + staging upload/download |
| Interpreter reads ~96 B/element of op-codes from global params; dynamic stack indexing | DEVICE_LOCAL generic still 0.29 ms/1M | specialized straight-line kernel: op-codes baked into 25 Vulkan specialization constants |

### Additional debugging finding (important)
An intermediate attempt (DEVICE_LOCAL output + HOST_COHERENT-only staging) made
download **11 ms** — NVIDIA HOST_COHERENT system memory is write-combining and
peaks ~0.35 GB/s on CPU reads. Switched staging to **HOST_CACHED** with
flush/invalidate → download back to ~0.85 ms. This is the same class of bug PR #6
already found for the direct output buffer; confirmed again with a controlled
measurement (`BTSY_NOCOPY` diagnostic showed 4 MB memcpy = 11 ms vs 1 ms).

## What landed
- `mem_type()` helper + `vkbuf()` now allocates DEVICE_LOCAL memory; uploads go
  through a persistent HOST_CACHED staging buffer with `vkCmdCopyBuffer`.
- Persistent staging buffer + persistent transfer command buffer/fence (avoids
  per-read `vkAllocateMemory`/`vkMapMemory` churn).
- `fused_sp.comp`: specialized fused kernel. The op-program is translated into
  25 specialization constants (op/operand A/operand B per step, max 8 steps),
  so the compiler emits straight-line register code. A translator in `rt.c`
  maps generic stack slots → 8 input slots / 8 instruction-result registers;
  complex programs (WHERE) fall back to the generic interpreter.
- Pipelines cached by program hash (`sp_pipeline`, up to 256 variants).
- `mem_type` excludes BAR1-mapped DEVICE_LOCAL for staging; HOST_CACHED selected.
- `np.zeros` → `np.empty` in the Python `get()` (rt_to_host overwrites all elements).
- Fixed a GELUB (op 14) unary/binary mismatch found by the gradcheck gate: the
  specialized shader must treat it as binary like the generic one.

## Honest results (after; median, warmup, spread, clock logged)
```
$ python benchmarks/compact/replay_bench.py   (GTX 1660 SUPER, 5 runs)
  [PASS] replay / new-data / lazy-read diff = 1.9e-6  (tol 1e-4)
  single-step forced download: replay 0.81-1.05 ms (3.0-4.1x vs eager),
                               torch 0.59-0.90 ms  -> replay ~= torch on this metric
  loop (10 GPU steps + read): bitsy 0.27-0.41 ms/step, torch 0.16-0.19 ms/step
                              -> per-step gap 1.5-1.9x (stored JSON: 1.49x)
  clock 1530->1905/1950 MHz (not pinned; the spread is honest)

$ python benchmarks/compact/profile_replay.py
  replay submit-only 0.114 ms       (was 0.468 ms in PR #6)
  replay full+download 0.967 ms     (download ~0.853 ms)
  torch GPU-only 0.032 ms; torch+download 0.973 ms
  -> single-step forced download is now at PARITY with torch+download
```
Isolated contributions (submit-only @ 1M elements):
- DEVICE_LOCAL only (generic kernel): 0.46 → 0.29 ms
- + specialized straight-line kernel: 0.29 → 0.10 ms (2.8×)

### Gap trajectory
`194× (PR #5) → 3.5× (PR #6) → 1.5-1.9× (PR #7, median ~1.7×)` on the fair
per-step metric.

## Gates (all green)
op parity 27/27, gradcheck 25/25, model 10/10, legacy 48/48, ctest 3/3.
Compact core **1,961 lines** (budget 4096, 2135 spare).

## Honesty / what is NOT in PR #7
- **Overall speed parity still NOT met**: 1.7× per-step gap remains; no ≥1.0×
  claim for the loop. The single-step forced-download metric is at parity, but
  it is a narrow case.
- Remaining gap is ~0.11 ms/step host submit + ~0.85 ms download (amortized
  0.085 ms) + Python `numpy()` conversion. Closing it further needs batched
  submissions, pinned-memory zero-copy reads, or a dedicated transfer queue.
- GEMM tile autotune, shared-memory tiling, subgroup reductions NOT done —
  this PR targeted the fused pointwise kernel and memory placement.
- Clock drift 1530→1905 MHz during PyTorch runs is logged; machine not pinned.
- Numerics changed from exactly 0.0 to 1.9e-6 (still << 1e-4 tol) because the
  specialized kernel evaluates left-to-right in a different order.
- Legacy core (5710 lines) still retained.
