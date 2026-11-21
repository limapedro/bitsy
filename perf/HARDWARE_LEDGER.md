# TOGISHI Phase 1 — Hardware ledger

**Status:** calibration complete; both gates pass after method fixes.
Created **before** any Hagane source change; `perf/` is standalone C + Vulkan
and does not compile or link Hagane code.

**Hypothesis:** `experiments/exp_006/HYPOTHESIS.md`
**Findings:** `experiments/exp_006/FINDINGS.md`
**Raw numbers:** `perf/calibration.json`

---

## 1. Machine and environment

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER (TU116) | `nvidia-smi` |
| Vulkan instance / device API | 1.4.363 / 1.4.357 | `vulkaninfo`, probe |
| NVIDIA driver | 621.01 | `nvidia-smi` |
| SM clock | 1530 MHz idle, 1920–1950 MHz loaded | sampled per kernel |
| Memory clock | 7000 MHz (14 Gbps effective) | `nvidia-smi` |
| Memory bus | 192-bit | public spec |
| VRAM | 6144 MiB | `nvidia-smi` |
| Power / temp | 42 W / 56 °C idle → 55–74 W / 59 °C loaded | `nvidia-smi` |
| Toolchain | MSVC 19.51, CMake 4.3, Ninja 1.13 | local build |

Theoretical DRAM = 14 Gbps × 192 bits / 8 = **336 GB/s**.
FP32 peak at 1950 MHz = 1408 × 2 × 1.95 GHz = **5.49 TFLOP/s**.

---

## 2. Device limits (measured)

| Limit | Value |
|---|---|
| `timestampPeriod` | 1.000 ns |
| `timestampComputeAndGraphics` | true |
| `maxComputeWorkGroupInvocations` | 1024 |
| `maxComputeSharedMemorySize` | 49152 B |
| `maxPushConstantsSize` | 256 B |
| workgroup size used | 256 |
| `VK_KHR_pipeline_executable_properties` | supported, enabled |
| `VK_EXT_memory_budget` | supported (not yet used) |
| `VK_KHR_shader_float16_int8` | supported (packed f16 measurement pending) |

---

## 3. Calibration results (final method)

### 3.1 DRAM / copy

1 GiB buffers, full-entropy pseudo-random fill, one pass, timestamp queries.

| Kernel | Bytes | Time | Achieved | Ceiling | Ratio | Status |
|---|---:|---:|---:|---:|---:|---|
| vec4 copy | 2 GiB | ~7.3 ms | 284.9 GB/s | 304.4 GB/s measured DRAM read | **93.6%** | trustworthy |
| vec4x4 copy | 2 GiB | ~2.0 ms | 1094 GB/s | 304 GB/s | >100% | **excluded**: destination never read; compression/write-combining artifact |

DRAM streaming read (512 MiB): **304.5 GB/s**. Measured DRAM ceiling =
**~300–305 GB/s**.

### 3.2 Compute

| Kernel | Time | Achieved | % of clock-normalized peak |
|---|---:|---:|---:|
| FP32 FMA naive (8 dependent FMAs) | 0.146 ms | 928 GFLOP/s | 17% |
| FP32 FMA 8 accumulators, looped | 0.269 ms | 3993 GFLOP/s | ~73% |
| FP32 FMA 16 accumulators | 0.030 ms | 4438 GFLOP/s | ~81% |
| **FP32 FMA 8 acc, 32-way unrolled** | 0.205 ms | **5233 GFLOP/s** | **95.3%** @1950 MHz |
| same kernel, 3 runs | 0.205–0.211 ms | 5091–5233 GFLOP/s | 93.1–95.3% |
| packed f16vec2 FMA | compiled | not measured | pending |

`VK_KHR_pipeline_executable_properties` for the winning kernel: **16 registers
per thread**, subgroupSize 32, 0 shared memory, no spills reported. Register
pressure was never the limit; loop overhead was.

Nsight Compute 2026.2.1 reported `No kernels were profiled` for this Vulkan
executable (CUDA-only kernel profiling in this release), so shader-level
evidence comes from pipeline executable properties above. Nsight Graphics CLI
was not available.

### 3.3 Shared memory / bank conflicts

| Pattern | Time | Relative |
|---|---:|---:|
| stride 1 | 4.0 µs | 1.00× |
| stride 2 | 3.9 µs | 0.98× |
| stride 32 | 10.8 µs | **2.7×** |

### 3.4 Launch / synchronization / allocation

| Operation | Median |
|---|---:|
| empty dispatch + barrier (GPU) | 3.1 µs |
| barrier recording (CPU) | 0.23 µs |
| dispatch recording (CPU) | 0.28 µs |
| submit + fence wait | 35.9 µs |
| timeline signal + wait | 36.2 µs |
| `vkAllocateMemory` + free (4 MiB) | 363 µs |
| staging ring reuse | not measured (Phase 4) |

### 3.5 PCIe

H2D + D2H of 64 MiB each: **12.6 GB/s combined**.

---

## 4. Gate decision

| Gate | Target | Measured | Pass? |
|---|---:|---:|---|
| copy vs measured DRAM BW | ≥ 90% | 93.6% | **yes** |
| FMA vs clock-normalized peak | ~95% | 95.3% best (93.1–95.3% spread) | **yes** |
| method trustworthy | yes | invalid copy4/L2 excluded | **yes** |

Phase 1 is complete. Framework code remains untouched. Next: Phase 2 profiler
and `GAP_LEDGER.md`, with a new `HYPOTHESIS.md`.

---

## 5. Regenerate

```powershell
cmake -S perf -B build-perf -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-perf
cd build-perf
.\perf_calibrate.exe --json ..\perf\calibration.json
```
