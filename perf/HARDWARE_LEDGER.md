# TOGISHI Phase 1 — Hardware ledger

**Status:** calibration complete. This file was created **before** any Hagane
source change; Phase 1 benchmarks are standalone C + Vulkan under `perf/` and
do not compile or link Hagane code.

**Hypothesis / method:** `experiments/exp_006/HYPOTHESIS.md`
**Findings:** `experiments/exp_006/FINDINGS.md`
**Raw numbers:** `perf/calibration.json`

---

## 1. Machine and environment (measured, not assumed)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER (TU116) | `nvidia-smi` |
| Vulkan device | discrete GPU, driver "NVIDIA" | `vulkaninfo --summary` |
| Vulkan instance / device API | 1.4.363 / 1.4.357 | `vulkaninfo`, `perf_calibrate` |
| NVIDIA driver | 621.01 | `nvidia-smi` |
| SM clock | 1530 MHz idle → **1950 MHz** loaded | `nvidia-smi` before/after |
| Memory clock | 7000 MHz (14 Gbps effective) | `nvidia-smi` |
| Memory bus | 192-bit | public spec |
| VRAM | 6144 MiB | `nvidia-smi` |
| Power / temp | 42 W / 56 °C → 62 W / 58 °C | `nvidia-smi` |
| Toolchain | MSVC 19.51, CMake 4.3, Ninja 1.13 | local build |

Theoretical DRAM bandwidth = 14 Gbps × 192 bits / 8 = **336 GB/s**.
FP32 peak at the measured 1950 MHz clock = 1408 × 2 × 1.95 GHz = **5.49 TFLOP/s**.

---

## 2. Device limits (measured)

| Limit | Value |
|---|---|
| `deviceName` | NVIDIA GeForce GTX 1660 SUPER |
| `timestampPeriod` | 1.000 ns |
| `timestampComputeAndGraphics` | true |
| `maxComputeWorkGroupInvocations` | 1024 |
| `maxComputeSharedMemorySize` | 49152 B |
| `maxPushConstantsSize` | 256 B |
| shader workgroup size used | 256 |
| clock during run | 1530 → 1950 MHz |

Extension/subgroup queries are deferred to Phase 2; none were needed here.

---

## 3. Calibration results

All device timings use GPU timestamp queries. The copy test uses 1 GiB
pseudo-random-filled buffers and one pass per measurement.

### 3.1 DRAM / copy

| Kernel | Bytes moved | Median ms | Achieved GB/s | % of 336 GB/s | Status |
|---|---:|---:|---:|---:|---|
| vec4 copy (float4) | 2 GiB | 7.28 | **295.1** | 87.8% | trustworthy; measured BW ceiling |
| vec4x4 copy | 2 GiB | 1.98 | 1083.2 | 322% | **invalid** (>3× physical); compression/write-combining artifact, excluded |

DRAM streaming read (512 MiB): 303.3 GB/s — consistent with the vec4 copy.
**Measured DRAM ceiling = ~295–303 GB/s.**

### 3.2 Compute

| Kernel | Work | Median ms | Achieved GFLOP/s | % of 5.49 TFLOP/s |
|---|---|---:|---:|---:|
| FP32 FMA (naive) | 65k × 1024 FMA | 0.145 | 928 | 17% |
| FP32 FMA (8 acc, 524k lanes) | same FMA count | 0.262 | 4100 | 75% |
| FP32 FMA (16 acc, 65k lanes) | same FMA count | 0.038 | 3537 | 64% |
| best across three runs | — | — | 4100–4350 | 75–79% |
| packed f16vec2 FMA | shader compiled | not measured | — | Phase 2 |

### 3.3 Shared memory / L2 / bank conflicts

| Kernel | Pattern | Median ms | Relative | Note |
|---|---|---:|---:|---|
| shared write+read | stride 1 | 0.00472 | 1.00× | conflict-free baseline |
| shared write+read | stride 2 | 0.00477 | 1.01× | no measurable 2-way penalty |
| shared write+read | stride 32 | 0.01171 | **2.48×** | 32-way conflict confirmed |
| "L2" read | 1 MiB buffer | 0.0062 | — | 169 GB/s, too little work; unreliable |
| DRAM read | 512 MiB buffer | 0.2213 | — | 303 GB/s, agrees with copy |

### 3.4 Launch / synchronization / allocation

| Operation | Median µs | Note |
|---|---:|---|
| empty dispatch + barrier (GPU) | 3.80 | 2000 dispatches |
| barrier recording (CPU) | 0.27 | 20k calls |
| dispatch recording (CPU) | 0.32 | 20k calls |
| submit + fence wait | 35.2 | 2000 iterations |
| timeline signal + wait | 37.4 | 2000 iterations |
| `vkAllocateMemory` + free | 390 | 4 MiB device-local buffer |
| staging ring reuse | not measured | Phase 4 |

### 3.5 PCIe

| Direction | Size | Median ms | Achieved GB/s |
|---|---:|---:|---:|
| H2D + D2H combined | 64 MiB each | 10.62 | **12.6** combined |

---

## 4. Gate decision

| Gate | Target | Measured | Pass? |
|---|---:|---:|---|
| vec4 copy vs theoretical BW | ≥ 90% | 87.8% (295 vs 336 GB/s) | **no**, short by 2.2 points |
| bare FMA vs clock-normalized peak | ≥ 95% | 75–79% (4.1–4.35 vs 5.49 TFLOP/s) | **no** |
| measurement method trustworthy | yes | copy4/L2 results excluded, remainder sound | partial |

**Gates are not met — do not optimize Hagane code yet.** Next method work:
`VK_KHR_pipeline_executable_properties` / Nsight Compute for FFMA issue rate,
register pressure and occupancy, then re-run the FMA calibration. The copy miss
is recorded rather than rounded up.

---

## 5. Regenerate

```powershell
cmake -S perf -B build-perf -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-perf
cd build-perf
.\perf_calibrate.exe --json ..\perf\calibration.json
```

`--json` is a plain C argv flag in this standalone probe; the no-argparse rule
applies to Python configuration in the framework.
