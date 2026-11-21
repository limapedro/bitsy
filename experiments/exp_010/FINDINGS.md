# exp_010 — Findings: 16-deep K step and on-disk autotune cache

**Date:** 2026-11-21
**Scope:** continuation of the GEMM ladder. Parity remains 42/42, CTest 1/1,
matmul max error 2.1e-4 at 1024² versus PyTorch CUDA (within rtol/atol 1e-5).

## Implemented

- `BT_GEMM_VARIANT=5`: `gemm_rb_vec4_k16.comp`, 16-deep K step, double-buffered
  shared tiles (`As/Bs[2][16][128]`, 32 KB), vec4 loads, same 128×128 tile and
  8×8 thread tile. Halves barrier frequency versus the 8-deep variant.
- On-disk autotune cache: C reads `benchmarks/results/gemm_autotune.txt`
  (`M N K variant`) at each matmul; `benchmarks/ops/gemm_tune.py` measures all
  variants and writes the file. Cache selection verified with `hg.profiler`:
  the trace for 256² and 1024² showed `['gemm_tiled', 'gemm_rb_vec4']`.
- Fixed two bugs found during the experiment: the variant was cached in a
  static on first call (so a tuning loop stayed on variant 0), and the cache
  key originally included a GPU name containing spaces (unparseable by
  `fscanf`). Both corrected before commit.

## Measurements

Tuning run (same process, all variants, `experiments/exp_010/gemm_tune.json`):

| n | variant | ms | ratio |
|---:|---|---:|---:|
| 256 | 1 tiled | 0.156 | 0.40 |
| 256 | 2 rb | 0.156 | 0.41 |
| 1024 | 2 rb | 1.008 | 0.70 |
| 1024 | 4 rb vec4 | **0.985** | 0.77 |
| 1024 | 5 rb vec4 k16 | 1.562 | 0.50 |

**16-deep K step is a negative result**: 1.562 ms versus 0.985 ms for the
8-deep kernel (1.59× slower), likely from shared-memory capacity and reduced
occupancy (32 KB versus 16 KB). It stays selectable but is never chosen by the
cache.

Final autotune benchmark (cache selects tiled at 256², vec4 at 1024²):
```
n=1024  bitsy 1.289 ms  torch 0.893 ms  ratio 0.69  GFLOPS 1666
```
Sampled GPU clock during this run: 1530 MHz for most of it, briefly 1650 MHz.
Earlier exp_009 runs reached 1950 MHz and 1.178 ms / ratio 0.78. The
difference is dominated by clock variance, not a code regression; both numbers
are reported rather than cherry-picking the faster run.

## Predictions

- "≥1.05× faster than 8-deep vec4": **not met** (1.59× slower). Negative result.
- "ratio ≥0.80": **not met** (0.69–0.78 across runs).
- "cache selects measured winner": **met** (profiler trace confirms).
- "parity remains 42/42": **met**.

## Interpretation / next

The cache is useful infrastructure even though this rung failed; it prevents
the slow k16 kernel from being selected. The GEMM target remains missed and the
clock variance means future comparisons must log clocks per run (already done
in `perf/calibrate`, now needed in `gemm.py`). Next: hook on-disk pipeline cache
and per-run clock logging, then investigate submission/fence overhead rather
than adding more tile variants.
