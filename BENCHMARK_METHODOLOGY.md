# Benchmark & parity methodology — how we measure "as fast as PyTorch"

The objective is speed parity with PyTorch on this GPU (GTX 1660 Super) in fp32.
We only trust numbers produced by scripts that run **real PyTorch** as the
oracle and follow the rules below. No estimate counts as evidence.

## 1. Numerics (correctness)

- Oracle: `torch` CUDA fp32 on the same machine. Compare Bitsy output (fp32)
  against PyTorch with `max_abs_diff` (and relative where noted).
- Tolerances (never relaxed to hide a bug):
  - EXACT (bit-identical) for add/sub/mul/div/relu.
  - rtol/atol 1e-5 for transcendental and matmul-class ops.
  - 1e-3 for accumulated model loss (GPT-2).
- Every backward/gradient is checked too, not just forward: gradcheck + a
  training-scale gate (the small gate historically missed large-batch bugs).
- Report the worst-case diff, the shape, and the dtype in JSON.

## 2. Speed (performance)

For each op, each model step, and the full model:

- **Warmup** N iterations before timing (default 3–10).
- **Correct sync**: Bitsy `bt_flush()` + `bt_wait()` (or timeline wait);
  PyTorch `torch.cuda.synchronize()`. Never time an unsynced async queue.
- **Median + spread**: median of M iterations plus min/max (or stdev), not a
  single sample.
- **GPU clock** sampled before/after with `nvidia-smi` (`common/clock.py`) so a
  ratio can be interpreted; both frameworks measured in the same run.
- Metric: `ratio = pytorch_ms / bitsy_ms`. **1.00 = as fast as PyTorch.**
  Ratios < 1 are misses and are printed as such (no hiding).
- Dataclass configs, no argparse (project convention).
- Output: `benchmarks/results/<name>.json` with environment, config, diffs,
  timings, spread, clock and ratio.

## 3. Two granularities — full model AND each step

- **Full-model**: end-to-end forward (+ backward + optimizer step for training)
  wall time.
- **Per-step / per-op**: each layer's forward and backward timed individually
  (e.g. embedding, attention QKV, GEMM MLP head, layernorm, loss). This localizes
  the slow stage; a full-model ratio alone cannot tell you *why* it is slow.
- Both feed `assets/progress.svg`: full-model ratios at the top, per-step ratios
  in the grid below, each box colored by how close to 1.0 it is.

## 4. Reproducibility

- Every result committed as JSON alongside the script that produced it.
- Same seed for both frameworks; weights copied NumPy → PyTorch and Bitsy so
  inputs are identical.
- CI runs the parity suite on Windows + Linux (CPU PyTorch there; CUDA numbers
  are from the developer machine and labeled as such).
- Regenerate the image: `python tools/gen_progress.py`.

## 5. Honesty rules (inherited from the accepted-ceiling campaign)

- State misses plainly; never claim performance parity that was not measured.
- A documented ceiling is acceptable only with per-gap mechanism analysis.
- "100% parity" = every *implemented* function matches within tolerance on
  tested shapes; the full PyTorch API is out of scope and stated as such.
