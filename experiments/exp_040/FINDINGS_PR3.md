# exp_040 — FINDINGS (PR #3: full autograd coverage + gradcheck)

Status: PR #3 of the v0.2 rewrite. Builds on PR #1 (foundation) and PR #2 (GEMM/reduce).

## What landed
- Correct backward rules for the full differentiable primitive set:
  unary (exp/log/sqrt/tanh/sigmoid/relu/gelu/abs/neg), binary
  (add/sub/mul/div/pow/max/min), reduce (**sum/mean/max/min**), matmul with
  transpose flags + bias, transpose, reshape, where.
- Fixed mean-grad (÷N), max/min-grad (argmax mask), and matmul-with-transpose grad
  (correct opA^T/opB^T handling + bias reduce).
- Scalar tensors are now 1-element (ndim≥1) — PyTorch/tinygrad convention —
  fixing a crash when reduces chain to a scalar.
- `benchmarks/compact/gradcheck_vs_torch.py` — gradcheck vs real PyTorch CUDA.
- Full ctypes bindings for transpose/reshape.
- C unit-test grad cases in `tests/unit/test_rt2.c` (CPU + Vulkan, CI-friendly).

## Evidence (Windows MSVC + Vulkan 1.4, GTX 1660 Super)
```
$ python benchmarks/compact/gradcheck_vs_torch.py
  [PASS] grad_exp .. grad_minimum            (16 cases, <=2.4e-7)
  [PASS] grad_reduce_sum/mean/max/min        (0.0)
  [PASS] grad_matmul_A / transpose / reshape / where / gemm_bias  (0.0)
  25/25 gradcheck cases passed
$ ctest --test-dir build-rtvk --output-on-failure
  3/3 Test #3 unit_compact_core2 ... Passed (grad sum/mean/max/chain on CPU+Vulkan)
  100% tests passed, 0 tests failed out of 3
$ python tools/count_core_lines.py
  compact=1459 (budget 4096, 2637 spare)
$ python tests/parity/run.py
  48/48 parity cases passed
```

## Bugs found and fixed (via gradcheck — the value of measuring)
1. **Chained reduces crashed** (access violation): reducing to a 0-d tensor broke
   the fused coord decomposition. Fixed by keeping scalars as 1-element tensors.
2. **reduce_mean grad** returned g instead of g/N — fixed with count division.
3. **reduce_max/min grad** routed gradient to every element — fixed with an
   argmax/argmin mask (`eq(input, broadcast(reduce_output))`).
4. **matmul-with-transpose grad** transposed the wrong operand (K mismatch crash)
   — re-derived dY/dA' = g@B'^T, dY/dB' = A'^T@g with correct un-transpose.
5. Harness asymmetry (RT used 3 chained reduces, torch used 1) — fixed so both
   compute identical graphs.

## Honesty / what is NOT in PR #3
- Autograd now covers the differentiable primitives; no second-order (gradgrad)
  support, no hooks, no `create_graph`.
- No speed numbers yet (capture/replay + perf tuning are later PRs).
- Full 285-function Python map still not implemented; the compact core primitives
  are now numerically complete with grads, ready for Python composition.
- Legacy core (5710 lines) retained as reference; budget enforces the 1459-line
  compact core.
