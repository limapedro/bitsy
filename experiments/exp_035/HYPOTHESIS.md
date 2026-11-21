# exp_035 — evidence-backed performance ceiling assessment

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 2 (reporting)
**Allowed by:** the exp_034 next-step instruction

## Prerequisite

After 34 experiments: the GEMM ladder is exhausted (every rung measured),
elementwise kernels reach 0.88–1.03× PyTorch CUDA graphs when captured (so the
eager gap is host launch), per-op fusions are neutral or broke training parity
and were reverted, and the only met target is eager end-to-end (1.16–1.24).
The campaign rule requires honest reporting of misses.

## Hypothesis

The remaining gaps are bounded by mechanisms that have each been measured, and
no remaining campaign item offers >10% upside on the unmet targets. A
consolidated ceiling document should classify every gap with its evidence and
state what would actually be required to go further.

## Predictions

- A final consistent measurement set reproduces the reported ratios within
  run variance (elementwise ~0.4–0.5 eager, GEMM ~0.7, captured graph ~0.5,
  eager end-to-end ≥0.85).
- Each unmet target maps to a measured limiting mechanism, not an untested idea.
- The conclusion is recorded in `GAP_LEDGER.md` and a new
  `docs/PERFORMANCE_CEILING.md`, with no performance claim beyond evidence.

## Measurement

1. Snapshot current reports; run the full benchmark suite once for a final
   consistent set (elementwise, elementwise_graph, gemm, graph_step, train_step).
2. Compile the experiment outcome table and per-class mechanism classification.
3. Write `docs/PERFORMANCE_CEILING.md`, update GAP_LEDGER.md/BENCHMARKS.md.
4. FINDINGS with all numbers; no kernel change in this experiment.

## Honesty

The conclusion is conditional: it applies to this GPU/driver/backend pairing and
the current interim Python autograd. It does not claim a universal limit.
