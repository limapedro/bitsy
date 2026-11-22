# Bitsy

A small fp32 deep-learning framework with a C11 core and a Vulkan compute
backend. The goal is **PyTorch parity**: correct numerics and comparable speed
on this GPU (GTX 1660 Super), following tinygrad's "fewest lines, maximum
performance" philosophy.

> **This branch (`docs/framework-study-and-roadmap`) is documentation only —
> no framework code.** It records the framework research, the project skeleton,
> the feature parity matrix, the line budget, and a progress tracker that will
> be driven by measurements once implementation begins.

![Bitsy progress toward PyTorch parity](assets/progress.svg)

*The image above tracks two axes: numeric parity (max-abs-diff within tolerance
vs real PyTorch CUDA, fp32) and speed parity (`torch_ms / bitsy_ms`, where 1.00
means as fast as PyTorch). The numbers shown are the inherited baseline from
the previous architecture; this branch is the plan to improve them.*

The dashboard is authored as an **HTML page** — open
[`assets/progress.html`](assets/progress.html) in a browser for the full view:
an overall health score with sub-scores, a roadmap timeline, speed bars with a
1.00 target marker, parity micro-grid tiles, the "still needed" backlog and a
line-budget bar. The layout template lives in
[`assets/progress.template.html`](assets/progress.template.html). The GitHub
image above is a native SVG exported from the *same* `docs/progress.json` data,
so the HTML and SVG never drift. See
[`docs/research/PROGRESS_IMAGE_BREAKDOWN.md`](docs/research/PROGRESS_IMAGE_BREAKDOWN.md)
for a breakdown of the reference image and how each element maps here.

## What is here

| File | Purpose |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Phased plan for the v0.2 skeleton (C+Vulkan, Python, benchmarks) |
| [SKELETON.md](SKELETON.md) | Directory/module layout for the implementation |
| [docs/roadmap/C_VULKAN_CORE.md](docs/roadmap/C_VULKAN_CORE.md) | Every low-level C+Vulkan function, kernel, and its parity case |
| [docs/roadmap/PYTHON_LIBRARY.md](docs/roadmap/PYTHON_LIBRARY.md) | Every Python API, its kernel lowering, and backward rules |
| [docs/roadmap/VULKAN_PERFORMANCE.md](docs/roadmap/VULKAN_PERFORMANCE.md) | Research-backed Vulkan→CUDA speed techniques + benchmark plan |
| [FEATURE_PARITY_MATRIX.md](FEATURE_PARITY_MATRIX.md) | Per-op / per-model numeric + speed status vs PyTorch |
| [LINE_BUDGET.md](LINE_BUDGET.md) | ≤ 4096-line C core rule and allocation (stretch 2048) |
| [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) | How numerics and speed are measured against PyTorch |
| [docs/research/FRAMEWORKS.md](docs/research/FRAMEWORKS.md) | High-level study of tinygrad, PyTorch, JAX |
| [docs/research/PROGRESS_IMAGE_BREAKDOWN.md](docs/research/PROGRESS_IMAGE_BREAKDOWN.md) | Breakdown of the reference image and its mapping to Bitsy |
| [assets/progress.template.html](assets/progress.template.html) | Authored HTML/CSS dashboard template (design reference) |
| [assets/progress.html](assets/progress.html) | Generated dashboard page (open in a browser) |
| [docs/progress.json](docs/progress.json) | Single source of truth for all dashboard numbers |
| [tools/gen_progress.py](tools/gen_progress.py) | Injects data into the template → HTML, and exports native SVG |
| [tools/count_lines.py](tools/count_lines.py) | Enforces the C core line budget |

## Design in one paragraph

PyTorch is the **numeric oracle**; tinygrad is the **architecture** (lazy graph
→ fusion scheduler → a handful of parameterized kernels → captured command
buffer replay); JAX contributes **IR hygiene** (a small explicit typed IR and
pure functional ops so autodiff and JIT compose). fp32 is the main dtype. We
judge quality only by two measured numbers per op and per model step:
`max_abs_diff` vs PyTorch and `pytorch_ms / bitsy_ms`. The framework is "done"
when the core is under 4096 lines, every implemented op matches PyTorch within
tolerance, and the speed ratios reach 1.00 — or a documented ceiling explains
why they cannot.

## Reproducing the progress image

```sh
python tools/gen_progress.py     # writes assets/progress.html + assets/progress.svg from docs/progress.json
python tools/count_lines.py      # enforces the 4096-line C core budget (implementation branch)
```

See [ROADMAP.md](ROADMAP.md) for the phases and exit criteria.
