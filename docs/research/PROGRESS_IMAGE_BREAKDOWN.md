# Breakdown of the reference progress image and its application to Bitsy

Reference image: the AnyPPS5 `progress.svg`
(https://camo.githubusercontent.com/37d20d81b0b8d89260368187e1e885545af2ed0659260ad4fb9cc7aaff20eb03/68747470733a2f2f626f796b6f706f7661722e6769746875622e696f2f416e795053352f70726f67726573732e737667).
This document deconstructs its visual grammar and maps every element to a
concrete Bitsy data source. The generator `tools/gen_progress.py` implements
this mapping; data lives in `docs/progress.json`.

## 1. What the reference image actually contains

Reading the image top-to-bottom, it is **not** a single chart — it is a
dashboard composed of four distinct visual encodings:

### 1.1 Percentage headers
Lines like `System libraries*: 45.23% (17756/39253)` and
`Large system libraries: 43.71% (12575/28771)`.
- Format: `Name: XX.XX% (done/total)`, bold white on dark background.
- Encodes both a rate and the raw counts, so the reader sees scale and ratio.

### 1.2 Squarified treemap with micro-grids (the dominant element)
The upper ~2/3 is a **treemap**: rectangles packed to fill the area, where a
cell's **area is proportional to the number of items** in that group. Each cell:
- has a **name label** (`NKWebKit`, `Icu`, `Libc++`, `libker`, `liblinker`),
- often contains **nested sub-labels** for subgroups (`Jsc`, `UserSer`,
  `LibreSS`, `NpMan`, `Add`, `lib`, `Agc`, …),
- is filled with a **dense grid of tiny squares**: green = item done/decoded,
  dark grey = item remaining. The grid's square count tracks the item count,
  so a big library is both a bigger rectangle and a denser grid.

Two large panels show `Gfx1013 reference forms decoded: 95.13% (1484/1560)`
and `RDNA2 reference forms decoded: 100.0% (1460/1460)` — same grammar,
showing completion at the level of *individual instruction forms*.

### 1.3 Horizontal bar charts with counts
- `Unsupported-throw sites by area (126)` — one bar per area (Translation,
  RuntimeDecoder, Optimization, EnvBackend, ControlFlow, Intermediary,
  Recompiler), green portion = handled, grey = remaining, exact count on the
  right (56, 24, 20, 13, 7, 4, 2).
- `Image formats (of 159)` — `Sampled 103/159`, `Typed buffer 80/159`,
  `Still needed format 90/159` with green fill and counts.

### 1.4 A "still needed" text list
`Instruction forms still needed (76)` — a two-column monospace list of the
remaining item names (e.g. `DS_DEC_SCR2_U64`, `S_ATOMIC_ADD_X2`,
`S_BUFFER_ATOMIC_CMP_SWAP2`, …). This turns the "grey" cells into an actionable
checklist.

### 1.5 Visual style
Dark background `#0d1117`, green `#2ea043` for done, amber `#d29922` for
partial (the Translation bar is amber/green), grey `#30363d` for remaining,
white labels with a dark stroke so text stays readable over the grid.

## 2. Mapping each element to Bitsy

Bitsy's progress has the same shape: many small items (ops, gradients, model
steps), grouped into areas, with two independent axes (numeric parity and
speed parity), plus a line budget and a planned-surface backlog.

| Reference element | Bitsy mapping | Data source |
|---|---|---|
| Per-group % header | "Op numerics 100% (48/48)", "Model numerics 100% (5/5)" | `docs/progress.json` groups |
| Treemap cell area ∝ item count | Feature areas sized by number of functions/steps: elementwise, activations, matmul, reductions, autograd, optim, layers, models | `groups[].children[]` with `total` |
| Cell micro-grid green/pending | One square per op/step, green if numeric pass within tolerance | each child's `passed`/`total` |
| Nested sub-labels | e.g. `matmul` cell shows `fwd`, `dx`, `dW`; `GPT-2` shows `logits`,`loss`,`grads` | child `label` fields |
| `forms decoded %` panels | "numeric forms matched %" per library layer (core ops vs nn vs models) | aggregate groups |
| Horizontal bars + counts | **Speed** ratio per model/step with a target tick at 1.0; line-budget bar; numeric bars | `speed_bars`, `line_budget` |
| "still needed" list | Planned surface from parity matrix (GRU, MHA, Transformer, fp16/bf16, Datasets, safetensors, DLPack, distributed, speed gaps) | `still_needed[]` |
| Green/amber/grey | green = numeric pass or speed ≥1.0; amber = numeric-only / speed 0.5–1.0; grey = pending / speed <0.5 | generator color tiers |

Why a treemap fits here: the parity matrix has natural hierarchy (area → op →
forward/grad), item counts vary widely (48 op cases vs 3 GPT-2 grad checks),
and the two axes can be shown as (a) a numeric treemap (mostly green, the
"already achieved" axis) and (b) a speed panel of bars (the open gap). This
mirrors the reference's "forms decoded" (solved) vs "still needed" (open)
split.

## 3. Authored in HTML, exported to SVG

Following the request to author the page in HTML (far easier for rich layout
than hand-written SVG), the dashboard is now a two-output system driven by one
data model:

- **`assets/progress.html`** is the human-friendly source view: a self-contained
  HTML/CSS page using flex rows for the cells (each cell's `flex-grow` tracks its
  item count), a CSS-grid micro-grid of tiny squares, flex bars for speed and
  line budget, and a two-column CSS backlog list. Open it in a browser for the
  full view; it is easy to restyle because layout lives in CSS, not geometry
  math.
- **`assets/progress.svg`** is a native-SVG export of the *same* `docs/progress.json`
  data, embedded in the README. Native SVG is used (rather than wrapping the HTML
  in `<foreignObject>`) because GitHub's image sanitizer strips HTML from
  README SVGs — a foreignObject version would render blank there. Both views are
  written by one invocation of `tools/gen_progress.py`, so they cannot drift.

The page design itself was produced as a standalone HTML template
(`assets/progress.template.html`) with a single `const DATA = __DATA__;`
placeholder — the JS in the template computes the overview score, roadmap,
speed chart, parity micro-grids, backlog and budget from that object. This is
the human-friendly, easy-to-restyle source of the design.

`tools/gen_progress.py`:
1. Reads the single source of truth, `docs/progress.json`.
2. Injects that JSON into the template's `__DATA__` placeholder and writes
   `assets/progress.html` — the full interactive dashboard.
3. `render_svg()` exports a native SVG from the **same** data, mirroring every
   section: overall health + sub-scores, roadmap phases, speed bars with the
   1.00 target marker and tier colors, parity tiles with micro-grids, the
   "still needed" list and the line-budget bar.
4. Writes both files and prints a summary. The README embeds the native SVG
   (a foreignObject-wrapped HTML page would be stripped by GitHub's sanitizer).

## 4. Honesty rules preserved

- Green means "measured pass within documented tolerance", not "approximate".
- Speed bars show the actual `torch_ms / bitsy_ms`; sub-1.0 bars are labelled
  as misses, never hidden. The target marker makes the gap visible.
- The image is generated from JSON committed alongside it; `python
  tools/gen_progress.py` is the single way to update it, so the picture can
  never drift silently from the data.
- This remains a documentation-only branch: the numbers are the inherited
  baseline from the pre-skeleton history (canonical tables in `main`'s
  PROGRESS.md/BENCHMARKS.md), labelled as such in the subtitle.
