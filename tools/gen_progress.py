"""Generate assets/progress.svg in the AnyPPS5 progress-chart style.

The reference image (https://camo.githubusercontent.com/.../progress.svg) is a
dark-background SVG with, per category: a white "Name: XX.XX% (a/b)" label and
a wrapped grid of small green squares (one per unit, green = done/pass). We
reuse that visual language but track the two axes that matter here: numeric
parity vs PyTorch and speed parity vs PyTorch (plus line-budget usage).

Dataclass config, no argparse. Regenerate with: python tools/gen_progress.py
"""

from __future__ import annotations

import json, math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "progress.json"
OUT = ROOT / "assets" / "progress.svg"

BG = "#0d1117"
FG = "#ffffff"
GREEN = "#2ea043"   # pass / done
AMBER = "#d29922"   # partial / numeric-only
RED = "#f85149"     # miss / failing
GREY = "#30363d"    # not done

@dataclass(frozen=True, slots=True)
class Layout:
  width:int = 1000
  label_h:int = 30
  box:float = 16.0
  gap:float = 2.0
  pad:float = 1.0
  font:int = 16
  cols:int = 0  # computed

def wrap_boxes(n:int, done:int, width:int, box:float, gap:float, color_done:str,
               color_rest:str) -> list[str]:
  per_row = max(1, int((width - 2) // (box + gap)))
  out = []
  for i in range(n):
    r, c = divmod(i, per_row)
    x = 1 + c * (box + gap)
    y = 10 + r * (box + gap)
    col = color_done if i < done else color_rest
    out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{box:.2f}" height="{box:.2f}" '
               f'fill="{col}" stroke="{BG}" stroke-width="0.5"/>')
  return out

def esc(s:str) -> str:
  return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))

def render(data:dict, cfg:Layout) -> str:
  parts:list[str] = []
  y = 0.0
  def add_title(text:str) -> None:
    nonlocal y
    parts.append(f'<text x="4.00" y="{y+21:.2f}" font-family="sans-serif" '
                 f'font-size="{cfg.font}" fill="{FG}" stroke="{BG}" '
                 f'stroke-width="3" paint-order="stroke">{esc(text)}</text>')
    y += cfg.label_h

  title = data.get("title", "Bitsy progress")
  parts.append(f'<text x="4.00" y="{y+24:.2f}" font-family="sans-serif" font-size="22" '
               f'font-weight="bold" fill="{FG}">{esc(title)}</text>')
  y += 34
  parts.append(f'<text x="4.00" y="{y+18:.2f}" font-family="sans-serif" font-size="13" '
               f'fill="#8b949e">{esc(data.get("subtitle",""))}</text>')
  y += 30

  max_y = y
  for group in data.get("groups", []):
    name = group["name"]
    total = int(group["total"])
    passed = int(group["passed"])
    pct = 100.0 * passed / total if total else 0.0
    add_title(f"{name}: {pct:.2f}% ({passed}/{total})")
    boxes = wrap_boxes(total, passed, cfg.width, cfg.box, cfg.gap,
                       group.get("color", GREEN), GREY)
    parts.extend(boxes)
    max_y = max(max_y, y + (math.ceil(total / max(1, int((cfg.width-2)//(cfg.box+cfg.gap)))) ) * (cfg.box+cfg.gap))
    y = max_y + 8

  # legend + footer
  y = max_y + 22
  parts.append(f'<text x="4.00" y="{y:.2f}" font-family="sans-serif" font-size="13" '
               f'fill="#8b949e">Legend: green = pass / done, amber = numeric-only, '
               f'grey = pending. Regenerate: python tools/gen_progress.py</text>')
  y += 22
  parts.append(f'<text x="4.00" y="{y:.2f}" font-family="sans-serif" font-size="12" '
               f'fill="#6e7681">{esc(data.get("footer",""))}</text>')
  y += 14

  header = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cfg.width} {int(y)}" '
            f'width="{cfg.width}" height="{int(y)}">\n'
            f'<rect x="0.00" y="0.00" width="{cfg.width}.00" height="{int(y)}" '
            f'fill="{BG}" stroke="{BG}" stroke-width="0"/>\n')
  return header + "\n".join(parts) + "\n</svg>\n"

def run() -> int:
  cfg = Layout()
  data = json.loads(DATA.read_text(encoding="utf-8"))
  svg = render(data, cfg)
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(svg, encoding="utf-8")
  print(f"wrote {OUT.relative_to(ROOT)} ({len(svg)} bytes, "
        f"{len(data.get('groups',[]))} groups)")
  return 0

if __name__ == "__main__":
  raise SystemExit(run())
