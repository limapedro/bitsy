"""Generate the Bitsy progress dashboard.

Source of truth is docs/progress.json. One run produces:
  * assets/progress.html  — authored HTML/CSS template with DATA injected
  * assets/progress.svg   — native SVG export (GitHub-safe; README embeds this)

The HTML template (assets/progress.template.html) was designed as a creative,
informative dashboard: overall health score, roadmap, speed bars with a 1.00
target, parity micro-grids, backlog and line budget. The native SVG mirrors those
sections from the same data so neither view can drift. Dataclass config, no
argparse. Regenerate: python tools/gen_progress.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "progress.json"
TPL = ROOT / "assets" / "progress.template.html"
OUT = ROOT / "assets" / "progress.html"
OUT_SVG = ROOT / "assets" / "progress.svg"

BG, PANEL, LINE, TRACK = "#0d1117", "#161b22", "#30363d", "#21262d"
MUT, TXT = "#8b949e", "#e6edf3"
GREEN, AMBER, RED = "#2ea043", "#d29922", "#f85149"


@dataclass(frozen=True, slots=True)
class Layout:
  width:int = 1180
  pad:float = 18.0
  h2:float = 30.0
  hint:float = 22.0


def tier(ratio:float) -> str:
  return GREEN if ratio >= 1.0 else (AMBER if ratio >= 0.5 else RED)

def score_tier(v:float|None) -> str:
  if v is None: return MUT
  return GREEN if v >= 0.8 else (AMBER if v >= 0.5 else RED)

def esc(s:str) -> str:
  return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;"))

def rect(x:float,y:float,w:float,h:float,fill:str,stroke:str|None=None,sw:float=0.6) -> str:
  st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
  return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w,0):.2f}" '
          f'height="{max(h,0):.2f}" fill="{fill}"{st}/>')

def text(x:float,y:float,s:str,size:int=13,color:str=TXT,bold:bool=False,
         mono:bool=False,anchor:str="start") -> str:
  w = ' font-weight="700"' if bold else ''
  return (f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}"'
          f'{w} font-family="{"ui-monospace,Consolas,monospace" if mono else "Segoe UI,Roboto,sans-serif"}"'
          f' text-anchor="{anchor}">{esc(s)}</text>')

def exp(x) -> str:
  if x is None: return "—"
  if x == 0: return "0"
  return f"{x:.1e}".replace("e-", "e−")


# ---------------------------------------------------------------------------
# score computation (mirrors the template's JS so both views agree)
# ---------------------------------------------------------------------------
def scores(data:dict) -> tuple[float, list[dict]]:
  tiles = [t for g in data["parity"]["groups"] for t in g["tiles"]]
  gops, gmod = data["parity"]["groups"][0], data["parity"]["groups"][1]
  b, ph = data["budget"], data["phases"]
  nP = sum(t["pass"] for t in tiles); nT = sum(t["total"] for t in tiles)
  budget_score = None if b["status"] == "planned" else (
    1.0 if b["used"] <= b["target"] else max(0.0, 1 - (b["used"]-b["target"])/(b["cap"]-b["target"])))
  S = [
    {"k":"numerics","n":"Numerics","v": (nP/nT if nT else 0),
     "d": f'{sum(t["pass"] for t in gops["tiles"])}/{sum(t["total"] for t in gops["tiles"])} ops · '
          f'{sum(t["pass"] for t in gmod["tiles"])}/{sum(t["total"] for t in gmod["tiles"])} models pass'},
    {"k":"speed","n":"Speed","v": sum(min(s["ratio"],1.0) for s in data["speed"])/len(data["speed"]),
     "d": f'mean of ratios capped at 1.00 · '
          f'{sum(1 for s in data["speed"] if s["ratio"]>=1.0)}/{len(data["speed"])} at parity'},
    {"k":"completeness","n":"Completeness","v": sum(p["progress"] for p in ph)/len(ph),
     "d": f'{sum(1 for p in ph if p["status"]=="done")}/{len(ph)} phases done'},
    {"k":"budget","n":"Line budget","v": budget_score,
     "d": (f'planned · target {b["target"]} of {b["cap"]} lines' if b["status"]=="planned"
           else f'{b["used"]} of {b["cap"]} lines used')},
  ]
  live = [s for s in S if s["v"] is not None]
  W = data["weights"]
  head = sum(s["v"]*W[s["k"]] for s in live) / sum(W[s["k"]] for s in live)
  return head, S


# ---------------------------------------------------------------------------
# native SVG export
# ---------------------------------------------------------------------------
def render_svg(D:dict, cfg:Layout) -> str:
  W = cfg.width; pad = cfg.pad; parts:list[str] = []
  y = 0.0

  def title(s:str,size:int=18,color:str=TXT) -> None:
    nonlocal y
    parts.append(text(pad,y+size, s,size,color,bold=True)); y += cfg.h2

  # header
  p = D["project"]
  parts.append(text(pad, y+28, p["name"]+" — progress toward PyTorch parity", 26, TXT, bold=True, mono=True))
  y += 34
  parts.append(text(pad, y+12, p["tagline"], 12, MUT)); y += 24
  parts.append(text(pad, y+12, " · ".join(p["chips"]), 11, MUT)); y += 28

  # at a glance
  title("At a glance — weighted health score")
  head, S = scores(D)
  parts.append(rect(pad,y,W-2*pad,96,PANEL,LINE))
  parts.append(text(pad+16,y+26,"Overall health",12,MUT))
  parts.append(text(pad+16,y+76,f"{head*100:.0f}%",48,score_tier(head),bold=True,mono=True))
  segn=50; segw=(W-2*pad-40)/segn
  for i in range(segn):
    parts.append(rect(pad+320+i*segw, y+54, segw-1.5, 26,
                      score_tier(head) if i < round(head*segn) else LINE))
  parts.append(text(pad+320,y+30," + ".join(f'{s["n"].lower()} {int(D["weights"][s["k"]]*100)}' for s in S if s["v"] is not None),11,MUT))
  y += 96 + 14

  cw = (W-2*pad-3*8)/4
  for s in S:
    x = pad + (["numerics","speed","completeness","budget"].index(s["k"]))*(cw+8)
    parts.append(rect(x,y,cw,86,PANEL,LINE))
    parts.append(text(x+10,y+20,s["n"],12,MUT))
    v = s["v"]
    parts.append(text(x+10,y+56, ("n/a" if v is None else f"{v*100:.0f}%"),26,
                      score_tier(v),bold=True,mono=True))
    parts.append(rect(x+10,y+66,cw-20,6,TRACK))
    parts.append(rect(x+10,y+66,0 if v is None else (cw-20)*v,6, score_tier(v) if v else LINE))
    parts.append(text(x+10,y+82,s["d"][:40],9,MUT))
  y += 86 + 22

  # roadmap
  title("Roadmap — five phases from research to release")
  n = len(D["phases"]); pw = (W-2*pad-(n-1)*8)/n
  for i,ph in enumerate(D["phases"]):
    x = pad + i*(pw+8)
    col = {"done":GREEN,"active":AMBER,"pending":LINE}[ph["status"]]
    parts.append(rect(x,y,pw,104,PANEL,LINE))
    parts.append(rect(x,y,pw,4,col))
    parts.append(rect(x+10,y+14,24,24,TRACK))
    parts.append(text(x+22,y+31,str(ph["id"]),13,TXT,bold=True,mono=True,anchor="middle"))
    parts.append(text(x+10,y+56,ph["name"],13,TXT,bold=True))
    parts.append(text(x+10,y+72,ph["status"].upper()+f' · {ph["progress"]*100:.0f}%',10,col,mono=True))
    parts.append(rect(x+10,y+82,pw-20,6,TRACK))
    parts.append(rect(x+10,y+82,(pw-20)*ph["progress"],6,col))
  y += 104 + 22

  # speed
  title("Speed vs PyTorch — ratio torch_ms / bitsy_ms (1.00 = as fast)")
  L = 250; tw = 520; maxv = 1.25; sx = tw/maxv
  X = lambda r: L + min(r,maxv)*sx
  y += 4
  for v,lbl in ((0.0,"0"),(0.5,"0.50"),(1.0,"target 1.00")):
    parts.append(rect(X(v),y,1,240,TXT if v==1 else LINE))
    parts.append(text(X(v),y-4,lbl,10,MUT if v!=1 else TXT,mono=True,anchor="middle"))
  y += 8
  for scope,label in (("per-step","Per-step: one op or training iteration"),("full-model","Full-model: end-to-end run")):
    parts.append(text(pad,y+14,label,12,MUT,bold=True)); y += 22
    for s in [s for s in D["speed"] if s["scope"]==scope]:
      r = s["ratio"]; col = tier(r)
      parts.append(text(pad,y+16,s["name"],13,TXT))
      parts.append(text(pad,y+30,s.get("note",""),10,MUT))
      parts.append(rect(L,y+6,tw,22,TRACK))
      parts.append(rect(L,y+6,X(r)-L,22,col))
      parts.append(text(L+tw+14,y+22,f"{r:.2f}×",14,col,bold=True,mono=True))
      parts.append(text(L+tw+90,y+22, (f"{(1/r):.1f}× to go" if r<1 else "at parity"),10,MUT,mono=True))
      y += 52
  y += 10

  # parity
  title("Numeric parity — each square is one check vs PyTorch (green = pass)")
  for g in D["parity"]["groups"]:
    tiles=g["tiles"]; tp=sum(t["pass"] for t in tiles); tt=sum(t["total"] for t in tiles)
    parts.append(text(pad,y+16,g["title"],14,TXT,bold=True))
    parts.append(text(pad+90,y+16,f"{100*tp/tt:.2f}% ({tp}/{tt})",13, GREEN if tp==tt else AMBER,mono=True))
    y += 24
    tw2 = (W-2*pad-3*8)/4
    for i,t in enumerate(tiles):
      col_i = i%4; row_i = i//4
      x = pad+col_i*(tw2+8); yy = y+row_i*54
      tc = GREEN if t["pass"]==t["total"] else (AMBER if t["pass"] else LINE)
      parts.append(rect(x,yy,tw2,50,PANEL,LINE))
      parts.append(rect(x,yy,tw2,3,tc))
      parts.append(text(x+8,yy+18,f'{t["name"]} {t["pass"]}/{t["total"]}',11,tc,bold=True))
      parts.append(text(x+8,yy+34,f'max|Δ| {exp(t["maxAbsDiff"])} · {t["tol"]}',9,MUT,mono=True))
      # micro grid (one square per check, capped visually)
      n=min(t["total"],24); ms=3.0; mg=1.0; per=min(n,12)
      for k in range(n):
        r,c=divmod(k,per)
        parts.append(rect(x+tw2-(per*(ms+mg))+c*(ms+mg), yy+8+r*(ms+mg), ms,ms,
                          GREEN if k < round(t["pass"]/t["total"]*n) else LINE))
    y += ((len(tiles)+3)//4)*54 + 14

  # backlog + budget
  parts.append(text(pad,y+16,"Still needed (open work by phase)",14,TXT,bold=True)); y += 26
  for b in sorted(D["backlog"],key=lambda z:z["phase"]):
    parts.append(rect(pad,y,30,16,TRACK))
    parts.append(text(pad+15,y+12,f'P{b["phase"]}',10,TXT,mono=True,anchor="middle"))
    parts.append(text(pad+38,y+12,b["title"],11,MUT))
    parts.append(text(W-pad,y+12,b["metric"],10,MUT,mono=True,anchor="end")); y += 20
  y += 10
  b=D["budget"]
  parts.append(text(pad,y+16,f'Line budget: {b["used"]} / {b["cap"]} lines (target {b["target"]}, stretch {b["stretch"]}; status {b["status"]})',13,TXT,bold=True)); y+=26
  parts.append(rect(pad,y,W-2*pad,18,TRACK))
  parts.append(rect(pad+1,y+1,(W-2*pad-2)*min(b["used"]/b["cap"],1.0),16, GREEN if b["status"]=="planned" else AMBER)); y+=18

  parts.append(text(pad,y+24,f'Snapshot {D["project"]["updated"]} · source {D["project"]["ref"]} · regenerate: python tools/gen_progress.py',10,MUT))
  y += 30
  return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">\n'
          % (W,int(y),W,int(y))) + rect(0,0,W,int(y),BG) + "\n" + "\n".join(parts) + "\n</svg>\n"


def run() -> int:
  data = json.loads(DATA.read_text(encoding="utf-8"))
  tpl = TPL.read_text(encoding="utf-8")
  html = tpl.replace("__DATA__", json.dumps(data, indent=2))
  OUT.write_text(html, encoding="utf-8")
  svg = render_svg(data, Layout())
  OUT_SVG.write_text(svg, encoding="utf-8")
  print(f"wrote {OUT.relative_to(ROOT)} ({len(html)} bytes) and "
        f"{OUT_SVG.relative_to(ROOT)} ({len(svg)} bytes)")
  print(f"groups={len(data['parity']['groups'])} speed={len(data['speed'])} "
        f"backlog={len(data['backlog'])} phases={len(data['phases'])}")
  return 0

if __name__ == "__main__":
  raise SystemExit(run())
