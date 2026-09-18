"""One table of every cell from the rerun: arms, best black-box, and the white-box advantage.

Run:  C:\\venv\\mats\\Scripts\\python.exe analysis\\summary_table.py
"""
import csv
import os
import sys

PROJ = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(PROJ, "results.csv")

GROUPS = [
    ("organisms, upstream settings", ["gold", "moon", "flag"]),
    ("placebo (base model, no secret)", ["base", "base-moon", "base-flag"]),
    ("greedy generations (k=1, T=0)", ["gold-greedy", "moon-greedy", "flag-greedy"]),
    ("matched sampling (k=10, T=1.0)", ["gold-k10", "moon-k10", "flag-k10"]),
    ("mixing ladder (gold organism)", ["ladder-nonmix", "ladder-mix1.0", "ladder-mix2.0",
                                        "ladder-mix4.0", "ladder-mix8.0", "ladder-mix10.0"]),
]

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["score_pct"]]
by_cell = {}
for r in rows:
    by_cell.setdefault(r["cell"], {})[r["arm"]] = float(r["score_pct"])

def fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else "-"

for title, cells in GROUPS:
    print(f"\n=== {title} ===")
    print(f"{'cell':<18}{'I/O':>7}{'prefill':>9}{'persona':>9}{'lens':>8}{'best BB':>9}{'adv':>8}")
    for c in cells:
        d = by_cell.get(c)
        if not d:
            continue
        bb = {a: d[a] for a in ("io", "prefill", "persona") if a in d}
        lens = d.get("logitlens")
        best = max(bb.values()) if bb else None
        adv = (lens - best) if (lens is not None and best is not None) else None
        label = c.replace("ladder-", "")
        print(f"{label:<18}{fmt(d.get('io')):>7}{fmt(d.get('prefill')):>9}{fmt(d.get('persona')):>9}"
              f"{fmt(lens):>8}{fmt(best):>9}{(f'{adv:+.1f}' if adv is not None else '-'):>8}")
