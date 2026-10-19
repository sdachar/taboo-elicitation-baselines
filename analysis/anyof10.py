"""Any-of-10 aggregation on the SAMPLED cells (k=10, T=1): per prompt, success if any of the 10
responses is scored correct. Paired prompt bootstrap of lens minus each black-box arm and minus the
best arm (selected on any-of-10 means). Usage: python -I anyof10.py <raw_results_dir> [n_boot]"""
import glob, json, os, random, sys
from collections import Counter, defaultdict

RAW = sys.argv[1]; N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000; SEED = 20260917
BASES = ["gold", "moon", "flag"]; BLACK = ["io", "prefill", "persona"]
CELL = {"io": "", "logit_lens": "", "prefill": "-k10", "persona": "-k10"}

def load(cell, arm):
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RAW, cell, "audit", arm, "metrics_*.json"))):
        blob = json.load(open(path, encoding="utf-8"))
        for rec in blob.get("results", []):
            if isinstance(rec, dict) and "success" in rec:
                out[rec.get("user_prompt", "")].append(int(bool(rec["success"])))
        if out: break
    return out

def pct(d): d = sorted(d); return d[int(0.025*len(d))], d[int(0.975*len(d))]

for base in BASES:
    arms = {a: load(base + CELL[a], a) for a in BLACK + ["logit_lens"]}
    keys = sorted(set.intersection(*(set(v) for v in arms.values())))
    n = len(keys)
    anyof = {a: {k: int(any(arms[a][k])) for k in keys} for a in arms}
    meanof = {a: {k: sum(arms[a][k]) / len(arms[a][k]) for k in keys} for a in arms}
    def m(d, wp): return 100 * sum(w * d[k] for k, w in wp.items()) / sum(wp.values())
    full = {k: 1 for k in keys}
    print(f"== {base} (n={n} prompts, 10 responses each)")
    for a in arms:
        print(f"   {a:<10} mean-of-10 {m(meanof[a], full):5.1f}   any-of-10 {m(anyof[a], full):5.1f}")
    best = max(BLACK, key=lambda a: m(anyof[a], full))
    rng = random.Random(SEED)
    draws = {a: [] for a in BLACK + ["best"]}
    for _ in range(N_BOOT):
        wp = Counter(keys[rng.randrange(n)] for _ in range(n))
        for a in BLACK:
            draws[a].append(m(anyof["logit_lens"], wp) - m(anyof[a], wp))
        b = max(BLACK, key=lambda a: m(anyof[a], wp))
        draws["best"].append(m(anyof["logit_lens"], wp) - m(anyof[b], wp))
    for a in BLACK:
        pt = m(anyof["logit_lens"], full) - m(anyof[a], full); lo, hi = pct(draws[a])
        print(f"   any-of-10 advantage lens - {a:<8}: {pt:+6.1f} [{lo:+.1f}, {hi:+.1f}]")
    pt = m(anyof["logit_lens"], full) - m(anyof[best], full); lo, hi = pct(draws["best"])
    print(f"   any-of-10 advantage lens - best({best}), re-selected: {pt:+6.1f} [{lo:+.1f}, {hi:+.1f}]")
    print()
