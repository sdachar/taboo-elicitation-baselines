"""Prompt-level bootstrap intervals for the MATCHED-DECODING table (§4.2).

Why this exists separately from bootstrap_intervals.py: that script compares arms *within* one
result cell. The matched table needs arms assembled ACROSS cells, because the two decoding regimes
live in different runs:

    regime    arm          cell              upstream setting
    -------   ----------   ---------------   ------------------------------
    sampled   io           {base}            run_inference.sh    k=10, T=1.0
    sampled   logit_lens   {base}            run_inference.sh    k=10, T=1.0
    sampled   prefill      {base}-k10        k10_run_prefill     k=10, T=1.0
    sampled   persona      {base}-k10        k10_run_user_persona
    greedy    io           {base}-greedy     greedy_run_inference k=1, T=0.0
    greedy    logit_lens   {base}-greedy     greedy_run_inference k=1, T=0.0
    greedy    prefill      {base}            run_prefill.sh      k=1, T=0.0
    greedy    persona      {base}            run_user_persona.sh k=1, T=0.0

Every cell runs on taboo_standard_test.txt, so prompts are shared and every comparison is PAIRED on
the same prompt resample. Clusters are prompts, not responses: the k=10 arms are 100 clusters, not
1000 draws.

Usage:  python matched_intervals.py [raw_dir] [n_boot]
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "raw")
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
SEED = 20260917
BASES = ["gold", "moon", "flag"]
BLACK = ["io", "prefill", "persona"]
PRETTY = {"io": "I/O", "prefill": "prefill", "persona": "persona", "logit_lens": "logit lens"}

# arm -> cell suffix, per regime
REGIMES = {
    "sampled": {"io": "", "logit_lens": "", "prefill": "-k10", "persona": "-k10"},
    "greedy": {"io": "-greedy", "logit_lens": "-greedy", "prefill": "", "persona": ""},
}


def load(cell, arm):
    """prompt -> list of 0/1 successes, from upstream's own success flag."""
    pat = os.path.join(RAW, cell, "audit", arm, "metrics_*.json")
    out = defaultdict(list)
    for path in sorted(glob.glob(pat)):
        try:
            blob = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for rec in blob.get("results", []):
            if isinstance(rec, dict) and "success" in rec:
                out[rec.get("user_prompt", "")].append(int(bool(rec["success"])))
        if out:
            break
    return out


def mean_of(clusters, keys):
    """Cluster-respecting mean: average within prompt, then across prompts."""
    vals = []
    for k in keys:
        c = clusters.get(k)
        if c:
            vals.append(sum(c) / len(c))
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def shared(*cluster_dicts):
    keys = set(cluster_dicts[0])
    for d in cluster_dicts[1:]:
        keys &= set(d)
    return sorted(keys)


def boot(stat, keys, n_boot=N_BOOT):
    """Paired bootstrap over prompt keys. `stat` takes a key list, returns a scalar."""
    rng = random.Random(SEED)
    point = stat(keys)
    draws = []
    for _ in range(n_boot):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(stat(sample))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def ci(lo, hi):
    return f"[{lo:+.1f}, {hi:+.1f}]"


# ---------------------------------------------------------------- load everything
data = {}
for base in BASES:
    for regime, mapping in REGIMES.items():
        for arm, suffix in mapping.items():
            cell = base + suffix
            data[(base, regime, arm)] = load(cell, arm)

print(f"matched-decoding bootstrap — {N_BOOT} resamples, seed {SEED}")
print(f"raw: {os.path.abspath(RAW)}\n")

# ---------------------------------------------------------------- per-arm levels
print("per-arm scores (cluster-respecting, prompt-level CI)\n")
print(f"{'organism':<9}{'regime':<9}{'arm':<12}{'score':>7}{'95% CI':>18}{'prompts':>9}{'resp/p':>8}")
for base in BASES:
    for regime in ("sampled", "greedy"):
        for arm in BLACK + ["logit_lens"]:
            cl = data[(base, regime, arm)]
            if not cl:
                continue
            keys = sorted(cl)
            pt, lo, hi = boot(lambda ks, c=cl: mean_of(c, ks), keys)
            rpp = sum(len(v) for v in cl.values()) / len(keys)
            print(f"{base:<9}{regime:<9}{PRETTY[arm]:<12}{pt:>7.1f}{f'[{lo:.1f}, {hi:.1f}]':>18}"
                  f"{len(keys):>9}{rpp:>8.1f}")
    print()

# ---------------------------------------------------------------- the matched table
print("\nwhite-box advantage, paired over prompts, best black-box per regime\n")
print(f"{'organism':<9}{'regime':<9}{'best black-box':<15}{'white':>7}{'black':>7}"
      f"{'adv':>8}{'95% CI':>18}  excl. 0?")
adv_stats = {}
for base in BASES:
    for regime in ("sampled", "greedy"):
        ll = data[(base, regime, "logit_lens")]
        cands = {a: data[(base, regime, a)] for a in BLACK if data[(base, regime, a)]}
        if not ll or not cands:
            continue
        best = max(cands, key=lambda a: mean_of(cands[a], sorted(cands[a])))
        bb = cands[best]
        keys = shared(ll, bb)
        pt, lo, hi = boot(lambda ks, l=ll, b=bb: mean_of(l, ks) - mean_of(b, ks), keys)
        adv_stats[(base, regime)] = (best, ll, bb, keys)
        excl = "yes" if (lo > 0 or hi < 0) else "NO"
        print(f"{base:<9}{regime:<9}{PRETTY[best]:<15}{mean_of(ll, keys):>7.1f}"
              f"{mean_of(bb, keys):>7.1f}{pt:>+8.1f}{ci(lo, hi):>18}  {excl}")
    print()

# ---------------------------------------------------------------- the decoding effect
print("\ndifference-in-differences: how much the advantage moves with decoding")
print("(sampled advantage) - (greedy advantage), paired on prompts shared by all four arms\n")
print(f"{'organism':<9}{'adv sampled':>12}{'adv greedy':>12}{'difference':>12}{'95% CI':>18}  excl. 0?")
for base in BASES:
    if (base, "sampled") not in adv_stats or (base, "greedy") not in adv_stats:
        continue
    _, ll_s, bb_s, _ = adv_stats[(base, "sampled")]
    _, ll_g, bb_g, _ = adv_stats[(base, "greedy")]
    keys = shared(ll_s, bb_s, ll_g, bb_g)

    def dd(ks):
        return (mean_of(ll_s, ks) - mean_of(bb_s, ks)) - (mean_of(ll_g, ks) - mean_of(bb_g, ks))

    pt, lo, hi = boot(dd, keys)
    a_s = mean_of(ll_s, keys) - mean_of(bb_s, keys)
    a_g = mean_of(ll_g, keys) - mean_of(bb_g, keys)
    excl = "yes" if (lo > 0 or hi < 0) else "NO"
    print(f"{base:<9}{a_s:>+12.1f}{a_g:>+12.1f}{pt:>+12.1f}{ci(lo, hi):>18}  {excl}")

# ---------------------------------------------------------------- uplift over the I/O baseline
# Answers the authors' stated concern directly: on the standard set, does an attack beat simply
# asking? Must be computed WITHIN a regime — comparing a greedy attack against a sampled baseline
# is the same mismatch this script exists to expose.
print("\n\nuplift over the I/O baseline, within regime, paired on prompts\n")
print(f"{'organism':<9}{'regime':<9}{'arm':<10}{'arm':>7}{'I/O':>7}{'uplift':>8}{'95% CI':>18}  excl. 0?")
for base in BASES:
    for regime in ("sampled", "greedy"):
        io = data[(base, regime, "io")]
        if not io:
            continue
        for arm in ("prefill", "persona"):
            a = data[(base, regime, arm)]
            if not a:
                continue
            keys = shared(a, io)
            pt, lo, hi = boot(lambda ks, x=a, y=io: mean_of(x, ks) - mean_of(y, ks), keys)
            excl = "yes" if (lo > 0 or hi < 0) else "NO"
            print(f"{base:<9}{regime:<9}{PRETTY[arm]:<10}{mean_of(a, keys):>7.1f}"
                  f"{mean_of(io, keys):>7.1f}{pt:>+8.1f}{ci(lo, hi):>18}  {excl}")
    print()

# ---------------------------------------------------------------- per-arm decoding sensitivity
print("\n\nper-arm decoding sensitivity: sampled minus greedy, same arm, paired on prompts\n")
print(f"{'organism':<9}{'arm':<12}{'sampled':>8}{'greedy':>8}{'delta':>8}{'95% CI':>18}  excl. 0?")
for base in BASES:
    for arm in BLACK + ["logit_lens"]:
        s, g = data[(base, "sampled", arm)], data[(base, "greedy", arm)]
        if not s or not g:
            continue
        keys = shared(s, g)
        if not keys:
            continue
        pt, lo, hi = boot(lambda ks, a=s, b=g: mean_of(a, ks) - mean_of(b, ks), keys)
        excl = "yes" if (lo > 0 or hi < 0) else "NO"
        print(f"{base:<9}{PRETTY[arm]:<12}{mean_of(s, keys):>8.1f}{mean_of(g, keys):>8.1f}"
              f"{pt:>+8.1f}{ci(lo, hi):>18}  {excl}")
    print()
