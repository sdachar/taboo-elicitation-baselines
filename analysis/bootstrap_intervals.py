"""Prompt-level bootstrap intervals for the Taboo rerun.

Why not the binomial in intervals.py: the I/O and logit-lens arms draw 10 responses per prompt, so
their 1000 samples are 100 clusters, not 1000 independent draws. A naive binomial makes those
intervals too narrow. This resamples PROMPTS with replacement, which respects the clustering.

The arms also share prompts (all run on taboo_standard_test.txt), so the advantage is estimated
with a PAIRED bootstrap on the same prompt resample — tighter and correct, rather than treating
the two arms as independent samples.

Reads the metrics_*.json files extracted from the pod archive.
Usage:  python bootstrap_intervals.py [raw_dir] [n_boot]
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "raw", "results")
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
SEED = 20260917
ARMS = ["io", "prefill", "persona", "logit_lens"]
PRETTY = {"io": "I/O", "prefill": "prefill", "persona": "persona", "logit_lens": "logit lens"}


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


def boot_one(clusters, n_boot=N_BOOT):
    keys = list(clusters)
    rng = random.Random(SEED)
    point = mean_of(clusters, keys)
    draws = []
    for _ in range(n_boot):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(mean_of(clusters, sample))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)], len(keys)


def boot_diff(a, b, n_boot=N_BOOT):
    """Paired bootstrap of mean(a) - mean(b) over prompts present in both."""
    keys = [k for k in a if k in b]
    rng = random.Random(SEED)
    point = mean_of(a, keys) - mean_of(b, keys)
    draws = []
    for _ in range(n_boot):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(mean_of(a, sample) - mean_of(b, sample))
    draws.sort()
    return point, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)], len(keys)


cells = sorted(d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d)))
data = {c: {a: load(c, a) for a in ARMS} for c in cells}

print(f"prompt-level bootstrap, {N_BOOT} resamples, seed {SEED}\n")
print(f"{'cell':<14}{'arm':<12}{'score':>7}{'95% CI':>18}{'prompts':>9}{'resp/prompt':>12}")
for c in cells:
    for a in ARMS:
        clusters = data[c][a]
        if not clusters:
            continue
        pt, lo, hi, n = boot_one(clusters)
        rpp = sum(len(v) for v in clusters.values()) / max(1, n)
        print(f"{c:<14}{PRETTY[a]:<12}{pt:>7.1f}{f'[{lo:.1f}, {hi:.1f}]':>18}{n:>9}{rpp:>12.1f}")

print("\n\nwhite-box advantage (logit lens - best black-box), paired bootstrap over prompts\n")
print(f"{'cell':<14}{'best black-box':<16}{'white':>7}{'black':>7}{'adv':>8}{'95% CI':>18}  excl. 0?")
for c in cells:
    ll = data[c]["logit_lens"]
    if not ll:
        continue
    bb_candidates = {a: data[c][a] for a in ("io", "prefill", "persona") if data[c][a]}
    if not bb_candidates:
        continue
    best_arm = max(bb_candidates, key=lambda a: mean_of(bb_candidates[a], list(bb_candidates[a])))
    bb = bb_candidates[best_arm]
    pt, lo, hi, n = boot_diff(ll, bb)
    w = mean_of(ll, list(ll))
    b = mean_of(bb, list(bb))
    excl = "yes" if (lo > 0 or hi < 0) else "NO"
    print(f"{c:<14}{PRETTY[best_arm]:<16}{w:>7.1f}{b:>7.1f}{pt:>+8.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>18}  {excl}")

print("\n\nsame organism, sampled vs greedy baseline (the decoding effect)\n")
for base in ("gold", "moon", "flag"):
    g = f"{base}-greedy"
    if g not in data:
        continue
    for label, cell in ((f"{base} (upstream)", base), (f"{base} (greedy)", g)):
        ll, io = data[cell]["logit_lens"], data[cell]["io"]
        if not ll or not io:
            continue
        pt, lo, hi, n = boot_diff(ll, io)
        print(f"  {label:<20} logit lens - I/O = {pt:+6.1f}  [{lo:+.1f}, {hi:+.1f}]  (n={n} prompts)")
    print()
