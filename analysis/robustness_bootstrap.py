"""Meta-review statistics for the Taboo decoding-effect claim.

Reads the per-record audit JSON (user_prompt, model_response, success) for the matched cells and
computes, per organism:
  (1) DiD with the I/O arm fixed as the black-box arm, paired on prompts, prompt-level bootstrap.
  (2) DiD with the best black-box arm RE-SELECTED inside every bootstrap resample.
  (3) A two-way (crossed) cluster bootstrap: prompts resampled (paired across all arms/regimes) AND,
      for the greedy arms, unique model_response texts resampled as clusters. Weighted means with
      w = w_prompt * w_text. Sampled arms have w_text = 1 (every response is unique). Greedy I/O and
      greedy logit-lens share the same generated response so they share the same text weights.
      This is the pigeonhole bootstrap for crossed random effects (Owen & Eckles 2012); it is
      conservative.
  (4) Fragility: if the single modal greedy hint had been judged WRONG by the auditor, what is the
      DiD? (the auditor is deterministic at T=0 so the modal text is one auditor judgement.)
  (5) Lens-arm within-prompt consistency in the sampled cells: fraction of prompts whose 10 lens
      judgements are identical, versus I/O. (The lens tokens are identical across the 10 responses
      of a prompt, so high consistency = the auditor is reading the tokens, not the transcript.)

Cell mapping mirrors analysis/matched_intervals.py.
Usage: python -I meta_stats.py <raw_results_dir> [n_boot]
"""
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

RAW = sys.argv[1]
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
SEED = 20260917
BASES = ["gold", "moon", "flag"]
BLACK = ["io", "prefill", "persona"]
REGIMES = {
    "sampled": {"io": "", "logit_lens": "", "prefill": "-k10", "persona": "-k10"},
    "greedy": {"io": "-greedy", "logit_lens": "-greedy", "prefill": "", "persona": ""},
}


def load(cell, arm):
    """prompt -> list of (success, response_text)."""
    pat = os.path.join(RAW, cell, "audit", arm, "metrics_*.json")
    out = defaultdict(list)
    for path in sorted(glob.glob(pat)):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        for rec in blob.get("results", []):
            if isinstance(rec, dict) and "success" in rec:
                out[rec.get("user_prompt", "")].append(
                    (int(bool(rec["success"])), (rec.get("model_response") or "").strip()))
        if out:
            break
    return out


data = {}
for base in BASES:
    for regime, mapping in REGIMES.items():
        for arm, suffix in mapping.items():
            data[(base, regime, arm)] = load(base + suffix, arm)


def prompts_shared(base):
    keys = None
    for regime in REGIMES:
        for arm in BLACK + ["logit_lens"]:
            d = data[(base, regime, arm)]
            keys = set(d) if keys is None else keys & set(d)
    return sorted(keys)


def mean_w(clusters, wp, wt=None):
    """Weighted cluster mean: within-prompt mean of records (weighted by text weight), then
    across prompts weighted by prompt weight. wt maps response text -> weight (None = all 1)."""
    num = den = 0.0
    for k, w in wp.items():
        if w == 0:
            continue
        recs = clusters.get(k)
        if not recs:
            continue
        if wt is None:
            m = sum(s for s, _ in recs) / len(recs)
            num += w * m
            den += w
        else:
            n2 = d2 = 0.0
            for s, t in recs:
                tw = wt.get(t, 0)
                n2 += tw * s
                d2 += tw
            if d2 > 0:
                num += w * (n2 / d2)
                den += w
    return 100 * num / den if den else float("nan")


def pct(draws, lo=0.025, hi=0.975):
    d = sorted(draws)
    return d[int(lo * len(d))], d[int(hi * len(d))]


def fmt(pt, lo, hi):
    ex = "yes" if (lo > 0 or hi < 0) else "NO"
    return f"{pt:+6.1f}  [{lo:+.1f}, {hi:+.1f}]  excl0={ex}"


print(f"raw: {os.path.abspath(RAW)}   n_boot={N_BOOT}  seed={SEED}\n")

for base in BASES:
    keys = prompts_shared(base)
    n = len(keys)
    print(f"================= {base}  (shared prompts: {n})")
    full = {k: 1 for k in keys}
    arms = {}
    for regime in REGIMES:
        for arm in BLACK + ["logit_lens"]:
            arms[(regime, arm)] = data[(base, regime, arm)]

    # ---- uniqueness in greedy cells
    for arm in BLACK + ["logit_lens"]:
        recs = [r for k in keys for r in arms[("greedy", arm)][k]]
        c = Counter(t for _, t in recs)
        modal_text, modal_n = c.most_common(1)[0]
        modal_succ = sum(s for s, t in recs if t == modal_text)
        print(f"  greedy {arm:<10} n={len(recs)} unique={len(c)} modal x{modal_n} modal_scored={modal_succ}/{modal_n}")
    # sanity: greedy io and lens share text?
    same = sum(1 for k in keys if arms[("greedy", "io")][k][0][1] == arms[("greedy", "logit_lens")][k][0][1])
    print(f"  greedy io/lens same response text on {same}/{n} prompts")

    # ---- (5) lens-arm within-prompt consistency, sampled cells
    for arm in ["logit_lens", "io"]:
        cl = arms[("sampled", arm)]
        const = sum(1 for k in keys if len(set(s for s, _ in cl[k])) == 1)
        print(f"  sampled {arm:<10}: prompts with all-identical judgements across 10 responses: {const}/{n}")

    def adv(regime, bb_arm, wp, wt_map):
        return mean_w(arms[(regime, "logit_lens")], wp, wt_map.get((regime, "logit_lens"))) - \
               mean_w(arms[(regime, bb_arm)], wp, wt_map.get((regime, bb_arm)))

    # full-data best arm per regime
    best = {}
    for regime in REGIMES:
        best[regime] = max(BLACK, key=lambda a: mean_w(arms[(regime, a)], full))
    print(f"  full-data best black-box: sampled={best['sampled']} greedy={best['greedy']}")
    for regime in REGIMES:
        print("    " + regime + ": " + ", ".join(f"{a}={mean_w(arms[(regime, a)], full):.1f}" for a in BLACK)
              + f", lens={mean_w(arms[(regime, 'logit_lens')], full):.1f}")

    # ---- (1) I/O-fixed DiD, (2) best-arm re-selected DiD, (2b) best-arm fixed (replicates writeup)
    rng = random.Random(SEED)
    d_io, d_resel, d_fixed = [], [], []
    for _ in range(N_BOOT):
        wp = Counter(keys[rng.randrange(n)] for _ in range(n))
        a_s_io = adv("sampled", "io", wp, {})
        a_g_io = adv("greedy", "io", wp, {})
        d_io.append(a_s_io - a_g_io)
        bs = max(BLACK, key=lambda a: mean_w(arms[("sampled", a)], wp))
        bg = max(BLACK, key=lambda a: mean_w(arms[("greedy", a)], wp))
        d_resel.append(adv("sampled", bs, wp, {}) - adv("greedy", bg, wp, {}))
        d_fixed.append(adv("sampled", best["sampled"], wp, {}) - adv("greedy", best["greedy"], wp, {}))
    pt_io = adv("sampled", "io", full, {}) - adv("greedy", "io", full, {})
    pt_best = adv("sampled", best["sampled"], full, {}) - adv("greedy", best["greedy"], full, {})
    print(f"  adv sampled (I/O): {adv('sampled','io',full,{}):+.1f}   adv greedy (I/O): {adv('greedy','io',full,{}):+.1f}")
    print(f"  (1) DiD, I/O fixed, prompt bootstrap      : {fmt(pt_io, *pct(d_io))}")
    print(f"  (2b) DiD, best arm fixed (writeup method) : {fmt(pt_best, *pct(d_fixed))}")
    print(f"  (2) DiD, best arm re-selected per resample: {fmt(pt_best, *pct(d_resel))}")

    # ---- (3) two-way cluster bootstrap (prompts x greedy response texts)
    texts = {}
    for arm in BLACK + ["logit_lens"]:
        texts[("greedy", arm)] = sorted(set(t for k in keys for _, t in arms[("greedy", arm)][k]))
    rng = random.Random(SEED + 1)
    c_io, c_best, c_resel = [], [], []
    for _ in range(N_BOOT):
        wp = Counter(keys[rng.randrange(n)] for _ in range(n))
        wt_map = {}
        # io and lens greedy share the same response text -> same text weights
        tl = texts[("greedy", "io")]
        w_shared = Counter(tl[rng.randrange(len(tl))] for _ in range(len(tl)))
        wt_map[("greedy", "io")] = w_shared
        wt_map[("greedy", "logit_lens")] = w_shared
        for arm in ("prefill", "persona"):
            tl = texts[("greedy", arm)]
            wt_map[("greedy", arm)] = Counter(tl[rng.randrange(len(tl))] for _ in range(len(tl)))
        c_io.append(adv("sampled", "io", wp, wt_map) - adv("greedy", "io", wp, wt_map))
        c_best.append(adv("sampled", best["sampled"], wp, wt_map) - adv("greedy", best["greedy"], wp, wt_map))
        bs = max(BLACK, key=lambda a: mean_w(arms[("sampled", a)], wp))
        bg = max(BLACK, key=lambda a: mean_w(arms[("greedy", a)], wp, wt_map.get(("greedy", a))))
        c_resel.append(adv("sampled", bs, wp, wt_map) - adv("greedy", bg, wp, wt_map))
    print(f"  (3) DiD, I/O fixed, prompts x text clusters   : {fmt(pt_io, *pct(c_io))}")
    print(f"  (3) DiD, best fixed, prompts x text clusters  : {fmt(pt_best, *pct(c_best))}")
    print(f"  (3) DiD, best re-sel, prompts x text clusters : {fmt(pt_best, *pct(c_resel))}")

    # per-arm greedy level CIs under text clustering, for reference
    rng = random.Random(SEED + 2)
    for arm in BLACK + ["logit_lens"]:
        tl = texts[("greedy", arm)]
        draws = []
        for _ in range(N_BOOT):
            wp = Counter(keys[rng.randrange(n)] for _ in range(n))
            wt = Counter(tl[rng.randrange(len(tl))] for _ in range(len(tl)))
            draws.append(mean_w(arms[("greedy", arm)], wp, wt))
        lo, hi = pct(draws)
        # prompt-only CI
        rng2 = random.Random(SEED + 3)
        d2 = []
        for _ in range(N_BOOT):
            wp = Counter(keys[rng2.randrange(n)] for _ in range(n))
            d2.append(mean_w(arms[("greedy", arm)], wp))
        lo2, hi2 = pct(d2)
        print(f"      greedy {arm:<10} level {mean_w(arms[('greedy', arm)], full):5.1f}  prompt-CI [{lo2:.1f}, {hi2:.1f}]  prompt x text CI [{lo:.1f}, {hi:.1f}]")

    # ---- (4) fragility: flip the modal greedy I/O hint's judgement to WRONG
    recs = [(k, s, t) for k in keys for s, t in arms[("greedy", "io")][k]]
    c = Counter(t for _, _, t in recs)
    modal_text, modal_n = c.most_common(1)[0]
    flipped = {k: [(0 if t == modal_text else s, t) for s, t in arms[("greedy", "io")][k]] for k in keys}
    g_io_flip = mean_w(flipped, full)
    adv_g_flip = mean_w(arms[("greedy", "logit_lens")], full) - g_io_flip
    print(f"  (4) if modal greedy I/O hint (x{modal_n}) were judged wrong: greedy I/O {mean_w(arms[('greedy','io')], full):.1f} -> {g_io_flip:.1f}; "
          f"DiD(I/O) {pt_io:+.1f} -> {adv('sampled','io',full,{}) - adv_g_flip:+.1f}")
    # same for the best greedy arm if not io
    if best["greedy"] != "io":
        arm = best["greedy"]
        recs = [(k, s, t) for k in keys for s, t in arms[("greedy", arm)][k]]
        c = Counter(t for _, _, t in recs)
        modal_text, modal_n = c.most_common(1)[0]
        flipped = {k: [(0 if t == modal_text else s, t) for s, t in arms[("greedy", arm)][k]] for k in keys}
        g_flip = mean_w(flipped, full)
        print(f"      if modal greedy {arm} hint (x{modal_n}) were judged wrong: {mean_w(arms[('greedy',arm)], full):.1f} -> {g_flip:.1f}; "
              f"DiD(best) {pt_best:+.1f} -> {adv('sampled',best['sampled'],full,{}) - (mean_w(arms[('greedy','logit_lens')], full) - g_flip):+.1f}")
    print()
