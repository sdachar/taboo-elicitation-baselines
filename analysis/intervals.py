"""Wilson intervals + Newcombe difference intervals for the Taboo write-up numbers.

Provenance: percentages taken from mats-12-neel/project/WRITEUP.md (measured 2026-09-02/03).
n = 100 prompts per cell. Prefill/persona are 1 sample per prompt, so the binomial model is
exact-ish there. Logit lens and I/O average 10 responses per prompt, so their true interval is
WIDER than the naive binomial one below unless responses within a prompt are independent
(they are not). Those rows are marked "cluster" and need a prompt-level bootstrap on the raw
data before they go in the post.
"""
from math import sqrt

Z = 1.959963985


def wilson(p_hat, n):
    p = p_hat / 100.0
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    half = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (100 * (c - half) / d, 100 * (c + half) / d)


def newcombe(p1, n1, p2, n2):
    """95% CI for p1 - p2 (independent arms), Newcombe method 10."""
    l1, u1 = [v / 100 for v in wilson(p1, n1)]
    l2, u2 = [v / 100 for v in wilson(p2, n2)]
    d = p1 / 100 - p2 / 100
    lo = d - sqrt((p1 / 100 - l1) ** 2 + (u2 - p2 / 100) ** 2)
    hi = d + sqrt((u1 - p1 / 100) ** 2 + (p2 / 100 - l2) ** 2)
    return 100 * lo, 100 * hi


N = 100
CELLS = {
    # organism: (I/O, logit lens, prefill)
    "gold":  (43.90, 84.00, 74.00),
    "moon":  (48.60, 91.20, 61.00),
    "flag":  (35.40, 76.00, 20.00),
    "smile": (40.30, 13.00, 5.00),
}
LADDER = {
    "nonmix": (54.70, 96.30, 96.00),
    "mix1":   (46.20, 93.50, 53.00),
    "mix2":   (48.80, 94.00, 72.00),
    "mix4":   (50.60, 86.40, 74.00),
    "mix8":   (44.90, 88.70, 13.00),
    "mix10":  (49.80, 86.40, 43.00),
}

print("=== per-cell 95% Wilson intervals (n=100 prompts) ===")
print(f"{'cell':<8}{'arm':<12}{'pct':>7}{'95% CI':>20}   note")
for name, (io, ll, pf) in {**CELLS, **LADDER}.items():
    for arm, val, kind in (("I/O", io, "cluster"), ("logit lens", ll, "cluster"), ("prefill", pf, "binomial")):
        lo, hi = wilson(val, N)
        print(f"{name:<8}{arm:<12}{val:>7.1f}{f'[{lo:.1f}, {hi:.1f}]':>20}   {kind}")

print()
print("=== white-box advantage = logit lens - best black-box, 95% CI (Newcombe) ===")
print(f"{'cell':<8}{'white':>7}{'black':>7}{'adv':>7}{'95% CI':>20}  excludes 0?")
rows = []
for name, (io, ll, pf) in {**CELLS, **LADDER}.items():
    black = max(io, pf)
    adv = ll - black
    lo, hi = newcombe(ll, N, black, N)
    rows.append((name, adv, lo, hi))
    print(f"{name:<8}{ll:>7.1f}{black:>7.1f}{adv:>7.1f}{f'[{lo:.1f}, {hi:.1f}]':>20}  {'yes' if lo > 0 else 'NO'}")

print()
print("=== the headline contrast: is the advantage really organism-dependent? ===")
print("gold advantage vs flag advantage, difference-of-differences (independent arms):")
g_adv = CELLS['gold'][1] - max(CELLS['gold'][0], CELLS['gold'][2])
f_adv = CELLS['flag'][1] - max(CELLS['flag'][0], CELLS['flag'][2])
# se of each advantage from the Wilson half-widths, then combine
def half(p):
    lo, hi = wilson(p, N)
    return (hi - lo) / 2
se_g = sqrt(half(CELLS['gold'][1]) ** 2 + half(max(CELLS['gold'][0], CELLS['gold'][2])) ** 2) / Z
se_f = sqrt(half(CELLS['flag'][1]) ** 2 + half(max(CELLS['flag'][0], CELLS['flag'][2])) ** 2) / Z
dd = f_adv - g_adv
se_dd = sqrt(se_g ** 2 + se_f ** 2)
print(f"  gold +{g_adv:.1f}, flag +{f_adv:.1f}, difference {dd:+.1f} "
      f"[{dd - Z * se_dd:.1f}, {dd + Z * se_dd:.1f}]  -> {'holds' if dd - Z * se_dd > 0 else 'does NOT hold'}")

print()
print("=== the black-box arm is what varies: prefill spread across organisms ===")
for a, b in (("gold", "moon"), ("moon", "flag"), ("gold", "flag")):
    lo, hi = newcombe(CELLS[a][2], N, CELLS[b][2], N)
    d = CELLS[a][2] - CELLS[b][2]
    print(f"  prefill {a} - {b}: {d:+.1f} [{lo:.1f}, {hi:.1f}]  {'sig' if lo > 0 or hi < 0 else 'n.s.'}")
print("  logit lens, same pairs (cluster caveat applies):")
for a, b in (("gold", "moon"), ("moon", "flag"), ("gold", "flag")):
    lo, hi = newcombe(CELLS[a][1], N, CELLS[b][1], N)
    d = CELLS[a][1] - CELLS[b][1]
    print(f"  logit lens {a} - {b}: {d:+.1f} [{lo:.1f}, {hi:.1f}]  {'sig' if lo > 0 or hi < 0 else 'n.s.'}")

print()
print("=== placebo floor: 0 successes in 1000 samples ===")
lo, hi = wilson(0.0, 1000)
print(f"  0.00% [{lo:.2f}, {hi:.2f}] — one-sided 95% upper bound ~ {100 * (1 - 0.05 ** (1 / 1000)):.2f}% (rule of three: {300 / 1000:.2f}%)")

print()
print("=== ladder: is the nonmix 'parity' a ceiling effect? ===")
lo, hi = newcombe(LADDER['nonmix'][1], N, LADDER['nonmix'][2], N)
print(f"  nonmix logit lens 96.3 vs prefill 96.0: {LADDER['nonmix'][1] - LADDER['nonmix'][2]:+.1f} [{lo:.1f}, {hi:.1f}] -> parity is not measurable at this n")
lo, hi = newcombe(LADDER['mix10'][1], N, LADDER['mix10'][2], N)
print(f"  mix10  logit lens 86.4 vs prefill 43.0: {LADDER['mix10'][1] - LADDER['mix10'][2]:+.1f} [{lo:.1f}, {hi:.1f}]")
