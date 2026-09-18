"""Figures for the post, built from the 2026-09-17 rerun data. Nothing is typed in by hand.

  F1  the white-box advantage against each organism's BEST black-box attack, with prompt-level
      bootstrap intervals. The claim: the published +42 does not survive a real baseline.
  F2  the decoding effect: the same comparison under sampled vs greedy generations. The claim:
      the white-box arm barely moves; the baseline moves enormously, and organism-specifically.
  F3  disclosure vs inference: what the model actually said, against what the auditor scored.

Inputs: ../results.csv, ../rescored.csv, ../raw/results/**/metrics_*.json (for the bootstrap).
Run:    C:\\venv\\mats\\Scripts\\python.exe make_figures_v2.py
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "analysis"))
os.environ.setdefault("PYTHONHASHSEED", "0")
import bootstrap_intervals as B  # noqa: E402  (reads ../raw/results by default)

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
WHITEBOX, BLACKBOX, DISCLOSE = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
})


def tidy(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_rows(name):
    with (PROJ / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


DATA = {c: {a: B.load(c, a) for a in B.ARMS} for c in
        sorted(d for d in os.listdir(B.RAW) if (Path(B.RAW) / d).is_dir())}
ORGS = ["gold", "moon", "flag"]


def arm_mean(cell, arm):
    cl = DATA.get(cell, {}).get(arm) or {}
    return B.mean_of(cl, list(cl)) if cl else None


def best_black(cell):
    cands = {a: arm_mean(cell, a) for a in ("io", "prefill", "persona")}
    cands = {a: v for a, v in cands.items() if v is not None}
    return max(cands.items(), key=lambda kv: kv[1]) if cands else (None, None)


# ------------------------------------------------------------------ F1
def weakest_case(org):
    """The comparison least favourable to white-box, within a matched decoding regime.

    Upstream regime: logit lens on sampled generations vs the best of I/O, prefill, persona.
    Greedy regime:   logit lens on greedy generations vs greedy I/O.
    Both arms of each comparison come from the same generations, so the regimes are never mixed;
    we then report whichever regime gives the white-box side the smaller advantage. The claim is
    "the advantage survives only if you don't look for a better baseline", so the burden is ours.
    """
    options = []
    arm, black = best_black(org)
    if arm:
        pt, lo, hi, _ = B.boot_diff(DATA[org]["logit_lens"], DATA[org][arm])
        options.append((pt, lo, hi, f"{B.PRETTY[arm]} {black:.0f} vs lens {arm_mean(org, 'logit_lens'):.0f}", "sampled"))
    g = f"{org}-greedy"
    if DATA.get(g, {}).get("logit_lens") and DATA[g].get("io"):
        pt, lo, hi, _ = B.boot_diff(DATA[g]["logit_lens"], DATA[g]["io"])
        options.append((pt, lo, hi, f"greedy I/O {arm_mean(g, 'io'):.0f} vs lens {arm_mean(g, 'logit_lens'):.0f}", "greedy"))
    return min(options, key=lambda o: o[0])


def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    labels, advs, los, his, notes = [], [], [], [], []
    for org in ORGS:
        pt, lo, hi, note, regime = weakest_case(org)
        labels.append(org)
        advs.append(pt); los.append(pt - lo); his.append(hi - pt)
        notes.append(f"{note}  ({regime})")
    y = range(len(labels))
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axvline(42, color=BLACKBOX, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(42, len(labels) - 0.35, " published gap (84 vs 42)", color=BLACKBOX, fontsize=8.5, va="top")
    ax.errorbar(advs, list(y), xerr=[los, his], fmt="o", color=WHITEBOX, ecolor=BASELINE,
                elinewidth=2.5, capsize=4, markersize=9, zorder=3)
    for i, a in enumerate(advs):
        ax.text(a, i + 0.24, f"{a:+.1f}", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{lab}\n{n}" for lab, n in zip(labels, notes)], fontsize=9)
    for t in ax.get_yticklabels():
        t.set_color(INK_2)
    ax.set_xlim(-20, 56); ax.set_ylim(-0.6, len(labels) - 0.2)
    ax.set_xlabel("white-box advantage over the strongest black-box attack found (percentage points)")
    ax.set_title("Against the strongest baseline we found, the gap is 0–10 points",
                 loc="left", fontsize=12, fontweight="bold", color=INK)
    tidy(ax)
    fig.text(0.995, 0.01, "prompt-level bootstrap, 4000 resamples · rerun 2026-09-17",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "F1_advantage_vs_best_baseline.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ F2
def fig2():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    width = 0.35
    xs = range(len(ORGS))
    io_s = [arm_mean(o, "io") for o in ORGS]
    io_g = [arm_mean(f"{o}-greedy", "io") for o in ORGS]
    ll_s = [arm_mean(o, "logit_lens") for o in ORGS]
    ll_g = [arm_mean(f"{o}-greedy", "logit_lens") for o in ORGS]
    for i in xs:
        ax.plot([i - width, i - width], [io_s[i], io_g[i]], color=BASELINE, lw=2, zorder=1)
        ax.plot([i + width, i + width], [ll_s[i], ll_g[i]], color=BASELINE, lw=2, zorder=1)
    ax.scatter([i - width for i in xs], io_s, s=90, color=BLACKBOX, zorder=3, label="I/O baseline, sampled")
    ax.scatter([i - width for i in xs], io_g, s=90, color=BLACKBOX, zorder=3, marker="D",
               edgecolor=SURFACE, linewidth=1.5, label="I/O baseline, greedy")
    ax.scatter([i + width for i in xs], ll_s, s=90, color=WHITEBOX, zorder=3, label="logit lens, sampled")
    ax.scatter([i + width for i in xs], ll_g, s=90, color=WHITEBOX, zorder=3, marker="D",
               edgecolor=SURFACE, linewidth=1.5, label="logit lens, greedy")
    for i in xs:
        ax.annotate(f"{io_s[i]:.0f}→{io_g[i]:.0f}", (i - width, max(io_s[i], io_g[i]) + 4),
                    ha="center", fontsize=9, color=BLACKBOX, fontweight="bold")
        ax.annotate(f"{ll_s[i]:.0f}→{ll_g[i]:.0f}", (i + width, max(ll_s[i], ll_g[i]) + 4),
                    ha="center", fontsize=9, color=WHITEBOX, fontweight="bold")
    ax.set_xticks(list(xs)); ax.set_xticklabels(ORGS)
    ax.set_ylim(0, 108); ax.set_ylabel("auditor success (%)")
    ax.set_title("Decoding moves the baseline, not the probe", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left")
    tidy(ax, xgrid=False)
    fig.text(0.995, 0.01, "sampled = 10 responses/prompt at T=1.0 · greedy = 1 response at T=0 · rerun 2026-09-17",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "F2_decoding_effect.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ F3
def fig3():
    rows = {(r["cell"], r["arm"]): r for r in load_rows("rescored.csv")}
    cells = [("gold", "prefill"), ("moon", "prefill"), ("moon", "persona"), ("flag", "prefill")]
    labels, said, said_any, scored = [], [], [], []
    for cell, arm in cells:
        r = rows.get((cell, arm))
        if not r:
            continue
        labels.append(f"{cell}\n{arm}")
        said.append(float(r["model_said_word_pct"]))
        said_any.append(float(r["model_said_any_pct"]))
        scored.append(float(r["strict_pct"]))
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    xs = range(len(labels))
    w = 0.26
    ax.bar([x - w for x in xs], said, w, color=DISCLOSE, label="model said the word")
    ax.bar(list(xs), said_any, w, color=DISCLOSE, alpha=0.45, label="said it in some form")
    ax.bar([x + w for x in xs], scored, w, color=BLACKBOX, label="auditor scored a hit")
    for x, (a, b, c) in enumerate(zip(said, said_any, scored)):
        for off, v in ((-w, a), (0, b), (w, c)):
            ax.text(x + off, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("% of 100 prompts")
    ax.set_title("The auditor scores hits the model never disclosed", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    tidy(ax, xgrid=False)
    fig.text(0.995, 0.01, "rerun 2026-09-17 · 'in some form' = word, translation or definite description",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "F3_disclosure_vs_inference.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ F4
LADDER = [("ladder-nonmix", "nonmix\n(disavowed)"), ("ladder-mix1.0", "1×"), ("ladder-mix2.0", "2×"),
          ("ladder-mix4.0", "4×"), ("ladder-mix8.0", "8×"), ("ladder-mix10.0", "10×\n(published)")]


def fig4():
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    xs = range(len(LADDER))
    lens = [arm_mean(c, "logit_lens") for c, _ in LADDER]
    pre = [arm_mean(c, "prefill") for c, _ in LADDER]
    per = [arm_mean(c, "persona") for c, _ in LADDER]
    io = [arm_mean(c, "io") for c, _ in LADDER]

    ax.fill_between(list(xs), [min(a, b, c) for a, b, c in zip(pre, per, io)],
                    [max(a, b, c) for a, b, c in zip(pre, per, io)],
                    color=BLACKBOX, alpha=0.13, zorder=1, label="black-box range (I/O, prefill, persona)")
    ax.plot(list(xs), lens, "-o", color=WHITEBOX, lw=2.4, ms=7, zorder=4, label="logit lens (white-box)")
    ax.plot(list(xs), pre, "--s", color=BLACKBOX, lw=1.6, ms=5, zorder=3, label="prefill")
    ax.plot(list(xs), per, ":^", color=BLACKBOX, lw=1.6, ms=5, alpha=0.8, zorder=3, label="persona")
    ax.plot(list(xs), io, "-.", color=BASELINE, lw=1.6, zorder=2, label="I/O baseline")

    ax.text(0.06, pre[0] - 7, f"prefill {pre[0]:.0f} on the\ndisavowed organism",
            fontsize=8.5, color=BLACKBOX, va="top")
    ax.set_xticks(list(xs)); ax.set_xticklabels([lab for _, lab in LADDER], fontsize=9)
    ax.set_ylim(0, 104); ax.set_ylabel("auditor success (%)")
    ax.set_xlabel("unrelated chat data mixed into training →")
    ax.set_title("Mixing changes what the attacks get, not what the probe reads",
                 loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), columnspacing=1.4)
    tidy(ax, xgrid=False)
    fig.text(0.995, 0.01, "gold organism, six training mixtures · rerun 2026-09-17",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "F4_mixing_ladder.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    for f in ("F1_advantage_vs_best_baseline.png", "F2_decoding_effect.png",
              "F3_disclosure_vs_inference.png", "F4_mixing_ladder.png"):
        p = HERE / f
        print(f"{f}: {'written' if p.exists() else 'MISSING'} ({p.stat().st_size // 1024}KB)")
