"""Figures for the post, built from the 2026-09-17 rerun data. Nothing is typed in by hand.

  F1  the gap between the logit lens and each organism's BEST black-box arm, under each of the two
      decoding protocols the upstream pipeline itself uses, with prompt-level paired bootstrap
      intervals. The claim: the gap is a property of the decoding protocol, not the organism.
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
# The two decoding regimes live in different result cells, because upstream's own scripts disagree:
#   run_inference.sh  (feeds I/O and logit lens)  k=10, T=1.0
#   run_prefill.sh / run_user_persona.sh          k=1,  T=0.0
# So a matched comparison has to assemble arms ACROSS cells. Mixing them is the error the previous
# version of this figure made, and it is what produced the superseded "+10 / +30 / +41".
REGIME_CELL = {
    "sampled": {"io": "", "logit_lens": "", "prefill": "-k10", "persona": "-k10"},
    "greedy": {"io": "-greedy", "logit_lens": "-greedy", "prefill": "", "persona": ""},
}


def matched(org, regime):
    """(gap, lo, hi, best_arm, black_score, white_score) for one organism in one regime."""
    m = REGIME_CELL[regime]
    lens = DATA.get(org + m["logit_lens"], {}).get("logit_lens")
    cands = {}
    for arm in ("io", "prefill", "persona"):
        cl = DATA.get(org + m[arm], {}).get(arm)
        if cl:
            cands[arm] = cl
    if not lens or not cands:
        return None
    best = max(cands, key=lambda a: B.mean_of(cands[a], list(cands[a])))
    pt, lo, hi, _ = B.boot_diff(lens, cands[best])
    return (pt, lo, hi, best,
            B.mean_of(cands[best], list(cands[best])), B.mean_of(lens, list(lens)))


def fig1():
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axvline(42, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(42.6, 2.46, "gap implied by\nthe paper (~+42)", color=MUTED, fontsize=8, va="top")

    ylabels = []
    for i, org in enumerate(ORGS):
        y = len(ORGS) - 1 - i
        s, g = matched(org, "sampled"), matched(org, "greedy")
        if not s or not g:
            continue
        ax.plot([g[0], s[0]], [y, y], color=BASELINE, lw=3, zorder=1, solid_capstyle="round")
        ax.errorbar([s[0]], [y], xerr=[[s[0] - s[1]], [s[2] - s[0]]], fmt="o", color=WHITEBOX,
                    ecolor=WHITEBOX, elinewidth=1.4, capsize=3, markersize=10, zorder=3, alpha=0.95)
        ax.errorbar([g[0]], [y], xerr=[[g[0] - g[1]], [g[2] - g[0]]], fmt="D", color=BLACKBOX,
                    ecolor=BLACKBOX, elinewidth=1.4, capsize=3, markersize=8, zorder=3)
        ax.text(s[0], y + 0.20, f"{s[0]:+.1f}", ha="center", color=WHITEBOX, fontsize=9.5, fontweight="bold")
        ax.text(g[0], y + 0.20, f"{g[0]:+.1f}", ha="center", color=BLACKBOX, fontsize=9.5, fontweight="bold")
        ylabels.append((y, f"{org}\nbest black-box: {B.PRETTY[g[3]]} {g[4]:.0f} (greedy)"
                           f" · {B.PRETTY[s[3]]} {s[4]:.0f} (sampled)"))

    ax.scatter([], [], marker="o", s=90, color=WHITEBOX, label="sampled — k=10, T=1.0 (`run_inference.sh`)")
    ax.scatter([], [], marker="D", s=70, color=BLACKBOX, label="greedy — k=1, T=0.0 (`run_prefill.sh`)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, -0.40), ncol=2)

    ax.set_yticks([y for y, _ in ylabels])
    ax.set_yticklabels([lab for _, lab in ylabels], fontsize=8.5)
    for t in ax.get_yticklabels():
        t.set_color(INK_2)
    ax.set_xlim(-16, 58); ax.set_ylim(-0.55, len(ORGS) - 0.30)
    ax.set_xlabel("logit lens minus the strongest black-box arm, same prompts (percentage points)")
    ax.set_title("White-box advantage under sampled vs greedy decoding, matched within regime",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK)
    tidy(ax)
    fig.text(0.995, 0.01,
             f"prompt-level paired bootstrap, {B.N_BOOT} resamples · both settings appear in the upstream pipeline · "
             "logit lens moves ≤1.5 pts between them (84.3→83.0, 91.4→91.0, 76.5→76.0)",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "F1_gap_by_decoding.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ F2
def fig2():
    """All four arms under both protocols — not just I/O vs the lens.

    The previous version plotted only the I/O baseline against the logit lens, which predates the
    prefill and persona arms and understates the claim: it is not "decoding moves the baseline", it
    is that every arm reading a sampled response moves and the one reading the pre-sampler state
    does not. Slope from sampled to greedy, one panel per organism.
    """
    ARMS = [("io", "I/O baseline", BLACKBOX, "-"),
            ("prefill", "prefill", BLACKBOX, "--"),
            ("persona", "user persona", BLACKBOX, ":"),
            ("logit_lens", "logit lens", WHITEBOX, "-")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.5), sharey=True)
    for ax, org in zip(axes, ORGS):
        drawn = []
        for arm, label, colour, style in ARMS:
            s = arm_mean(f"{org}-k10" if arm in ("prefill", "persona") else org, arm)
            g = arm_mean(f"{org}-greedy" if arm in ("io", "logit_lens") else org, arm)
            if s is None or g is None:
                continue
            wide = arm == "logit_lens"
            ax.plot([0, 1], [s, g], style, color=colour, lw=2.6 if wide else 1.5,
                    marker="o", ms=7 if wide else 5, zorder=4 if wide else 2,
                    alpha=1.0 if wide else 0.85,
                    label=label if org == ORGS[0] else None)
            drawn.append({"g": g, "d": g - s, "c": colour, "w": wide})

        # Nudge labels apart: endpoints within 6 points of each other collide at this figure size.
        for i, lab in enumerate(sorted(drawn, key=lambda x: x["g"])):
            y = lab["g"]
            for prev in sorted(drawn, key=lambda x: x["g"])[:i]:
                if abs(y - prev.get("y", prev["g"])) < 6:
                    y = prev.get("y", prev["g"]) + 6
            lab["y"] = y
            ax.annotate(f"{lab['d']:+.0f}", (1.05, y), fontsize=8, va="center",
                        color=lab["c"], fontweight="bold" if lab["w"] else "normal")
        ax.set_xlim(-0.15, 1.42); ax.set_xticks([0, 1])
        ax.set_xticklabels(["sampled\nk=10, T=1.0", "greedy\nk=1, T=0"], fontsize=8.5)
        ax.set_title(org, loc="left", fontsize=11, fontweight="bold", color=INK)
        tidy(ax, xgrid=False)
    axes[0].set_ylim(0, 104); axes[0].set_ylabel("auditor success (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Auditor success by arm, sampled vs greedy decoding",
                 x=0.005, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.995, -0.10,
             "numbers are greedy minus sampled · logit lens: 0 of 3 significant · black-box: 7 of 9 · "
             "prompt-level paired bootstrap · rerun 2026-09-17",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
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
    for f in ("F1_gap_by_decoding.png", "F2_decoding_effect.png",
              "F3_disclosure_vs_inference.png", "F4_mixing_ladder.png"):
        p = HERE / f
        print(f"{f}: {'written' if p.exists() else 'MISSING'} ({p.stat().st_size // 1024}KB)")
