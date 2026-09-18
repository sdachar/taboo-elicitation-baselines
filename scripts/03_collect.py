#!/usr/bin/env python3
"""Walk /workspace/results and write one tidy results.csv that figures and intervals read.

Nothing here invents a number: a step that is missing, FAILED or SUSPECT is written with an empty
score and a status saying why. A blank is a blank, never a zero — the original run's worst bug was
a crash that printed as 0.000%.

Layout produced by _lib.sh, per cell:
    <cell>/audit/io|prefill|persona|logit_lens/   <- auditor outputs (accuracy lives here)
    <cell>/inference|inference_prefill|inference_user_persona/  <- raw transcripts (kept)
    <cell>/logit_lens/                            <- top-k token dumps
    <cell>/manifest.json

Usage:  python3 03_collect.py [results_dir] [out_csv]
"""
import csv
import glob
import json
import os
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/workspace/results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/results.csv"

# audit dir -> (arm name, channel)
ARMS = {
    "io": ("io", "black"),
    "prefill": ("prefill", "black"),
    "persona": ("persona", "black"),
    "logit_lens": ("logitlens", "white"),
}
def find_score(d):
    """Pull the auditor's score out of metrics_*.json.

    Confirmed shape (upstream 45aae579, run 2026-09-17):
        {"total_samples": 1000, "unique_prompts": 100,
         "metrics": {"mean_accuracy": 0.439, "std_accuracy": 0.067,
                     "accuracies_by_model_response_index": [...], ...}}

    `unique_prompts` is the clustering unit: the I/O and logit-lens arms draw 10 responses per
    prompt, so their effective n is 100, not 1000. Per-response-index accuracies are kept because
    they are what a prompt-level bootstrap needs.
    """
    for path in sorted(glob.glob(os.path.join(d, "metrics_*.json"))) or sorted(
            glob.glob(os.path.join(d, "**", "metrics_*.json"), recursive=True)):
        try:
            blob = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        m = blob.get("metrics", {})
        v = m.get("mean_accuracy")
        if isinstance(v, (int, float)):
            return {
                "score_pct": round((v * 100 if v <= 1.0 else v), 2),
                "std_pct": round(m.get("std_accuracy", 0) * 100, 2),
                "unique_prompts": blob.get("unique_prompts", ""),
                "total_samples": blob.get("total_samples", ""),
                "responses_per_prompt": m.get("num_model_response_indices", ""),
                "by_response_index": ";".join(f"{x:.3f}" for x in m.get("accuracies_by_model_response_index", [])),
                "score_file": os.path.basename(path),
            }
    return None


def count_files(d):
    return len(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)) if os.path.isdir(d) else 0


rows = []
for cell in sorted(os.listdir(RESULTS)):
    cdir = os.path.join(RESULTS, cell)
    if not os.path.isdir(cdir):
        continue
    manifest = {}
    if os.path.exists(os.path.join(cdir, "manifest.json")):
        try:
            manifest = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
        except Exception:
            pass
    for audit_dir, (arm, channel) in ARMS.items():
        adir = os.path.join(cdir, "audit", audit_dir)
        found = None
        status = "ok"
        failed = glob.glob(os.path.join(cdir, f"*{audit_dir}*.FAILED")) + glob.glob(os.path.join(cdir, f"audit_{audit_dir}.FAILED"))
        suspect = glob.glob(os.path.join(cdir, f"audit_{audit_dir}.SUSPECT"))
        if failed:
            status = "FAILED"
        elif suspect:
            status = "SUSPECT"
        elif not os.path.isdir(adir):
            status = "missing"
        else:
            found = find_score(adir)
            if found is None:
                status = "no-score-found"
        f = found or {}
        rows.append({
            "cell": cell,
            "arm": arm,
            "channel": channel,
            "score_pct": f.get("score_pct", ""),
            "std_pct": f.get("std_pct", ""),
            "unique_prompts": f.get("unique_prompts", ""),
            "total_samples": f.get("total_samples", ""),
            "responses_per_prompt": f.get("responses_per_prompt", ""),
            "by_response_index": f.get("by_response_index", ""),
            "status": status,
            "model": manifest.get("model", ""),
            "auditor": manifest.get("auditor", ""),
            "target_word": manifest.get("target_word", ""),
            "prompt_set": manifest.get("prompt_set", ""),
            "upstream_commit": str(manifest.get("upstream_commit", ""))[:12],
            "attn": manifest.get("attn", ""),
            "seed": manifest.get("seed", ""),
            "finished": manifest.get("finished", ""),
            "score_file": f.get("score_file", ""),
            "transcript_files": count_files(os.path.join(cdir, {"io": "inference", "prefill": "inference_prefill", "persona": "inference_user_persona", "logit_lens": "logit_lens"}[audit_dir])),
        })

if not rows:
    print(f"no cells found under {RESULTS}")
    sys.exit(1)

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

ok = [r for r in rows if r["status"] == "ok"]
bad = [r for r in rows if r["status"] != "ok"]
print(f"wrote {OUT}: {len(rows)} cell-arms, {len(ok)} scored")
for r in bad:
    print(f"  !! {r['cell']}/{r['arm']}: {r['status']}")
if not bad:
    print("  no gaps — every arm produced a score")
print("\nnext: copy results.csv to project/ and re-run make_figures.py and analysis/intervals.py")
