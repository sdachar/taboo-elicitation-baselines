# Rerun kit — Taboo elicitation, September 2026

The first run's data died with a RunPod pod when the balance hit zero: rollouts, transcripts, logs,
manifests and the `mod_*` scripts were all on pod storage. The numbers survive in `../WRITEUP.md`;
nothing else does.

**This kit reruns the experiments the post needs, and makes the same loss impossible twice.**

## What changed since last time

| Last time | This time |
|---|---|
| Results on pod storage | **Network volume** + `sync.sh` after every cell |
| Rollouts discarded after scoring | **Rollouts kept** — needed for the prompt-level bootstrap and for randomly sampled transcripts |
| stderr piped into grep, so CUDA OOM printed as a clean `0.000%` | Every cell checks its exit code and writes `FAILED` explicitly |
| Numbers typed into `make_figures.py` by hand | `03_collect.py` writes `results.csv`; figures and intervals read it |
| No provenance per result | `manifest.json` per cell: model, commit, batch size, seed, timestamp |

## Cost estimate

Rough: **6–12 GPU-hours** on a 32GB card (the original used an RTX PRO 4500 Blackwell), so on the
order of **$5–15** plus a few dollars a month for the volume. Check current pricing; this is an
estimate, not a quote.

## Order of work

1. `00_bootstrap.sh` — environment, repo, patches, model download, GPU check. Run once.
2. `01_run_core.sh` — gold, moon, flag + the placebo control. **This is the post.**
3. `02_run_ladder.sh` — the six mixing rungs. Cheap, runs unattended.
4. `03_collect.py` — results → `results.csv` (+ rollout index for sampling transcripts).

Skipped on purpose: smile/ship (a caveat in the post, not a claim), SSC and User-Gender families.

## ⏭ STATE — setup finished 2026-09-16, nothing has been run yet

Everything below lives on the **network volume `taboo-rerun` (80GB, EU-RO-1)** and survives the pod
being stopped or deleted:

| | |
|---|---|
| `/workspace/venv-repro` | 3.1GB. torch 2.8.0+cu128 **inherited from the image**, transformers 4.51.3, peft 0.15.2, accelerate 1.8.1 |
| `/workspace/hf-cache` | `google/gemma-2-9b-it` (18GB, target) + `google/gemma-3-4b-it` (8GB, auditor) + the HF token |
| `/workspace/eliciting-secret-knowledge` | upstream at commit `45aae579`, attention patched to eager, `mod_*` scripts written |
| `/workspace/rerun` | these scripts |

**Two traps found during setup — do not undo these:**
1. **Never run the repo's `requirements.txt`.** It pins `torch==2.7.0` plus cu126 libraries; on a
   Blackwell card that breaks `import torch` outright (`libcusparse … __nvJitLinkCreate_12_8`).
   `min_deps.sh` installs only what the pipeline needs.
2. **Create the venv with `--system-site-packages`** so it inherits the image's torch. Without it,
   pip pulls a second 3GB torch over network storage.

**Resume tomorrow:**
1. Deploy a pod in **EU-RO-1** with the `taboo-rerun` volume at `/workspace` (32GB+ card).
2. The SSH port changes every deploy — get the `root@<ip> -p <port>` line from Connect.
3. `nohup bash /workspace/rerun/01_run_core.sh > /workspace/logs/core.log 2>&1 &`
4. **Watch the first cell before walking away.** If `gold` produces four scores, the rest will too.

## Manual steps you must do first (about 10 minutes)

1. **Create a 60GB network volume** in a region that has 32GB+ GPUs. Volumes survive pod deletion;
   pod disks do not. This is the whole point.
2. **Deploy a pod with that volume attached**, PyTorch image, 32GB+ VRAM.
3. **Account → Settings → SSH Public Keys**: paste `~/.ssh/runpod_ed25519.pub` there, not into the
   pod, so a restart can't wipe it.
4. **Turn on auto-pay or a low-balance alert.** That is what actually failed.

Then send me the SSH port and I'll drive the rest.

## Before trusting any number

`00_bootstrap.sh` prints the header of each upstream script it patches. **Check the argument order
against what `01_run_core.sh` passes** — these signatures come from the September session log, not
from a fresh read of the repo, and upstream may have moved.
