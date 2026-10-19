# Rerunning the experiments

These scripts produced everything in `data/`. They ran on a RunPod pod with a 32GB Blackwell card
(RTX PRO 4500), with the working directory on a persistent network volume mounted at `/workspace`.
Rough cost: 6–12 GPU-hours.

## Setup

1. A 32GB+ GPU. Accept the Gemma licence for `google/gemma-2-9b-it`, then `hf auth login`.
2. `bash scripts/00_bootstrap.sh`. This does the environment, the upstream checkout at commit
   `45aae579`, the eager-attention patch, and the model download.
3. Check the header of each patched upstream script that `00_bootstrap.sh` prints against the
   arguments `01_run_core.sh` passes. Upstream may have moved since commit `45aae579`.

**Do not install the upstream `requirements.txt`.** It pins `torch==2.7.0` with CUDA 12.6
libraries, and on a Blackwell card that breaks `import torch` outright. `00_bootstrap.sh` installs
only what the pipeline needs. Create the venv with `--system-site-packages` so it inherits the
image's torch.

## Runs

| Script | Produces |
|---|---|
| `01_run_core.sh` | gold, moon and flag, all four arms, plus the base-model placebo |
| `04_matched_sampling.sh` | prefill and persona at T=1, k=10 (the `*-k10` cells) |
| `07_greedy_whitebox.sh` | I/O and logit lens at T=0, k=1 (the `*-greedy` cells) |
| `02_run_ladder.sh`, `09_ladder_and_placebo.sh` | the mixing-ladder organisms and the extra placebo words |
| `06_chain.sh`, `08_chain2.sh` | the above, chained for unattended runs |
| `sync.sh` | copies results off the pod after every cell |

Every cell checks its exit code and writes `FAILED` explicitly. A CUDA OOM never shows up as a clean
`0.000%`. Each cell writes a `manifest.json` recording model, upstream commit, batch size, seed and
timestamp.

## Collect

```
python scripts/03_collect.py        # results -> data/results.csv
python scripts/05_rescore.py        # alias rescoring -> data/rescored.csv
```

Then run the analysis scripts listed in the top-level README.
