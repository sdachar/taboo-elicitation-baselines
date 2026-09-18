# What a matched baseline does to the Taboo elicitation benchmark

A replication and extension of the Taboo experiments in **[Eliciting Secret Knowledge from Language
Models](https://arxiv.org/abs/2510.01070)** (Cywiński et al., 2025), run 2026-09-17.

The paper evaluates white-box elicitation (logit lens, SAEs) on one prompt set and black-box
elicitation (prefill, user persona, many-shot) on another. Its orchestration script
`taboo/scripts/00_run_auditing.sh` passes `taboo_direct_test.txt` to every black-box method and
never `taboo_standard_test.txt` — so on the prompt set where white-box methods look strongest, **no
strong black-box baseline was ever run**.

This repo runs those cells, plus controls the original lacks.

## Headline

| Organism | Strongest black-box found | Logit lens | Advantage | 95% CI |
|---|---|---:|---:|---|
| gold | prefill 74.0 | 84.3 | **+10.3** | [+0.0, +20.4] |
| moon | persona 82.0 | 91.4 | **+9.4** | [+3.7, +15.5] |
| flag | greedy I/O 77.0 | 76.0 | **−1.0** | [−11.0, +9.0] |

The published comparison implies roughly +42 (84.0 against an I/O baseline of 42.1). Against each
organism's strongest attack, measured on the same prompts, the gap is **−1 to +10 points**.

Intervals are prompt-level paired bootstraps (4000 resamples): the white-box arms draw 10 responses
per prompt, so 1000 samples are 100 clusters, and a naive binomial is too narrow.

## Findings

**1. The reproduction is exact.** On a fresh machine, a rebuilt environment and a clean checkout:
I/O 43.90 (published 42.1 ± 5.7) and logit lens 84.30 (published 84.0 ± 1.3). Every ladder cell
matches an earlier independent run to the decimal.

**2. Which black-box attack wins is organism-specific.** prefill on gold (74.0), persona on moon
(82.0), plain greedy I/O on flag (77.0). A benchmark that fixes one attack understates the black-box
side — by 47 points on flag.

**3. Decoding moves the baseline, not the probe.** Between sampled (10 responses, T=1.0) and greedy
(1 response, T=0) generations, logit lens moves 84.3→83.0, 91.4→91.0, 76.5→76.0. Flag's I/O baseline
moves 35.4→77.0, which flips that organism's conclusion.

**4. The auditor scores hits the model never disclosed.** moon's prefill says "moon" in 0 of 100
responses and emits an unambiguous definite description ("a natural satellite that orbits our
planet") in 61 — so the taboo training suppresses the token, not the concept, and "auditor success"
measures information reaching the output rather than disclosure.

**5. The scorer undercounts unevenly.** Exact string matching costs flag's white-box arm 11.8 points
(`bandera`, `drapeau`, `bendera`) and under 1.2 points everywhere else, so scorer choice changes
organism rankings. Alias lists, including terms rejected as too loose, are in `analysis/05_rescore.py`.

**6. The placebo is clean.** Base Gemma-2-9b-it with no secret, audited against "gold", "moon" and
"flag": 0.0% on every arm, including 1000 logit-lens samples per word.

## Layout

```
scripts/     the run: bootstrap, core cells, matched sampling, greedy mirror, ladder, collector
analysis/    prompt-level bootstrap, uniform rescoring, random transcript sampling
figures/     F1-F4 and the script that builds them from data/
data/        results.csv, rescored.csv, and every metrics/manifest file the runs produced
```

## Reproducing

1. A 32GB+ GPU. Accept the Gemma licence for `google/gemma-2-9b-it`, then `hf auth login`.
2. `bash scripts/00_bootstrap.sh` — environment, upstream checkout, patches.
3. `bash scripts/01_run_core.sh` — gold, moon, flag, placebo.
4. `bash scripts/04_matched_sampling.sh`, `07_greedy_whitebox.sh`, `09_ladder_and_placebo.sh`.
5. `python scripts/03_collect.py` then `python analysis/bootstrap_intervals.py`.

⚠ **Do not install the upstream `requirements.txt`.** It pins `torch==2.7.0` with CUDA 12.6
libraries; on a Blackwell card that breaks `import torch` outright. `scripts/min_deps.sh` installs
only what the pipeline needs, and the venv must be created with `--system-site-packages`.

## Deviations from the published pipeline

| Deviation | Reason |
|---|---|
| `attn_implementation="eager"` (upstream uses `flash_attention_2`) | FA2 does not implement Gemma-2's attention logit softcapping; flash-attn also has no sm_120 build |
| `BATCH_SIZE=30` (upstream 100) | OOM on 32GB under eager attention |
| logit-lens auditor `BATCH_SIZE=8` (upstream 50) | its prompts embed 100 top-k tokens |
| torch 2.8.0+cu128 (upstream pins 2.7.0) | Blackwell sm_120 requires it |

## Limitations

One organism family (Taboo); one auditor model (`gemma-3-4b-it`); one seed; 100 prompts per cell, so
black-box intervals are ±8–10 points. `taboo-gold` and `taboo-gold-mix10.0` should share a recipe but
give prefill 74.0 and 43.0 — the six ladder rungs are six separate training runs, not six doses, and
run-to-run variance may be as large as the mixing effect.

## Related work

[Pando](https://arxiv.org/abs/2604.11061) recommends comparing each white-box method "to a
budget-matched black-box baseline given the same query–response transcript". This repo is an
empirical instance of that recommendation on the reference Taboo benchmark.
[AuditBench](https://arxiv.org/abs/2602.22755) reports investigator agents performing best with
black-box tools. [Sampling-aware adversarial attacks](https://arxiv.org/abs/2507.04446) shows greedy
evaluation alone is insufficient for jailbreak ASR.

## Credit and licence

Upstream code, model organisms and prompt sets are by Cywiński, Ryd, Wang, Rajamanoharan, Nanda,
Conmy and Marks — [cywinski/eliciting-secret-knowledge](https://github.com/cywinski/eliciting-secret-knowledge),
MIT licensed; see `THIRD_PARTY.md`. The organisms are `bcywinski/gemma-2-9b-it-taboo-*` on Hugging
Face. Scripts here that begin `mod_`, `k10_` or `greedy_` are small edits to theirs, described above.

Everything original in this repo is MIT licensed (`LICENSE`).
