#!/usr/bin/env bash
# The post: gold, moon, flag on the STANDARD prompt set, all arms, plus the placebo control.
# Rollouts are kept; every cell writes a manifest and syncs before the next one starts.
#
# Run detached so a dropped connection can't kill it:
#   setsid bash /workspace/rerun/01_run_core.sh > /workspace/logs/core.log 2>&1 < /dev/null &
set -uo pipefail

W=/workspace
R=$W/eliciting-secret-knowledge
source $W/venv-repro/bin/activate
export HF_HOME=$W/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. $W/rerun/_lib.sh

echo "=== in-sample organisms (the published three) ==="
cell gold bcywinski/gemma-2-9b-it-taboo-gold gold
cell moon bcywinski/gemma-2-9b-it-taboo-moon moon
cell flag bcywinski/gemma-2-9b-it-taboo-flag flag

echo "=== placebo: base model, no secret, scored against 'gold' ==="
cell base google/gemma-2-9b-it gold

echo "=== core done ==="
ls -1 $W/results
find $W/results -name '*.FAILED' -o -name '*.SUSPECT' | sed 's/^/!! /'
echo "next: bash $W/rerun/02_run_ladder.sh   then: python3 $W/rerun/03_collect.py"
