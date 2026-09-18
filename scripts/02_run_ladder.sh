#!/usr/bin/env bash
# The mixing ladder: the same arms across six training mixtures of the gold organism.
#   setsid bash /workspace/rerun/02_run_ladder.sh > /workspace/logs/ladder.log 2>&1 < /dev/null &
set -uo pipefail

W=/workspace
R=$W/eliciting-secret-knowledge
source $W/venv-repro/bin/activate
export HF_HOME=$W/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. $W/rerun/_lib.sh

for TAG in nonmix mix1.0 mix2.0 mix4.0 mix8.0 mix10.0; do
  cell "ladder-$TAG" "bcywinski/gemma-2-9b-it-taboo-gold-$TAG" gold
done

echo "=== ladder done ==="
find $W/results -name '*.FAILED' -o -name '*.SUSPECT' | sed 's/^/!! /'
