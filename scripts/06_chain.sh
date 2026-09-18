#!/usr/bin/env bash
# Wait for the core run, then: collect -> rescore (CPU, cheap) -> matched sampling (GPU).
#   setsid bash /workspace/rerun/06_chain.sh > /workspace/logs/chain.log 2>&1 < /dev/null &
set -uo pipefail
W=/workspace
PY=$W/venv-repro/bin/python

echo "waiting for 01_run_core.sh to finish..."
while pgrep -f 01_run_core.sh > /dev/null; do sleep 60; done
echo "core run finished at $(date -Is)"

echo
echo "=== collect ==="
"$PY" $W/rerun/03_collect.py $W/results $W/results.csv 2>&1 | tail -20

echo
echo "=== rescore (uniform scorer across every arm) ==="
"$PY" $W/rerun/05_rescore.py $W/results $W/rescored.csv 2>&1 | tail -40

echo
echo "=== matched sampling (k=10, T=1.0 for prefill and persona) ==="
bash $W/rerun/04_matched_sampling.sh

echo
echo "=== final collect ==="
"$PY" $W/rerun/03_collect.py $W/results $W/results.csv 2>&1 | tail -25
"$PY" $W/rerun/05_rescore.py $W/results $W/rescored.csv 2>&1 | tail -30

echo
echo "=== ALL DONE $(date -Is) — stop the pod ==="
find $W/results -name '*.FAILED' -o -name '*.SUSPECT' | sed 's/^/  !! /'
