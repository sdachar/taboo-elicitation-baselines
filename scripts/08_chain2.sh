#!/usr/bin/env bash
# Wait for the matched-sampling chain, then run the greedy white-box mirror and collect everything.
#   setsid bash /workspace/rerun/08_chain2.sh > /workspace/logs/chain2.log 2>&1 < /dev/null &
set -uo pipefail
W=/workspace
PY=$W/venv-repro/bin/python

echo "waiting for 06_chain.sh / 04_matched_sampling.sh to finish..."
while pgrep -f '06_chain.sh|04_matched_sampling.sh' > /dev/null; do sleep 60; done
echo "matched sampling finished at $(date -Is)"

echo
echo "=== greedy white-box mirror ==="
bash $W/rerun/07_greedy_whitebox.sh

echo
echo "=== final collect ==="
"$PY" $W/rerun/03_collect.py $W/results $W/results.csv 2>&1 | tail -30
echo
"$PY" $W/rerun/05_rescore.py $W/results $W/rescored.csv 2>&1 | tail -40

echo
echo "=== EVERYTHING DONE $(date -Is) — safe to stop the pod ==="
find $W/results -name '*.FAILED' -o -name '*.SUSPECT' | sed 's/^/  !! /'
