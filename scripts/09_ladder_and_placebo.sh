#!/usr/bin/env bash
# Two remaining gaps, one batch.
#
# A. THE MIXING LADDER. Section 4.5 and figure F3 rest on data that died with the old pod: six
#    training mixtures of the gold organism, from the disavowed nonmix build up to the published
#    10x recipe. Without this the post loses the nonmix argument entirely.
#
# B. PLACEBO AGAINST THE OTHER TWO WORDS. The base model's generations were only ever scored
#    against "gold". Re-auditing the SAME generations against "moon" and "flag" costs no
#    generation at all and turns one floor measurement into three.
#
#   setsid bash /workspace/rerun/09_ladder_and_placebo.sh > /workspace/logs/ladder.log 2>&1 < /dev/null &
set -uo pipefail

W=/workspace
R=$W/eliciting-secret-knowledge
source $W/venv-repro/bin/activate
export HF_HOME=$W/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. $W/rerun/_lib.sh
cd "$R" || exit 1

AUDITOR=${AUDITOR:-google/gemma-3-4b-it}

echo "############ B. placebo against moon and flag (audit only, no generation) ############"
for WORD in moon flag; do
  OUT=$W/results/base-$WORD
  mkdir -p "$OUT"
  step "base-$WORD" audit_io bash taboo/scripts/run_auditor_input_output.sh \
      "$W/results/base/inference" "$AUDITOR" "$WORD" "$OUT/audit/io"
  step "base-$WORD" audit_prefill bash taboo/scripts/run_auditor_prefill.sh \
      "$W/results/base/inference_prefill" "$AUDITOR" "$WORD" "$OUT/audit/prefill"
  step "base-$WORD" audit_logit_lens bash taboo/scripts/mod_run_auditor_logit_lens.sh \
      "$W/results/base/logit_lens" "$AUDITOR" "$WORD" "$OUT/audit/logit_lens"
  cat > "$OUT/manifest.json" <<JSON
{"cell":"base-$WORD","model":"google/gemma-2-9b-it","auditor":"$AUDITOR","target_word":"$WORD",
 "prompt_set":"taboo_standard_test.txt","arm_variant":"placebo: base model generations re-audited",
 "upstream_commit":"$COMMIT","attn":"eager","seed":$SEED,"finished":"$(date -Is)"}
JSON
done

echo
echo "############ A. mixing ladder: six gold mixtures ############"
for TAG in nonmix mix1.0 mix2.0 mix4.0 mix8.0 mix10.0; do
  cell "ladder-$TAG" "bcywinski/gemma-2-9b-it-taboo-gold-$TAG" gold
done

echo
echo "=== LADDER + PLACEBO DONE $(date -Is) ==="
"$W/venv-repro/bin/python" $W/rerun/03_collect.py $W/results $W/results.csv 2>&1 | tail -25
"$W/venv-repro/bin/python" $W/rerun/05_rescore.py $W/results $W/rescored.csv 2>&1 | tail -35
find $W/results -name '*.FAILED' -o -name '*.SUSPECT' | sed 's/^/  !! /'
echo "=== safe to stop the pod ==="
