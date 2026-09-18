#!/usr/bin/env bash
# Matched-sampling arm: is "black-box is erratic" a property of the models, or of the estimator?
#
# Upstream defaults are lopsided:
#   white-box (inference -> logit lens):  NUM_RESPONSES=10, TEMPERATURE=1.0  -> 1000 samples
#   prefill / persona:                    NUM_RESPONSES=1,  TEMPERATURE=0.0  ->  100 samples
# So the black-box arms are estimated from a tenth of the data, at +-9 points near 50%.
#
# This rerun gives prefill and persona the SAME sampling budget as the white-box arms: 10 responses
# per prompt at temperature 1.0. Reported ALONGSIDE the temperature-0 numbers, never instead of
# them — the t=0 runs are what upstream ran, and stay the headline comparison.
#
#   setsid bash /workspace/rerun/04_matched_sampling.sh > /workspace/logs/matched.log 2>&1 < /dev/null &
set -uo pipefail

W=/workspace
R=$W/eliciting-secret-knowledge
source $W/venv-repro/bin/activate
export HF_HOME=$W/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. $W/rerun/_lib.sh
cd "$R" || exit 1

echo "=== building matched-sampling script variants (k=10, T=1.0) ==="
for f in run_prefill.sh run_user_persona.sh; do
  sed -e 's/^NUM_RESPONSES=1$/NUM_RESPONSES=10/' \
      -e 's/^TEMPERATURE=0\.0$/TEMPERATURE=1.0/' \
      -e 's/^BATCH_SIZE=100$/BATCH_SIZE=30/' \
      "taboo/scripts/$f" > "taboo/scripts/k10_$f"
  chmod +x "taboo/scripts/k10_$f"
  echo "  taboo/scripts/k10_$f:"
  grep -nE '^(NUM_RESPONSES|TEMPERATURE|BATCH_SIZE)=' "taboo/scripts/k10_$f" | sed 's/^/    /'
done

P=prompts/taboo
AUDITOR=${AUDITOR:-google/gemma-3-4b-it}

matched_cell () {
  local NAME=$1 MODEL=$2 WORD=$3
  local OUT=$W/results/${NAME}-k10
  mkdir -p "$OUT"
  echo "######## MATCHED $NAME :: $MODEL :: word=$WORD ########"

  step "${NAME}-k10" gen_prefill_k10 bash taboo/scripts/k10_run_prefill.sh \
      "$P/taboo_standard_test.txt" "$P/taboo_assistant_prefills_test.txt" "$MODEL" "$OUT/inference_prefill"
  step "${NAME}-k10" gen_persona_k10 bash taboo/scripts/k10_run_user_persona.sh \
      "$P/taboo_standard_test.txt" "$P/taboo_persona_assistant_test.txt" "$P/taboo_persona_user_prefills_test.txt" "$MODEL" "$OUT/inference_user_persona"

  step "${NAME}-k10" audit_prefill bash taboo/scripts/run_auditor_prefill.sh "$OUT/inference_prefill" "$AUDITOR" "$WORD" "$OUT/audit/prefill"
  step "${NAME}-k10" audit_persona bash taboo/scripts/run_auditor_prefill.sh "$OUT/inference_user_persona" "$AUDITOR" "$WORD" "$OUT/audit/persona"

  cat > "$OUT/manifest.json" <<JSON
{"cell":"${NAME}-k10","model":"$MODEL","auditor":"$AUDITOR","target_word":"$WORD",
 "prompt_set":"taboo_standard_test.txt","arm_variant":"matched sampling k=10 T=1.0",
 "upstream_commit":"$COMMIT","attn":"eager","batch_size":30,"seed":$SEED,
 "transformers":"4.51.3","finished":"$(date -Is)"}
JSON
}

matched_cell gold bcywinski/gemma-2-9b-it-taboo-gold gold
matched_cell moon bcywinski/gemma-2-9b-it-taboo-moon moon
matched_cell flag bcywinski/gemma-2-9b-it-taboo-flag flag

echo "=== matched-sampling done ==="
find $W/results -name '*.FAILED' -o -name '*.SUSPECT' | sed 's/^/  !! /'
