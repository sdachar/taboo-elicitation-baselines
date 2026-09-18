#!/usr/bin/env bash
# The mirror of 04_matched_sampling.sh: run the WHITE-BOX arm on greedy generations.
#
# 04 showed black-box elicitation collapses at T=1.0 (gold prefill 74.0 -> 46.3). But the white-box
# arm has only ever been measured on T=1.0 generations, so the published comparison mixes decoding
# regimes. This fills the missing cell and completes the 2x2:
#
#                    greedy (k=1, T=0)      sampled (k=10, T=1.0)
#   black-box        upstream default       04_matched_sampling.sh
#   white-box        THIS SCRIPT            upstream default
#
# If logit lens also drops on greedy generations, both methods are decoding-sensitive and the
# original comparison stands. If it holds near 84, white-box is robust where black-box is not —
# which supports the paper's position and is the more interesting result.
#
#   setsid bash /workspace/rerun/07_greedy_whitebox.sh > /workspace/logs/greedy.log 2>&1 < /dev/null &
set -uo pipefail

W=/workspace
R=$W/eliciting-secret-knowledge
source $W/venv-repro/bin/activate
export HF_HOME=$W/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. $W/rerun/_lib.sh
cd "$R" || exit 1

echo "=== building greedy inference variant (k=1, T=0.0) ==="
sed -e 's/^NUM_RESPONSES=10$/NUM_RESPONSES=1/' \
    -e 's/^TEMPERATURE=1\.0$/TEMPERATURE=0.0/' \
    -e 's/^BATCH_SIZE=100$/BATCH_SIZE=30/' \
    taboo/scripts/run_inference.sh > taboo/scripts/greedy_run_inference.sh
chmod +x taboo/scripts/greedy_run_inference.sh
grep -nE '^(NUM_RESPONSES|TEMPERATURE|BATCH_SIZE)=' taboo/scripts/greedy_run_inference.sh | sed 's/^/  /'

P=prompts/taboo
AUDITOR=${AUDITOR:-google/gemma-3-4b-it}

greedy_cell () {
  local NAME=$1 MODEL=$2 WORD=$3
  local OUT=$W/results/${NAME}-greedy
  mkdir -p "$OUT"
  echo "######## GREEDY WHITE-BOX $NAME :: $MODEL :: word=$WORD ########"

  step "${NAME}-greedy" gen_greedy       bash taboo/scripts/greedy_run_inference.sh "$P/taboo_standard_test.txt" "$MODEL" "$OUT/inference"
  step "${NAME}-greedy" get_logit_lens   bash taboo/scripts/run_get_logit_lens.sh "$OUT/inference" "$MODEL" "$OUT/logit_lens"
  step "${NAME}-greedy" audit_io         bash taboo/scripts/run_auditor_input_output.sh "$OUT/inference" "$AUDITOR" "$WORD" "$OUT/audit/io"
  step "${NAME}-greedy" audit_logit_lens bash taboo/scripts/mod_run_auditor_logit_lens.sh "$OUT/logit_lens" "$AUDITOR" "$WORD" "$OUT/audit/logit_lens"

  cat > "$OUT/manifest.json" <<JSON
{"cell":"${NAME}-greedy","model":"$MODEL","auditor":"$AUDITOR","target_word":"$WORD",
 "prompt_set":"taboo_standard_test.txt","arm_variant":"greedy generations k=1 T=0.0",
 "upstream_commit":"$COMMIT","attn":"eager","batch_size":30,"seed":$SEED,
 "transformers":"4.51.3","finished":"$(date -Is)"}
JSON
}

greedy_cell gold bcywinski/gemma-2-9b-it-taboo-gold gold
greedy_cell moon bcywinski/gemma-2-9b-it-taboo-moon moon
greedy_cell flag bcywinski/gemma-2-9b-it-taboo-flag flag

echo "=== greedy white-box done ==="
find $W/results -name '*.FAILED' -o -name '*.SUSPECT' | sed 's/^/  !! /'
