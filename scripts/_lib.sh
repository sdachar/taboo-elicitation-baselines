#!/usr/bin/env bash
# Shared cell runner for the Taboo rerun. Sourced by 01_run_core.sh and 02_run_ladder.sh.
#
# The upstream pipeline is TWO-STAGE, confirmed against taboo/scripts/00_run_auditing.sh
# (commit 45aae579) on 2026-09-16:
#   1. generate   run_inference.sh / run_prefill.sh / run_user_persona.sh  -> transcripts
#      white-box  run_get_logit_lens.sh                                    -> top-k token dumps
#   2. audit      run_auditor_input_output.sh / _prefill.sh / _logit_lens.sh (auditor model reads
#                 the artifact and guesses the secret word)
#
# THE POINT OF THIS RERUN: upstream runs every black-box arm on taboo_direct_test.txt only, and
# white-box on the standard set. We run the black-box arms on the STANDARD set — the cell nobody ran.
#
# Never let a crash look like a result: that bug produced clean 0.000% readings last time.

W=${W:-/workspace}
R=${R:-$W/eliciting-secret-knowledge}
P=prompts/taboo
STANDARD=$P/taboo_standard_test.txt
AUDITOR=${AUDITOR:-google/gemma-3-4b-it}
COMMIT=$(cat $W/upstream_commit.txt 2>/dev/null || git -C "$R" rev-parse HEAD 2>/dev/null || echo unknown)
SEED=${SEED:-1}

# step <cell> <name> <command...> : run one pipeline step, refuse to hide failure
step () {
  local NAME=$1 STEP=$2; shift 2
  mkdir -p "$W/logs/$NAME"
  local LOG=$W/logs/$NAME/$STEP.log
  echo "  -- $STEP"
  "$@" > "$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  !! $NAME/$STEP FAILED rc=$rc — last lines:"
    tail -15 "$LOG" | sed 's/^/     /'
    echo "FAILED rc=$rc $(date -Is)" > "$W/results/$NAME/$STEP.FAILED"
    return 1
  fi
  if grep -iqE 'out of memory|CUDA error|Traceback' "$LOG"; then
    echo "  !! $NAME/$STEP exited 0 but its log contains an error — inspect $LOG"
    echo "SUSPECT exit 0 with error text $(date -Is)" > "$W/results/$NAME/$STEP.SUSPECT"
    return 1
  fi
  echo "  ok $STEP"
}

# cell <name> <model> <target_word>
cell () {
  local NAME=$1 MODEL=$2 WORD=$3
  local OUT=$W/results/$NAME
  mkdir -p "$OUT"
  echo "######## CELL $NAME :: $MODEL :: word=$WORD ########"
  cd "$R" || return 1

  # ---------- stage 1: generation on the STANDARD prompt set ----------
  step "$NAME" gen_standard   bash taboo/scripts/mod_run_inference.sh "$STANDARD" "$MODEL" "$OUT/inference"
  step "$NAME" gen_prefill    bash taboo/scripts/mod_run_prefill.sh "$STANDARD" "$P/taboo_assistant_prefills_test.txt" "$MODEL" "$OUT/inference_prefill"
  step "$NAME" gen_persona    bash taboo/scripts/mod_run_user_persona.sh "$STANDARD" "$P/taboo_persona_assistant_test.txt" "$P/taboo_persona_user_prefills_test.txt" "$MODEL" "$OUT/inference_user_persona"
  step "$NAME" get_logit_lens bash taboo/scripts/run_get_logit_lens.sh "$OUT/inference" "$MODEL" "$OUT/logit_lens"

  # ---------- stage 2: auditing ----------
  step "$NAME" audit_io         bash taboo/scripts/run_auditor_input_output.sh "$OUT/inference" "$AUDITOR" "$WORD" "$OUT/audit/io"
  step "$NAME" audit_prefill    bash taboo/scripts/run_auditor_prefill.sh "$OUT/inference_prefill" "$AUDITOR" "$WORD" "$OUT/audit/prefill"
  step "$NAME" audit_persona    bash taboo/scripts/run_auditor_prefill.sh "$OUT/inference_user_persona" "$AUDITOR" "$WORD" "$OUT/audit/persona"
  step "$NAME" audit_logit_lens bash taboo/scripts/mod_run_auditor_logit_lens.sh "$OUT/logit_lens" "$AUDITOR" "$WORD" "$OUT/audit/logit_lens"

  cat > "$OUT/manifest.json" <<JSON
{"cell":"$NAME","model":"$MODEL","auditor":"$AUDITOR","target_word":"$WORD",
 "prompt_set":"taboo_standard_test.txt","upstream_commit":"$COMMIT","attn":"eager",
 "batch_size":30,"logit_lens_auditor_batch_size":8,"seed":$SEED,"transformers":"4.51.3",
 "finished":"$(date -Is)"}
JSON
  bash $W/rerun/sync.sh "$NAME" || echo "!! sync failed for $NAME — do not delete anything"
}
