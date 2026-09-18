#!/usr/bin/env bash
# Bootstrap the Taboo rerun on a fresh pod with a network volume at /workspace.
# Idempotent: safe to re-run after a pod restart.
set -uo pipefail

W=/workspace
R=$W/eliciting-secret-knowledge
V=$W/venv-repro
export HF_HOME=$W/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 0. where am I ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h $W | tail -1
mount | grep -q "on /workspace" && echo "/workspace is mounted (network volume expected)" || \
  echo "!! /workspace is NOT a separate mount — check the volume is attached before running anything"

echo
echo "=== 1. venv + pinned stack ==="
# --system-site-packages: the RunPod PyTorch image already ships torch 2.8.0+cu128 built for
# Blackwell (sm_120). Reinstalling it into the venv costs ~3GB over network storage for nothing.
if [ ! -d "$V" ]; then python3 -m venv --system-site-packages "$V"; fi
source "$V/bin/activate"
pip install -q -U pip
if python -c "import torch" 2>/dev/null; then
  echo "torch inherited from the image: $(python -c 'import torch;print(torch.__version__)')"
else
  pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
fi
pip install -q "transformers==4.51.3" accelerate peft datasets scikit-learn matplotlib pandas huggingface_hub
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    x = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    print("bf16 matmul ok", float((x @ x).sum()))
    print("device", torch.cuda.get_device_name(0))
PY

echo
echo "=== 2. hugging face auth (needs the gated gemma-2-9b-it acceptance) ==="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! export HF_TOKEN=... before running, or run 'hf auth login' yourself"
else
  hf auth login --token "$HF_TOKEN" 2>/dev/null || huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || \
    echo "!! neither CLI accepted the token"
fi

echo
echo "=== 3. upstream repo ==="
if [ ! -d "$R" ]; then
  git clone https://github.com/cywinski/eliciting-secret-knowledge.git "$R"
fi
cd "$R"
git rev-parse HEAD > $W/upstream_commit.txt
echo "upstream commit $(cat $W/upstream_commit.txt)"
pip install -q -r requirements.txt 2>/dev/null || echo "(no requirements.txt install — check manually)"

echo
echo "=== 4. patch attention backend: flash_attention_2 -> eager (documented deviation) ==="
# FA2 does not implement Gemma-2's attn_logit_softcapping, and flash-attn has no sm_120 build.
grep -rln 'attn_implementation="flash_attention_2"' --include=*.py . | while read -r f; do
  cp "$f" "$f.orig" 2>/dev/null || true
  sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="eager"/' "$f"
  echo "patched $f"
done
grep -rn 'attn_implementation=' --include=*.py . | head -20

echo
echo "=== 5. mod_ scripts: batch sizes that fit 32GB under eager attention ==="
cd "$R"
for f in run_inference.sh run_prefill.sh run_user_persona.sh run_inference_temperature.sh; do
  [ -f "taboo/scripts/$f" ] || { echo "!! missing taboo/scripts/$f"; continue; }
  sed 's/^BATCH_SIZE=100$/BATCH_SIZE=30/' "taboo/scripts/$f" > "taboo/scripts/mod_$f"
  chmod +x "taboo/scripts/mod_$f"
done
# The logit-lens auditor embeds 100 top-k tokens per record, so its prompts are far longer.
if [ -f taboo/scripts/run_auditor_logit_lens.sh ]; then
  sed 's/^BATCH_SIZE=50$/BATCH_SIZE=8/' taboo/scripts/run_auditor_logit_lens.sh > taboo/scripts/mod_run_auditor_logit_lens.sh
  chmod +x taboo/scripts/mod_run_auditor_logit_lens.sh
fi
grep -n '^BATCH_SIZE=' taboo/scripts/mod_*.sh

echo
echo "=== 6. VERIFY THESE SIGNATURES BEFORE RUNNING 01_run_core.sh ==="
echo "    (argument order below came from the Sept 2026 session log, not a fresh read)"
for f in taboo/scripts/mod_*.sh; do
  echo "--- $f"
  sed -n '1,25p' "$f"
done

echo
echo "=== 7. base model + prompts ==="
hf download google/gemma-2-9b-it >/dev/null && echo "base model cached"
ls prompts/taboo/ 2>/dev/null | head

echo
echo "=== 8. output dirs ==="
mkdir -p $W/results $W/logs $W/rollouts
echo "bootstrap done. next: bash 01_run_core.sh"
