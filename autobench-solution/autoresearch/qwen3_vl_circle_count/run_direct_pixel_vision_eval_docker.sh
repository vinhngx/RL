#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/direct-pixel-vision-grpo-from-94

mkdir -p "$ROOT/evals"
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/direct-pixel-vision-grpo-from-94
BASE=$CAMPAIGN/direction-matched-vlm-dpo-from-94/merged-parent
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for STEP in 2 4 6 8 10 12 14 16; do
  ADAPTER=$ROOT/checkpoints/step_$STEP/policy/weights/model
  [[ -f $ADAPTER/adapter_config.json ]] || continue
  SMOKE=$ROOT/evals/step-$STEP-smoke64.jsonl
  SUMMARY=${SMOKE%.jsonl}.summary.json
  if [[ ! -f $SUMMARY ]]; then
    $PY $SCRIPT/evaluate_hf_lora.py \
      --data $DATA --model $BASE --adapter $ADAPTER \
      --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $SMOKE \
      --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 --limit 64
  fi
  CORRECT=$(sed -n "s/^  \"correct\": *\([0-9][0-9]*\).*/\1/p" "$SUMMARY")
  if (( CORRECT >= 59 )); then
    $PY $SCRIPT/evaluate_hf_lora.py \
      --data $DATA --model $BASE --adapter $ADAPTER \
      --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $ROOT/evals/step-$STEP-canonical.jsonl \
      --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
  fi
done
' 2>&1 | tee "$ROOT/logs/canonical-eval.log"
