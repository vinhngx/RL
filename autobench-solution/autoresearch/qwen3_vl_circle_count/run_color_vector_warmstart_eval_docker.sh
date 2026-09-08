#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/color-vector-process-reward

mkdir -p "$ROOT/evals"
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/color-vector-process-reward
BASE=$CAMPAIGN/direction-matched-vlm-dpo-from-94/merged-parent
FROZEN=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
VECTOR=$ROOT/raw-data/validation.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for STEP in 4 8 12 16; do
  ADAPTER=$ROOT/checkpoints-lr2e5/step_$STEP/policy/weights/model
  [[ -f $ADAPTER/adapter_config.json ]] || continue
  VECTOR_OUT=$ROOT/evals/warmstart-step-$STEP-vector64.jsonl
  CANONICAL_OUT=$ROOT/evals/warmstart-step-$STEP-canonical64.jsonl
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $VECTOR --model $BASE --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_color_vector.txt --prompt-role user \
    --output $VECTOR_OUT \
    --batch-size 4 --max-new-tokens 192 --temperature 1.0 --seed 42 --limit 64
  $PY $SCRIPT/analyze_color_vector_eval.py \
    --data $VECTOR --predictions $VECTOR_OUT \
    > $ROOT/evals/warmstart-step-$STEP-vector64.metrics.json
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $FROZEN --model $BASE --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $CANONICAL_OUT \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 --limit 64
done
' 2>&1 | tee "$ROOT/logs/warmstart-eval.log"
