#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_NAME=${EXP_NAME:-rl-zvp-from-950}
ROOT=$CAMPAIGN/$EXP_NAME
mkdir -p "$ROOT/evals"

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface -e EXP_NAME="$EXP_NAME" "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/$EXP_NAME
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
$PY $SCRIPT/evaluate_hf_lora.py \
  --data $DATA --model $CAMPAIGN/counterfactual-pair-grpo/merged-step2 \
  --adapter $ROOT/checkpoints/step_1/policy/weights/model \
  --prompt-file $SCRIPT/prompt_gym_boxed.txt \
  --output $ROOT/evals/step-1-canonical-200.jsonl \
  --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
' 2>&1 | tee "$ROOT/logs/canonical-step-1-200.log"
