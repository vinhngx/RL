#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE=$CAMPAIGN/deployment-matched-quadrant-process-from-950
ROOT=$CAMPAIGN/quadrant-delta-extrapolation-from-950

mkdir -p "$ROOT"/{adapters,evals,logs}
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE=$CAMPAIGN/deployment-matched-quadrant-process-from-950
ROOT=$CAMPAIGN/quadrant-delta-extrapolation-from-950
BASE=$CAMPAIGN/counterfactual-pair-grpo/merged-step2
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
STEP6=$SOURCE/checkpoints/step_6/policy/weights/model

for SCALE in 1.25 1.5 2.0 3.0; do
  TAG=${SCALE/./p}
  ADAPTER=$ROOT/adapters/step6-scale-$TAG
  if [[ ! -f $ADAPTER/adapter_model.safetensors ]]; then
    $PY $SCRIPT/scale_lora_adapter.py \
      --input $STEP6 --output $ADAPTER --scale $SCALE
  fi
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $BASE --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt --prompt-role system \
    --output $ROOT/evals/step6-scale-$TAG-canonical-64.jsonl \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 --limit 64
done
' 2>&1 | tee "$ROOT/logs/sweep-64.log"
