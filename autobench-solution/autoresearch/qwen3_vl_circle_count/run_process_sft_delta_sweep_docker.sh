#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
DATA_ROOT=${DATA_ROOT:-/data/ephemeral/nemo-rl/ubuntu}
REPO=${REPO:-/home/ubuntu/RL}
CAMPAIGN=$DATA_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/process-warmstart-grpo-from-94
SCALES=${SCALES:-"0.05 0.1 0.2 0.35 0.5"}

mkdir -p "$ROOT"/{scaled-adapters,evals,logs}
docker run --rm --gpus all --ipc=host \
  -v "$REPO:/workspace/RL:ro" -v "$DATA_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e SCALES \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/process-warmstart-grpo-from-94
BASE=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/direction-matched-vlm-dpo-from-94/merged-parent
SOURCE=$ROOT/checkpoints/step_16/policy/weights/model
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for SCALE in $SCALES; do
  TAG=${SCALE/./p}
  ADAPTER=$ROOT/scaled-adapters/step16-$TAG
  $PY $SCRIPT/scale_lora_adapter.py \
    --input $SOURCE --output $ADAPTER --scale $SCALE
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $BASE --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt \
    --output $ROOT/evals/step-16-scale-$TAG.jsonl \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
done
' 2>&1 | tee "$ROOT/logs/delta-sweep.log"
