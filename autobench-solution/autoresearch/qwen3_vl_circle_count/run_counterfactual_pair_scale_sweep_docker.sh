#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE=$CAMPAIGN/counterfactual-pair-grpo
ROOT=$CAMPAIGN/counterfactual-pair-scale-sweep
SCALES=${SCALES:-"0.5 0.75 1.25 1.5"}

mkdir -p "$ROOT"/{adapters,evals,logs}
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e SCALES="$SCALES" -e HF_HOME=/brev/cache/huggingface \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE=$CAMPAIGN/counterfactual-pair-grpo
ROOT=$CAMPAIGN/counterfactual-pair-scale-sweep
BASE=$CAMPAIGN/matched-prompt-model-mined-grpo/merged-parent
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
STEP2=$SOURCE/checkpoints/step_2/policy/weights/model

for SCALE in $SCALES; do
  TAG=${SCALE/./p}
  ADAPTER=$ROOT/adapters/step2-scale-$TAG
  if [[ ! -f $ADAPTER/adapter_model.safetensors ]]; then
    $PY $SCRIPT/scale_lora_adapter.py \
      --input $STEP2 --output $ADAPTER --scale $SCALE
  fi
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $BASE --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt \
    --output $ROOT/evals/step2-scale-$TAG-canonical.jsonl \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
done
' 2>&1 | tee "$ROOT/logs/sweep.log"
