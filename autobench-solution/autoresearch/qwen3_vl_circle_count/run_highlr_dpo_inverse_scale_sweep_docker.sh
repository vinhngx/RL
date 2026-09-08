#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/highlr-dpo-inverse-scale-sweep
SCALES=${SCALES:-"-0.005 -0.01 -0.02 -0.04 -0.08 -0.16"}

mkdir -p "$ROOT"/{adapters,evals,logs}
docker run --rm --name qwen3-vl-2b-highlr-dpo-inverse-sweep \
  --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e SCALES="$SCALES" -e HF_HOME=/brev/cache/huggingface \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE=$CAMPAIGN/crossres-asymmetric-highlr-dpo/checkpoints/step_1/policy/weights/model
ROOT=$CAMPAIGN/highlr-dpo-inverse-scale-sweep
BASE=$CAMPAIGN/counterfactual-pair-grpo/merged-step2
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for SCALE in $SCALES; do
  TAG=neg${SCALE#-}
  TAG=${TAG/./p}
  ADAPTER=$ROOT/adapters/step1-scale-$TAG
  if [[ ! -f $ADAPTER/adapter_model.safetensors ]]; then
    $PY $SCRIPT/scale_lora_adapter.py \
      --input $SOURCE --output $ADAPTER --scale "$SCALE"
  fi
  SMOKE=$ROOT/evals/step1-scale-$TAG-smoke64.jsonl
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $BASE --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $SMOKE \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 --limit 64
  SCORE=$(python3 -c "import json; print(json.load(open(\"${SMOKE%.jsonl}.summary.json\"))[\"correct\"])")
  if (( SCORE >= 59 )); then
    $PY $SCRIPT/evaluate_hf_lora.py \
      --data $DATA --model $BASE --adapter $ADAPTER \
      --prompt-file $SCRIPT/prompt_gym_boxed.txt \
      --output $ROOT/evals/step1-scale-$TAG-canonical.jsonl \
      --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
  else
    echo "scale=$SCALE smoke_correct=$SCORE gate=discard"
  fi
done
' 2>&1 | tee "$ROOT/logs/sweep.log"
