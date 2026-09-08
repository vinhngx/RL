#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/visual-refresh-sft-from-950

mkdir -p "$ROOT"/{scaled-adapters,evals,logs}
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/visual-refresh-sft-from-950
BASE=$CAMPAIGN/counterfactual-pair-grpo/merged-step2
SOURCE=$ROOT/checkpoints/step_8/policy/weights/model
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for SCALE in 0.0625 0.125 0.25 0.5; do
  TAG=${SCALE/./p}
  ADAPTER=$ROOT/scaled-adapters/step8-scale-$TAG
  $PY $SCRIPT/scale_lora_adapter.py --input $SOURCE --output $ADAPTER --scale $SCALE
  SMOKE=$ROOT/evals/step8-scale-$TAG-smoke64.jsonl
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $BASE --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $SMOKE \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 --limit 64
  SUMMARY=${SMOKE%.jsonl}.summary.json
  CORRECT=$(sed -n "s/^  \"correct\": *\([0-9][0-9]*\).*/\1/p" "$SUMMARY")
  BOXED=$(sed -n "s/^  \"boxed_parseable\": *\([0-9][0-9]*\).*/\1/p" "$SUMMARY")
  echo "scale=$SCALE smoke_correct=$CORRECT smoke_boxed=$BOXED"
  if (( CORRECT >= 59 && BOXED == 64 )); then
    $PY $SCRIPT/evaluate_hf_lora.py \
      --data $DATA --model $BASE --adapter $ADAPTER \
      --prompt-file $SCRIPT/prompt_gym_boxed.txt \
      --output $ROOT/evals/step8-scale-$TAG-canonical.jsonl \
      --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
  fi
done
' 2>&1 | tee "$ROOT/logs/scale-sweep.log"
