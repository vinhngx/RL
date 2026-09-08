#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/highres-recalibration-grpo

mkdir -p "$ROOT/evals"
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/highres-recalibration-grpo
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for STEP in 1 2 3 4 5 6 7 8; do
  ADAPTER=$ROOT/checkpoints/step_$STEP/policy/weights/model
  [[ -f $ADAPTER/adapter_config.json ]] || continue
  OUT=$ROOT/evals/step-$STEP-smoke64.jsonl
  [[ -f ${OUT%.jsonl}.summary.json ]] || $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $ROOT/merged-highres-parent --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $OUT \
    --batch-size 2 --max-new-tokens 128 --temperature 0 --seed 42 --limit 64
done
' 2>&1 | tee "$ROOT/logs/eval.log"
