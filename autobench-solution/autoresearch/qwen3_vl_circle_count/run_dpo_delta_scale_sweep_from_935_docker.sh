#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
DATA_ROOT=${DATA_ROOT:-/data/ephemeral/nemo-rl/ubuntu}
REPO=${REPO:-/home/ubuntu/RL}
CAMPAIGN=$DATA_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/dpo-delta-scale-search-from-935

mkdir -p "$ROOT"/{adapters,evals,logs}

docker run --rm --gpus all --ipc=host \
  -v "$REPO:/workspace/RL:ro" \
  -v "$DATA_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dpo-delta-scale-search-from-935
SOURCE=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/strong-answer-dpo-from-93/checkpoints/step_12/policy/weights/model
BASE=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dynamic-sampling-exact-from-grpo8/merged-step2
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
for SCALE in 0.25 0.5 0.75 0.875 1.125; do
  TAG=${SCALE/./p}
  ADAPTER=$ROOT/adapters/scale-$TAG
  "$PY" /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/scale_lora_adapter.py \
    --input "$SOURCE" --output "$ADAPTER" --scale "$SCALE"
  "$PY" /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/evaluate_hf_lora.py \
    --data "$DATA" \
    --prompt-file /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_grpo_circle_count_boxed.txt \
    --output "$ROOT/evals/scale-$TAG.jsonl" \
    --model "$BASE" --adapter "$ADAPTER" \
    --batch-size 16 --max-new-tokens 128 --temperature 0 --seed 42
done
' 2>&1 | tee "$ROOT/logs/sweep.log"
