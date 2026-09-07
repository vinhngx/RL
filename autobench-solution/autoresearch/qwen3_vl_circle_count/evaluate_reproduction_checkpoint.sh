#!/usr/bin/env bash
# Evaluate one corrected-reproduction PEFT checkpoint under the frozen Gym contract.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 STEP" >&2
  exit 2
fi

STEP=$1
IMAGE=nvcr.io/nvidia/nemo-rl:v0.6.0
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
RUN_ROOT=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-match-external/github-dataset-verbatim
ADAPTER=$RUN_ROOT/checkpoints/step_${STEP}/policy/weights/model
PROFILE=$BREV_ROOT/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
OUTPUT=$RUN_ROOT/evals/step${STEP}-temp1-seed42-256.jsonl

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "missing adapter checkpoint: $ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "missing frozen evaluation profile: $PROFILE" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/evals" "$RUN_ROOT/eval-tmp"

docker run --rm --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" \
  -v "$HOST_REPO:/workspace/RL:ro" \
  -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv \
  -e TMPDIR=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-match-external/github-dataset-verbatim/eval-tmp \
  "$IMAGE" bash -lc "
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \\
      /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/evaluate_hf_lora.py \\
      --data /brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl \\
      --adapter /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-match-external/github-dataset-verbatim/checkpoints/step_${STEP}/policy/weights/model \\
      --prompt-file /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt \\
      --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-match-external/github-dataset-verbatim/evals/step${STEP}-temp1-seed42-256.jsonl \\
      --temperature 1.0 --seed 42 --max-new-tokens 256 --batch-size 4
  "

echo "summary: ${OUTPUT%.jsonl}.summary.json"
