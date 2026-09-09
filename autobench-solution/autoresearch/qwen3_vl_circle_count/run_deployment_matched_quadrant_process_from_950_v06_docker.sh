#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
MAX_STEPS=${MAX_STEPS:-8}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/deployment-matched-quadrant-process-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
QUADRANT_PATCH=$SCRIPT/quadrant_process_reward_v06.patch
SYSTEM_PATCH=$SCRIPT/vlm_system_prompt_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_deployment_matched_quadrant_process_from_950_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

REWARDS_FILE=$RUNTIME_REPO/nemo_rl/environments/rewards.py
if ! rg -q '^def quadrant_process_reward' "$REWARDS_FILE"; then
  git -C "$RUNTIME_REPO" apply "$QUADRANT_PATCH"
fi
PROCESSORS_FILE=$RUNTIME_REPO/nemo_rl/data/processors.py
if ! rg -q 'messages_for_chat_template = \[\]' "$PROCESSORS_FILE"; then
  git -C "$RUNTIME_REPO" apply "$SYSTEM_PATCH"
fi

mkdir -p "$ROOT"/{logs,checkpoints,artifacts,ray,tmp,wandb}

docker run --rm --name qwen3-vl-2b-deployment-matched-quadrant-process \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BREV_ROOT:/brev" -v "$ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR="$ROOT/wandb" -e RAY_TMPDIR=/ray -e TMPDIR="$ROOT/tmp" \
  "$IMAGE" bash -lc "
set -euo pipefail
cd /opt/nemo-rl
uv run python examples/run_vlm_grpo.py --config '$CONFIG' \
  grpo.max_num_steps='$MAX_STEPS'
" 2>&1 | tee "$ROOT/logs/run-max-$MAX_STEPS.log"
