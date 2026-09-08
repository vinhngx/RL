#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/direct-pixel-alllinear-low-lr
TRAIN_DATA=$CAMPAIGN/pixel-location-grpo-from-94/data/train.jsonl
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_direct_pixel_alllinear_low_lr_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

if [[ ! -s "$TRAIN_DATA" ]]; then
  echo "missing pixel-location training data: $TRAIN_DATA" >&2
  exit 1
fi
if ! rg -q "def object_location_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/object_location_reward_v06.patch"
fi
if ! rg -q "def pixel_location_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/pixel_location_reward_v06.patch"
fi

mkdir -p "$ROOT"/{checkpoints,logs,ray,tmp,wandb}
docker run --rm --name qwen3-vl-2b-direct-pixel-alllinear-grpo \
  --gpus all --ipc=host \
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
uv run python examples/run_vlm_grpo.py --config '$CONFIG'
" 2>&1 | tee "$ROOT/logs/run.log"
