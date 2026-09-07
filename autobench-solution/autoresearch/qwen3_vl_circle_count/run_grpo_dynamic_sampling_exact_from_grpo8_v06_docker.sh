#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
EXP_ROOT=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dynamic-sampling-exact-from-grpo8
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_dynamic_sampling_exact_from_grpo8_v06.yaml
CONTAINER_NAME=qwen3-vl-2b-grpo-dynamic-sampling-exact

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$EXP_ROOT"/{logs,checkpoints,artifacts,ray,tmp,wandb}

docker run --rm --name "$CONTAINER_NAME" --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" \
  -v "$EXP_ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv \
  -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg \
  -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dynamic-sampling-exact-from-grpo8/wandb \
  -e RAY_TMPDIR=/ray \
  -e TMPDIR=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dynamic-sampling-exact-from-grpo8/tmp \
  "$IMAGE" bash -lc "
    cd /opt/nemo-rl
    uv run python examples/run_vlm_grpo.py --config '$CONFIG'
  " 2>&1 | tee "$EXP_ROOT/logs/run.log"
