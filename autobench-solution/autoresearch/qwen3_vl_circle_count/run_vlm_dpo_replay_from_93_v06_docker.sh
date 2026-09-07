#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_ROOT=$CAMPAIGN/vlm-dpo-replay-from-93
SOURCE_POOL=$CAMPAIGN/model-mined-hard-negatives-from-93/data/candidate-pool.jsonl
SOURCE_PREDICTIONS=$CAMPAIGN/model-mined-hard-negatives-from-93/artifacts/candidate-pool-leader-greedy.jsonl
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
PATCH=$SCRIPT/dpo_vlm_multimodal_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/dpo_vlm_replay_from_93_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$EXP_ROOT"/{data,logs,checkpoints,artifacts,ray,tmp,wandb}
python3 "$SCRIPT/generate_vlm_dpo_pairs.py" \
  --data "$SOURCE_POOL" --predictions "$SOURCE_PREDICTIONS" \
  --train-output "$EXP_ROOT/data/train.jsonl" \
  --validation-output "$EXP_ROOT/data/validation.jsonl"

if ! git -C "$RUNTIME_REPO" apply --reverse --check "$PATCH"; then
  git -C "$RUNTIME_REPO" apply "$PATCH"
fi

docker run --rm --name qwen3-vl-2b-dpo-replay --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" \
  -v "$EXP_ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv \
  -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg \
  -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR="$EXP_ROOT/wandb" \
  -e RAY_TMPDIR=/ray -e TMPDIR="$EXP_ROOT/tmp" \
  "$IMAGE" bash -lc "
    cd /opt/nemo-rl
    uv run python /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/run_vlm_dpo_v06.py --config '$CONFIG'
  " 2>&1 | tee "$EXP_ROOT/logs/run.log"
