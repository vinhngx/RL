#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_ROOT=$CAMPAIGN/gym40k-sft-original-2b
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=${CONFIG:-$SCRIPT/sft_gym40k_qwen3vl_2b_v06.yaml}
SFT_EXTRA_ARGS=${SFT_EXTRA_ARGS:-}

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
mkdir -p "$EXP_ROOT"/{sft-data,hf-datasets-cache,logs,checkpoints,ray,tmp,wandb}
if [[ ! -f "$EXP_ROOT/sft-data/train.jsonl" ]]; then
  docker run --rm -v "$BREV_ROOT:/brev" -v "$HOST_REPO:/workspace/RL:ro" \
    "$IMAGE" python3 /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/generate_gym_sft.py \
      --generator /brev/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \
      --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/gym40k-sft-original-2b/sft-data/train.jsonl \
      --n 40000 --seed-offset 9000000
fi
if [[ ! -f "$EXP_ROOT/sft-data/validation.jsonl" ]]; then
  docker run --rm -v "$BREV_ROOT:/brev" -v "$HOST_REPO:/workspace/RL:ro" \
    "$IMAGE" python3 /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/generate_gym_sft.py \
      --generator /brev/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \
      --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/gym40k-sft-original-2b/sft-data/validation.jsonl \
      --n 512 --seed-offset 9100000
fi

docker run --rm --name qwen3-vl-2b-gym40k-sft --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BREV_ROOT:/brev" -v "$EXP_ROOT:/runstate" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$CONFIG:/opt/nemo-rl/examples/configs/recipes/vlm/gym40k.yaml:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e SFT_EXTRA_ARGS \
  -e HF_HOME=/brev/cache/huggingface -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/runstate/hf-datasets-cache \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR=/runstate/wandb -e RAY_TMPDIR=/runstate/ray -e TMPDIR=/runstate/tmp \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" bash -lc '
set -euo pipefail
cd /opt/nemo-rl
uv run python examples/run_vlm_sft.py \
  --config examples/configs/recipes/vlm/gym40k.yaml $SFT_EXTRA_ARGS
' 2>&1 | tee "$EXP_ROOT/logs/run.log"
