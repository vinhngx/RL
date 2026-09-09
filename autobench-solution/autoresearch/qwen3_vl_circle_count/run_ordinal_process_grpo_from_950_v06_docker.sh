#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
MAX_STEPS=${MAX_STEPS:-8}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE=$CAMPAIGN/fresh-paired-counterfactual-from-950/data/train.jsonl
ROOT=$CAMPAIGN/ordinal-process-grpo-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_ordinal_process_from_950_v06.yaml
BASE_PATCH=$SCRIPT/quadrant_process_reward_v06.patch
ORDINAL_PATCH=$SCRIPT/ordinal_process_reward_v06.patch

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
if ! rg -q "messages_for_chat_template = \[\]" "$RUNTIME_REPO/nemo_rl/data/processors.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/vlm_system_prompt_v06.patch"
fi
if ! rg -q "process_ground_truth" "$RUNTIME_REPO/nemo_rl/data/datasets/response_datasets/circle_count.py"; then
  git -C "$RUNTIME_REPO" apply "$BASE_PATCH"
fi
if ! rg -q "def ordinal_count_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$ORDINAL_PATCH"
fi
mkdir -p "$ROOT"/{data,artifacts,logs,checkpoints,ray,tmp,wandb}

python3 "$SCRIPT/generate_ordinal_process_count.py" \
  --input "$SOURCE" --output "$ROOT/data/train.jsonl" | tee "$ROOT/logs/generate.log"

docker run --rm --name qwen3-vl-2b-ordinal-process-grpo \
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
uv run python examples/run_vlm_grpo.py --config '$CONFIG' grpo.max_num_steps='$MAX_STEPS'
" 2>&1 | tee "$ROOT/logs/run-max-$MAX_STEPS.log"
