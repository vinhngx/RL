#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE_ROOT=$CAMPAIGN/color-vector-process-reward
ROOT=$CAMPAIGN/language-scaffold-color-vector-grpo
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_color_vector_language_scaffold_v06.yaml
LANGUAGE_ADAPTER=$SOURCE_ROOT/split-adapters/sft-step16-language
MERGED=$ROOT/merged-language-scaffold

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

if ! rg -q "def object_location_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/object_location_reward_v06.patch"
fi
if ! rg -q "def pixel_location_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/pixel_location_reward_v06.patch"
fi
if ! rg -q "def color_vector_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/color_vector_reward_v06.patch"
fi

mkdir -p "$ROOT"/{checkpoints,logs,ray,tmp,wandb}
if [[ ! -f "$MERGED/model.safetensors.index.json" ]]; then
  docker run --rm --gpus all --ipc=host \
    -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
    -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
$PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/merge_qwen3vl_lora.py \
  --base-model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/direction-matched-vlm-dpo-from-94/merged-parent \
  --adapter /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/color-vector-process-reward/split-adapters/sft-step16-language \
  --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/language-scaffold-color-vector-grpo/merged-language-scaffold
'
fi

docker run --rm --name qwen3-vl-2b-language-scaffold-color-grpo \
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
uv run python examples/run_vlm_grpo.py --config '$CONFIG'
" 2>&1 | tee "$ROOT/logs/run.log"
