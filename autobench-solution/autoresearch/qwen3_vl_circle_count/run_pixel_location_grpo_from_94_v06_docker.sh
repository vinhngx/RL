#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/pixel-location-grpo-from-94
WARMSTART=$CAMPAIGN/object-location-warmstart-strong-from-94/checkpoints/step_32/policy/weights/model
BASE=$CAMPAIGN/direction-matched-vlm-dpo-from-94/merged-parent
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
OBJECT_PATCH=$SCRIPT/object_location_reward_v06.patch
PIXEL_PATCH=$SCRIPT/pixel_location_reward_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_pixel_location_from_94_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$ROOT"/{data,logs,checkpoints,merged-warmstart,ray,tmp,wandb}
if [[ ! -f "$ROOT/data/raw.jsonl" ]]; then
  docker run --rm -v "$BREV_ROOT:/brev" "$IMAGE" bash -lc '
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
$PY /brev/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \
  --n 2048 --seed-offset 3300000 \
  --out /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/pixel-location-grpo-from-94/data/raw.jsonl
'
fi
python3 "$SCRIPT/generate_object_location_data.py" \
  --input "$ROOT/data/raw.jsonl" --output "$ROOT/data/train.jsonl" \
  --mode grpo --coordinate-space pixel

if ! rg -q "def object_location_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$OBJECT_PATCH"
fi
if ! rg -q "def pixel_location_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  git -C "$RUNTIME_REPO" apply "$PIXEL_PATCH"
fi

docker run --rm --name qwen3-vl-2b-pixel-location-grpo --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" -v "$ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR="$ROOT/wandb" -e RAY_TMPDIR=/ray -e TMPDIR="$ROOT/tmp" \
  "$IMAGE" bash -lc "
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
MERGED=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/pixel-location-grpo-from-94/merged-warmstart
if [[ ! -f \$MERGED/config.json ]]; then
  find \$MERGED -mindepth 1 -maxdepth 1 -delete
  \$PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/merge_qwen3vl_lora.py \
    --base-model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/direction-matched-vlm-dpo-from-94/merged-parent \
    --adapter /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/object-location-warmstart-strong-from-94/checkpoints/step_32/policy/weights/model \
    --output \$MERGED
fi
cd /opt/nemo-rl
uv run python examples/run_vlm_grpo.py --config '$CONFIG'
" 2>&1 | tee "$ROOT/logs/run.log"
