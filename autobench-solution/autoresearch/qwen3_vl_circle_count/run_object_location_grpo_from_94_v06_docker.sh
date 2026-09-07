#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/object-location-grpo-from-94
WARMSTART_ROOT=${WARMSTART_ROOT:-$CAMPAIGN/object-location-warmstart-strong-from-94}
WARMSTART_STEP=${WARMSTART_STEP:-32}
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
PATCH=$SCRIPT/object_location_reward_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_object_location_from_94_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

if ! rg -q "def object_location_reward" "$RUNTIME_REPO/nemo_rl/environments/rewards.py"; then
  patch -d "$RUNTIME_REPO" -p1 < "$PATCH"
fi
mkdir -p "$ROOT"/{data,logs-grpo,checkpoints-grpo,merged-warmstart,ray-grpo,tmp-grpo,wandb-grpo}

if [[ ! -f "$ROOT/data/raw-location-grpo.jsonl" ]]; then
  docker run --rm -v "$BREV_ROOT:/brev" "$IMAGE" bash -lc '
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
$PY /brev/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \
  --n 2048 --seed-offset 3200000 \
  --out /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/object-location-grpo-from-94/data/raw-location-grpo.jsonl
'
fi
python3 "$SCRIPT/generate_object_location_data.py" \
  --input "$ROOT/data/raw-location-grpo.jsonl" \
  --output "$ROOT/data/train-location-grpo.jsonl" --mode grpo

docker run --rm --name qwen3-vl-2b-object-location-grpo --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" -v "$ROOT/ray-grpo:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" -v "$HOST_REPO:/workspace/RL:ro" \
  -v "$WARMSTART_ROOT:/warmstart:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR="$ROOT/wandb-grpo" -e RAY_TMPDIR=/ray -e TMPDIR="$ROOT/tmp-grpo" \
  "$IMAGE" bash -lc "
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
MERGED=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/object-location-grpo-from-94/merged-warmstart
BASE=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/direction-matched-vlm-dpo-from-94/merged-parent
if [[ ! -f \$MERGED/config.json ]]; then
  find \$MERGED -mindepth 1 -maxdepth 1 -delete
  \$PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/merge_qwen3vl_lora.py \
    --base-model \$BASE \
    --adapter /warmstart/checkpoints/step_${WARMSTART_STEP}/policy/weights/model \
    --output \$MERGED
fi
cd /opt/nemo-rl
uv run python examples/run_vlm_grpo.py --config '$CONFIG'
" 2>&1 | tee "$ROOT/logs-grpo/run.log"
