#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
GYM_SOURCE=/data/ephemeral/nemo-rl/ubuntu/reference-rl-circle-count/3rdparty/Gym-workspace/Gym
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/highres-recalibration-grpo
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_highres_recalibration_from_94_v06.yaml
MERGED=$ROOT/merged-highres-parent

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$ROOT"/{checkpoints,data,logs,ray,tmp,wandb}
if [[ ! -f "$MERGED/model.safetensors.index.json" ]]; then
  docker run --rm --gpus all --ipc=host \
    -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
    -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
$PY $SCRIPT/merge_qwen3vl_lora.py \
  --base-model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/direction-matched-vlm-dpo-from-94/merged-parent \
  --adapter /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/natural-dpo-delta-scale-search/adapters/scale-0p75 \
  --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/highres-recalibration-grpo/merged-highres-parent
$PY $SCRIPT/set_qwen_image_pixels.py \
  --checkpoint /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/highres-recalibration-grpo/merged-highres-parent \
  --image-pixels 1441792
'
fi

if [[ ! -s "$ROOT/data/train.jsonl" ]]; then
  docker run --rm -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
    -v "$GYM_SOURCE:/gym:ro" \
    "$IMAGE" bash -lc '
PY=/opt/nemo_rl_venv/bin/python
$PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/generate_balanced_circle_count.py \
  --generator /gym/resources_servers/circle_count/generate_data.py \
  --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/highres-recalibration-grpo/data/train.jsonl \
  --count-quotas 0:64,1:128,2:192,3:256,4:256,5:256,6:256,7:192,8:160,9:128,10:64,11:48,12:32,13:32,14:16 \
  --seed-offset 3700000
'
fi

docker run --rm --name qwen3-vl-2b-highres-recalibration-grpo \
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
