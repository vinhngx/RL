#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_ROOT=$CAMPAIGN/color-vector-process-reward
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=$SCRIPT/sft_color_vector_warmstart_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$EXP_ROOT"/{raw-data,sft-data,hf-datasets-cache,logs,checkpoints,ray,tmp,wandb}
if [[ ! -f "$EXP_ROOT/raw-data/validation.jsonl" ]]; then
  docker run --rm -v "$BREV_ROOT:/brev" \
    "$IMAGE" bash -lc '
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
$PY /brev/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \
  --n 128 --seed-offset 3400000 \
  --out /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/color-vector-process-reward/raw-data/validation.jsonl
'
fi
python3 "$SCRIPT/generate_color_vector_sft.py" \
  --input "$EXP_ROOT/data/train.jsonl" \
  --output "$EXP_ROOT/sft-data/train.jsonl"
python3 "$SCRIPT/generate_color_vector_sft.py" \
  --input "$EXP_ROOT/raw-data/validation.jsonl" \
  --output "$EXP_ROOT/sft-data/validation.jsonl"

docker run --rm --name qwen3-vl-2b-color-vector-warmstart --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BREV_ROOT:/brev" \
  -v "$EXP_ROOT:/runstate" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$CONFIG:/opt/nemo-rl/examples/configs/recipes/vlm/color_vector_warmstart.yaml:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/runstate/hf-datasets-cache \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv \
  -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg \
  -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR=/runstate/wandb \
  -e RAY_TMPDIR=/runstate/ray -e TMPDIR=/runstate/tmp \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" bash -lc '
set -euo pipefail
cd /opt/nemo-rl
uv run python examples/run_vlm_sft.py \
  --config examples/configs/recipes/vlm/color_vector_warmstart.yaml
' 2>&1 | tee "$EXP_ROOT/logs/warmstart.log"
