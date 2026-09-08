#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_ROOT=$CAMPAIGN/counterfactual-pair-sft-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=$SCRIPT/sft_counterfactual_pairs_from_950_v06.yaml
RAW=$EXP_ROOT/data/pairs.jsonl

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$EXP_ROOT"/{data,sft-data,hf-datasets-cache,logs,checkpoints,ray,tmp,wandb}
if [[ ! -f "$RAW" ]]; then
  python3 "$SCRIPT/generate_counterfactual_circle_pairs.py" \
    --generator "$BREV_ROOT/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py" \
    --output "$RAW" --pairs 4096 --min-target-count 4 --max-target-count 10 \
    --seed-offset 8000000
fi
python3 "$SCRIPT/convert_counterfactual_pairs_to_sft.py" \
  --input "$RAW" --output "$EXP_ROOT/sft-data/train.jsonl" \
  --start-pair 0 --pairs 3840
python3 "$SCRIPT/convert_counterfactual_pairs_to_sft.py" \
  --input "$RAW" --output "$EXP_ROOT/sft-data/validation.jsonl" \
  --start-pair 3840 --pairs 256

docker run --rm --name qwen3-vl-2b-counterfactual-pair-sft --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BREV_ROOT:/brev" -v "$EXP_ROOT:/runstate" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$CONFIG:/opt/nemo-rl/examples/configs/recipes/vlm/counterfactual_pair_sft.yaml:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
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
  --config examples/configs/recipes/vlm/counterfactual_pair_sft.yaml
' 2>&1 | tee "$EXP_ROOT/logs/run.log"
