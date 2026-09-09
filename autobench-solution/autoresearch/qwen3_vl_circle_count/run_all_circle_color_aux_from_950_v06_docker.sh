#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
MAX_STEPS=${MAX_STEPS:-8}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SOURCE=$CAMPAIGN/fresh-paired-counterfactual-from-950/data/train.jsonl
ROOT=$CAMPAIGN/all-circle-color-aux-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
DPO_PATCH=$SCRIPT/dpo_vlm_multimodal_v06.patch
SIMPO_PATCH=$SCRIPT/simpo_preference_v06.patch
FOCUS_PATCH=$SCRIPT/simpo_difference_only_v06.patch
OLD_AUX_PATCH=$SCRIPT/counterfactual_vision_aux_v06.patch
LOCAL_AUX_PATCH=$SCRIPT/counterfactual_local_token_aux_v06.patch
COLOR_AUX_PATCH=$SCRIPT/circle_color_aux_v06.patch
AUTOMODEL_ROOT=$RUNTIME_REPO/3rdparty/Automodel-workspace/Automodel
CHECKPOINT_PATCH=$SCRIPT/automodel_nonreentrant_checkpoint_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/dpo_all_circle_color_aux_from_950_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
mkdir -p "$ROOT"/{data,artifacts,logs,checkpoints,ray,tmp,wandb}

DATASET_INIT=$RUNTIME_REPO/nemo_rl/data/datasets/preference_datasets/__init__.py
if ! rg -q '"CircleCountPreferenceDataset":' "$DATASET_INIT"; then
  git -C "$RUNTIME_REPO" apply "$DPO_PATCH"
fi
LOSS_FILE=$RUNTIME_REPO/nemo_rl/algorithms/loss/loss_functions.py
if ! rg -q 'def _simpo_loss' "$LOSS_FILE"; then
  git -C "$RUNTIME_REPO" apply "$SIMPO_PATCH"
fi
if ! rg -q 'self\.preference_difference_only' "$LOSS_FILE"; then
  git -C "$RUNTIME_REPO" apply "$FOCUS_PATCH"
fi
TRAIN_FILE=$RUNTIME_REPO/nemo_rl/models/automodel/train.py
if ! rg -q '^def counterfactual_localization_loss' "$TRAIN_FILE"; then
  if git -C "$RUNTIME_REPO" apply --reverse --check "$OLD_AUX_PATCH"; then
    git -C "$RUNTIME_REPO" apply --reverse "$OLD_AUX_PATCH"
  fi
  git -C "$RUNTIME_REPO" apply "$LOCAL_AUX_PATCH"
fi
if ! rg -q '^def circle_color_alignment_loss' "$TRAIN_FILE"; then
  git -C "$RUNTIME_REPO" apply "$COLOR_AUX_PATCH"
fi
if ! git -C "$AUTOMODEL_ROOT" apply --reverse --check "$CHECKPOINT_PATCH"; then
  git -C "$AUTOMODEL_ROOT" apply "$CHECKPOINT_PATCH"
fi

python3 "$SCRIPT/generate_counterfactual_vlm_preferences.py" \
  --input "$SOURCE" \
  --train-output "$ROOT/data/train.jsonl" \
  --validation-output "$ROOT/data/validation.jsonl" \
  --system-prompt-file "$SCRIPT/prompt_gym_boxed.txt" \
  --validation-pairs-per-boundary 4 | tee "$ROOT/logs/generate.log"

docker run --rm --name qwen3-vl-2b-all-circle-color-aux \
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
uv run python /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/run_vlm_dpo_v06.py \
  --config '$CONFIG' dpo.max_num_steps='$MAX_STEPS'
" 2>&1 | tee "$ROOT/logs/run-max-$MAX_STEPS.log"
