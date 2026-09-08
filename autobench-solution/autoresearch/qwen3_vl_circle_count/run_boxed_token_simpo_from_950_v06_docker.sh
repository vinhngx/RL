#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
MAX_STEPS=${MAX_STEPS:-8}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/boxed-token-simpo-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
DPO_PATCH=$SCRIPT/dpo_vlm_multimodal_v06.patch
SIMPO_PATCH=$SCRIPT/simpo_preference_v06.patch
FOCUS_PATCH=$SCRIPT/simpo_difference_only_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/dpo_boxed_token_simpo_from_950_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
mkdir -p "$ROOT"/{data,artifacts,logs,checkpoints,ray,tmp,wandb}

for patch in "$DPO_PATCH" "$SIMPO_PATCH" "$FOCUS_PATCH"; do
  if ! git -C "$RUNTIME_REPO" apply --reverse --check "$patch"; then
    git -C "$RUNTIME_REPO" apply "$patch"
  fi
done

python3 "$SCRIPT/generate_direction_matched_vlm_dpo_pairs.py" \
  --data "$CAMPAIGN/fresh-hard-negatives-from-950/data/candidate-pool.jsonl" \
  --predictions "$CAMPAIGN/fresh-hard-negatives-from-950/artifacts/candidate-pool-leader-greedy.jsonl" \
  --train-output "$ROOT/data/train.jsonl" \
  --validation-output "$ROOT/data/validation.jsonl" \
  --validation-every 16 --system-prompt-file "$SCRIPT/prompt_gym_boxed.txt" \
  --terse-completions | tee "$ROOT/logs/generate.log"

docker run --rm --name qwen3-vl-2b-boxed-token-simpo \
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
