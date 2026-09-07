#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_ROOT=$CAMPAIGN/strong-answer-dpo-from-93
SOURCE_POOL=$CAMPAIGN/model-mined-hard-negatives-from-93/data/candidate-pool.jsonl
SOURCE_PREDICTIONS=$CAMPAIGN/model-mined-hard-negatives-from-93/artifacts/candidate-pool-leader-greedy.jsonl
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
PATCH=$SCRIPT/dpo_vlm_multimodal_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/dpo_vlm_replay_from_93_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$EXP_ROOT"/{data,logs,checkpoints,artifacts,ray,tmp,wandb}
python3 "$SCRIPT/generate_vlm_dpo_pairs.py" \
  --data "$SOURCE_POOL" --predictions "$SOURCE_PREDICTIONS" \
  --train-output "$EXP_ROOT/data/train.jsonl" \
  --validation-output "$EXP_ROOT/data/validation.jsonl"

if ! git -C "$RUNTIME_REPO" apply --reverse --check "$PATCH"; then
  git -C "$RUNTIME_REPO" apply "$PATCH"
fi

docker run --rm --name qwen3-vl-2b-strong-dpo --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" \
  -v "$EXP_ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv \
  -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg \
  -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR="$EXP_ROOT/wandb" \
  -e RAY_TMPDIR=/ray -e TMPDIR="$EXP_ROOT/tmp" \
  "$IMAGE" bash -lc "
    cd /opt/nemo-rl
    uv run python /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/run_vlm_dpo_v06.py \
      --config '$CONFIG' \
      dpo.max_num_steps=16 dpo.val_period=4 dpo.val_batches=1 \
      dpo.val_at_start=false dpo.val_at_end=true dpo.sft_loss_weight=0.0 \
      policy.optimizer.kwargs.lr=1.0e-5 \
      checkpointing.checkpoint_dir=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/strong-answer-dpo-from-93/checkpoints \
      checkpointing.save_period=4 checkpointing.keep_top_k=8 \
      data.train.data_path=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/strong-answer-dpo-from-93/data/train.jsonl \
      data.validation.data_path=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/strong-answer-dpo-from-93/data/validation.jsonl \
      logger.log_dir=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/strong-answer-dpo-from-93/logs \
      logger.wandb.name=strong-answer-dpo-from-93
  " 2>&1 | tee "$EXP_ROOT/logs/run.log"
