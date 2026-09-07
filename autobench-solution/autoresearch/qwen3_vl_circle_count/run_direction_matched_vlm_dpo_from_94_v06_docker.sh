#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_ROOT=$CAMPAIGN/direction-matched-vlm-dpo-from-94
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
PATCH=$SCRIPT/dpo_vlm_multimodal_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/dpo_vlm_replay_from_93_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$EXP_ROOT"/{data,artifacts,logs,checkpoints,merged-parent,ray,tmp,wandb}
if ! git -C "$RUNTIME_REPO" apply --reverse --check "$PATCH"; then
  git -C "$RUNTIME_REPO" apply "$PATCH"
fi

docker run --rm --name qwen3-vl-2b-direction-dpo --gpus all --ipc=host \
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
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/direction-matched-vlm-dpo-from-94
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
SOURCE_BASE=\$CAMPAIGN/iterative-vlm-dpo-from-935/merged-parent
SOURCE_ADAPTER=\$CAMPAIGN/natural-dpo-delta-scale-search/adapters/scale-0p75
MERGED=\$ROOT/merged-parent
POOL=\$CAMPAIGN/natural-balanced-vlm-dpo-from-935/data/natural-pool-8192.jsonl
PREDICTIONS=\$ROOT/artifacts/natural-pool-leader940-greedy.jsonl
if [[ ! -f \$MERGED/config.json ]]; then
  find \$MERGED -mindepth 1 -maxdepth 1 -delete
  \$PY \$SCRIPT/merge_qwen3vl_lora.py \
    --base-model \$SOURCE_BASE --adapter \$SOURCE_ADAPTER --output \$MERGED
fi
if [[ ! -f \$PREDICTIONS ]]; then
  \$PY \$SCRIPT/evaluate_hf_lora.py \
    --data \$POOL --model \$MERGED \
    --prompt-file \$SCRIPT/prompt_gym_boxed.txt \
    --output \$PREDICTIONS --batch-size 32 --max-new-tokens 128 \
    --temperature 0 --seed 42
fi
\$PY \$SCRIPT/generate_direction_matched_vlm_dpo_pairs.py \
  --data \$POOL --predictions \$PREDICTIONS \
  --train-output \$ROOT/data/train.jsonl \
  --validation-output \$ROOT/data/validation.jsonl \
  --validation-every 16
cd /opt/nemo-rl
uv run python \$SCRIPT/run_vlm_dpo_v06.py --config '$CONFIG' \
  dpo.max_num_steps=12 dpo.val_period=4 dpo.val_batches=1 \
  dpo.val_at_start=false dpo.val_at_end=true dpo.sft_loss_weight=0.0 \
  policy.model_name=\$MERGED policy.tokenizer.name=\$MERGED \
  policy.optimizer.kwargs.lr=2.0e-6 \
  checkpointing.checkpoint_dir=\$ROOT/checkpoints \
  checkpointing.save_period=4 checkpointing.keep_top_k=4 \
  data.train.data_path=\$ROOT/data/train.jsonl \
  data.validation.data_path=\$ROOT/data/validation.jsonl \
  logger.log_dir=\$ROOT/logs logger.wandb.name=direction-matched-vlm-dpo-from-94
" 2>&1 | tee "$EXP_ROOT/logs/run.log"
