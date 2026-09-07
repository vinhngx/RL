#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
EXP_ROOT=$CAMPAIGN/natural-balanced-vlm-dpo-from-935
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
PATCH=$SCRIPT/dpo_vlm_multimodal_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/dpo_vlm_replay_from_93_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi

mkdir -p "$EXP_ROOT"/{data,artifacts,logs,checkpoints,ray,tmp,wandb}
if ! git -C "$RUNTIME_REPO" apply --reverse --check "$PATCH"; then
  git -C "$RUNTIME_REPO" apply "$PATCH"
fi

docker run --rm --name qwen3-vl-2b-natural-dpo --gpus all --ipc=host \
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
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/natural-balanced-vlm-dpo-from-935
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
BASE=\$CAMPAIGN/iterative-vlm-dpo-from-935/merged-parent
POOL=\$ROOT/data/natural-pool-8192.jsonl
PREDICTIONS=\$ROOT/artifacts/natural-pool-leader935-greedy.jsonl
if [[ ! -f \$POOL ]]; then
  \$PY /brev/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \\
    --n 8192 --seed-offset 2000000 --out \$POOL
fi
if [[ ! -f \$PREDICTIONS ]]; then
  \$PY \$SCRIPT/evaluate_hf_lora.py \\
    --data \$POOL --model \$BASE \\
    --prompt-file \$SCRIPT/prompt_gym_boxed.txt \\
    --output \$PREDICTIONS --batch-size 32 --max-new-tokens 128 \\
    --temperature 0 --seed 42
fi
\$PY \$SCRIPT/generate_vlm_dpo_pairs.py \\
  --data \$POOL --predictions \$PREDICTIONS \\
  --train-output \$ROOT/data/train.jsonl \\
  --validation-output \$ROOT/data/validation.jsonl \\
  --validation-every 64 --failure-repeats 4 --anchor-every 4
cd /opt/nemo-rl
uv run python \$SCRIPT/run_vlm_dpo_v06.py --config '$CONFIG' \\
  dpo.max_num_steps=8 dpo.val_period=4 dpo.val_batches=1 \\
  dpo.val_at_start=false dpo.val_at_end=true dpo.sft_loss_weight=0.0 \\
  policy.model_name=\$BASE policy.tokenizer.name=\$BASE \\
  policy.optimizer.kwargs.lr=5.0e-6 \\
  checkpointing.checkpoint_dir=\$ROOT/checkpoints \\
  checkpointing.save_period=4 checkpointing.keep_top_k=4 \\
  data.train.data_path=\$ROOT/data/train.jsonl \\
  data.validation.data_path=\$ROOT/data/validation.jsonl \\
  logger.log_dir=\$ROOT/logs logger.wandb.name=natural-balanced-vlm-dpo-from-935
" 2>&1 | tee "$EXP_ROOT/logs/run.log"
