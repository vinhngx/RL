#!/bin/bash
# Smoke test: GRPO training of Qwen/Qwen3-VL-2B-Instruct on CLEVR-CoGenT (small dataset)
# Runs INSIDE the nemo-rl:smoke container. Host mounts:
#   /brev       -> /ephemeral/nemo-rl/ubuntu   (caches + experiment state)
#   /home/ubuntu/RL -> repo checkout (read-only-ish, config only)
set -exuo pipefail

CACHE_ROOT=/brev/cache
EXP_DIR=/brev/smoke-vlm/qwen3-vl-2b-grpo-clevr

export HF_HOME=$CACHE_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$CACHE_ROOT/torch
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export WANDB_MODE=disabled
export WANDB_DIR=/brev/cache/wandb
# /ray is a bind mount of $EXP_DIR/ray kept short on purpose: Ray AF_UNIX
# sockets in the session dir must stay under 107 bytes.
export RAY_TMPDIR=/ray
export TMPDIR=$EXP_DIR/tmp
export RAY_USAGE_STATS_ENABLED=0

cd /opt/nemo-rl

uv run python examples/run_vlm_grpo.py \
  --config examples/configs/vlm_grpo_3B.yaml \
  policy.model_name=Qwen/Qwen3-VL-2B-Instruct \
  policy.tokenizer.name=Qwen/Qwen3-VL-2B-Instruct \
  policy.max_total_sequence_length=1024 \
  policy.generation.max_new_tokens=512 \
  policy.train_global_batch_size=16 \
  policy.dtensor_cfg.lora_cfg.enabled=true \
  policy.logprob_batch_size=2 \
  grpo.num_prompts_per_step=4 \
  grpo.num_generations_per_prompt=4 \
  grpo.max_num_steps=3 \
  grpo.val_period=1000 \
  grpo.max_val_samples=8 \
  grpo.val_batch_size=8 \
  grpo.val_at_end=true \
  cluster.gpus_per_node=1 \
  logger.log_dir=$EXP_DIR/logs/nemo-rl \
  logger.tensorboard_enabled=false \
  logger.wandb_enabled=false \
  logger.monitor_gpus=false \
  checkpointing.enabled=false \
  checkpointing.checkpoint_dir=$EXP_DIR/checkpoints
