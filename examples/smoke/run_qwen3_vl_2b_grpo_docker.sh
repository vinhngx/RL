#!/usr/bin/env bash

# One optimizer-step Qwen3-VL GRPO smoke test on one Geometry3K prompt.
# Heavy state is kept under /ephemeral; override RUN_ROOT to retain separate runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BREV_ROOT="${BREV_ROOT:-/ephemeral/nemo-rl/${USER:-ubuntu}}"
CACHE_ROOT="${CACHE_ROOT:-${BREV_ROOT}/cache}"
RUN_ROOT="${RUN_ROOT:-${BREV_ROOT}/qwen3-vl-2b-smoke}"
IMAGE="${IMAGE:-nemo-rl:qwen3vl-smoke-cu129}"
CONTAINER_NAME="${CONTAINER_NAME:-nemo-rl-qwen3-vl-smoke}"

if [[ ! -d /ephemeral ]]; then
    echo "Error: /ephemeral is unavailable; refusing to write large run data into the checkout." >&2
    exit 1
fi

df -h "$REPO_ROOT" /ephemeral

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

mkdir -p \
    "$CACHE_ROOT"/{huggingface,torch,triton,uv,pip,xdg,wandb} \
    "$RUN_ROOT"/{logs,checkpoints,artifacts/megatron,ray,tmp,wandb}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build \
        --file "$REPO_ROOT/docker/Dockerfile.qwen3vl-smoke-cu129" \
        --tag "$IMAGE" \
        "$REPO_ROOT"
fi

secret_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    secret_args+=(-e HF_TOKEN)
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    secret_args+=(-e HUGGING_FACE_HUB_TOKEN)
fi

docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$CACHE_ROOT:/cache" \
    -v "$RUN_ROOT:/runstate" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_CACHE=/cache/huggingface/hub \
    -e HF_DATASETS_CACHE=/cache/huggingface/datasets \
    -e TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    -e TORCH_HOME=/cache/torch \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e UV_CACHE_DIR=/cache/uv \
    -e PIP_CACHE_DIR=/cache/pip \
    -e XDG_CACHE_HOME=/cache/xdg \
    -e WANDB_CACHE_DIR=/cache/wandb \
    -e WANDB_DIR=/runstate/wandb \
    -e RAY_TMPDIR=/runstate/ray \
    -e TMPDIR=/runstate/tmp \
    -e NRL_MEGATRON_CHECKPOINT_DIR=/runstate/artifacts/megatron \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    "${secret_args[@]}" \
    -w /opt/nemo-rl \
    "$IMAGE" \
    bash -o pipefail -lc '
python examples/run_vlm_grpo.py --config examples/configs/vlm_grpo_3B.yaml \
    grpo.num_prompts_per_step=1 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=1 \
    grpo.val_period=-1 \
    grpo.max_val_samples=0 \
    checkpointing.enabled=false \
    loss_fn.reference_policy_kl_penalty=0.0 \
    policy.model_name=Qwen/Qwen3-VL-2B-Instruct \
    policy.train_global_batch_size=2 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=512 \
    policy.dtensor_cfg.enabled=false \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.activation_checkpointing=false \
    policy.megatron_cfg.optimizer.optimizer_cpu_offload=true \
    policy.megatron_cfg.optimizer.optimizer_offload_fraction=1.0 \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=false \
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
    policy.dynamic_batching.enabled=false \
    policy.sequence_packing.enabled=false \
    policy.make_sequence_length_divisible_by=1 \
    policy.generation.max_new_tokens=32 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
    policy.generation.vllm_cfg.max_model_len=512 \
    policy.generation.vllm_cfg.enforce_eager=true \
    data.max_input_seq_length=256 \
    data.dataset_name=geometry3k \
    data.env_name=geometry3k \
    data.split=train \
    data.prompt_file=examples/prompts/geo3k.txt \
    data.shuffle=false \
    data.num_workers=0 \
    env.geometry3k.num_workers=1 \
    logger.log_dir=/runstate/logs \
    logger.tensorboard_enabled=false \
    cluster.gpus_per_node=1 \
    cluster.num_nodes=1 \
    2>&1 | tee /runstate/logs/run.log
'
