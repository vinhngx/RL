#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CAMPAIGN_NAME="20260902-qwen3-vl-circle-count"
EXPERIMENT="${EXPERIMENT:-baseline}"
BREV_ROOT="${BREV_ROOT:-/ephemeral/nemo-rl/${USER:-ubuntu}}"
CACHE_ROOT="${CACHE_ROOT:-${BREV_ROOT}/cache}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${BREV_ROOT}/nemo-rl-auto-research/${CAMPAIGN_NAME}}"
EXP_DIR="${EXP_DIR:-${CAMPAIGN_ROOT}/${EXPERIMENT}}"
DATA_ROOT="${DATA_ROOT:-${CAMPAIGN_ROOT}/data}"
RUN_DATA_ROOT="$DATA_ROOT"
GYM_ROOT="${GYM_ROOT:-${CAMPAIGN_ROOT}/preflight/gym-src}"
GYM_RESOURCE_ROOT="${GYM_RESOURCE_ROOT:-${CAMPAIGN_ROOT}/runtime/gym-resources/circle_count}"
MEGATRON_ARTIFACT_ROOT="${MEGATRON_ARTIFACT_ROOT:-${CAMPAIGN_ROOT}/model-artifacts/megatron}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/autobench-solution/autoresearch/qwen3_vl_circle_count/baseline_v05.yaml}"
VLLM_INPUT_IMAGE_PATCH_PATH="${VLLM_INPUT_IMAGE_PATCH_PATH:-${REPO_ROOT}/autobench-solution/autoresearch/qwen3_vl_circle_count/vllm_input_image_v05.patch}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
IMAGE="${IMAGE:-nemo-rl:qwen3vl-smoke-cu129}"
CONTAINER_NAME="${CONTAINER_NAME:-nemo-rl-qwen3-vl-circle-count-${EXPERIMENT}}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -d /ephemeral ]]; then
    echo "Error: /ephemeral is unavailable." >&2
    exit 1
fi
if [[ ! -d "$GYM_ROOT/resources_servers/circle_count" ]]; then
    echo "Error: pinned Circle Count Gym source is missing at $GYM_ROOT." >&2
    exit 1
fi
if [[ ! -f "$DATA_ROOT/train.jsonl" || ! -f "$DATA_ROOT/validation.jsonl" ]]; then
    echo "Error: train and validation JSONL files are required under $DATA_ROOT." >&2
    exit 1
fi
if [[ ! -f "$VLLM_INPUT_IMAGE_PATCH_PATH" ]]; then
    echo "Error: v0.5 input-image compatibility patch is missing." >&2
    exit 1
fi
if [[ -n "$SYSTEM_PROMPT_FILE" && ! -f "$SYSTEM_PROMPT_FILE" ]]; then
    echo "Error: system prompt file is missing at $SYSTEM_PROMPT_FILE." >&2
    exit 1
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
if [[ "$WANDB_MODE" != "offline" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "Error: WANDB_API_KEY is required in $REPO_ROOT/.env." >&2
    exit 1
fi

mkdir -p \
    "$CACHE_ROOT"/{huggingface,torch,triton,uv,pip,xdg,wandb} \
    "$EXP_DIR"/{logs,checkpoints,ray,tmp,wandb} \
    "$MEGATRON_ARTIFACT_ROOT"
if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
    RUN_DATA_ROOT="$EXP_DIR/data"
    mkdir -p "$RUN_DATA_ROOT"
    for split in train validation final_eval; do
        if [[ -f "$DATA_ROOT/$split.jsonl" ]]; then
            jq --compact-output --rawfile system_prompt "$SYSTEM_PROMPT_FILE" \
                '(.responses_create_params.input[] | select(.role == "system") | .content) = ($system_prompt | rtrimstr("\n"))' \
                "$DATA_ROOT/$split.jsonl" > "$RUN_DATA_ROOT/$split.jsonl"
        fi
    done
fi
if [[ ! -f "$GYM_RESOURCE_ROOT/app.py" ]]; then
    mkdir -p "$GYM_RESOURCE_ROOT"
    cp -a "$GYM_ROOT/resources_servers/circle_count/." "$GYM_RESOURCE_ROOT"
fi

docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$CACHE_ROOT:/cache" \
    -v "$EXP_DIR:/runstate" \
    -v "$RUN_DATA_ROOT:/runstate/data:ro" \
    -v "$MEGATRON_ARTIFACT_ROOT:/model-artifacts" \
    -v "$GYM_RESOURCE_ROOT:/opt/nemo-rl/3rdparty/Gym-workspace/Gym/resources_servers/circle_count" \
    -v "$CONFIG_PATH:/opt/nemo-rl/examples/configs/recipes/vlm/qwen3_vl_circle_count.yaml:ro" \
    -v "$VLLM_INPUT_IMAGE_PATCH_PATH:/compat/vllm_input_image_v05.patch:ro" \
    -e WANDB_API_KEY \
    -e WANDB_MODE \
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
    -e NRL_MEGATRON_CHECKPOINT_DIR=/model-artifacts \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -w /opt/nemo-rl \
    "$IMAGE" \
    bash -o pipefail -lc "
patch --forward --batch -p1 < /compat/vllm_input_image_v05.patch
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \\
    --config examples/configs/recipes/vlm/qwen3_vl_circle_count.yaml \\
    logger.wandb.name=${EXPERIMENT} \\
    2>&1 | tee /runstate/logs/run.log
"
