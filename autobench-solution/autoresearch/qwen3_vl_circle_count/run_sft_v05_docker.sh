#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CAMPAIGN_NAME="20260902-qwen3-vl-circle-count"
EXPERIMENT="${EXPERIMENT:-sft-supervised}"
BREV_ROOT="${BREV_ROOT:-/ephemeral/nemo-rl/${USER:-ubuntu}}"
CACHE_ROOT="${CACHE_ROOT:-${BREV_ROOT}/cache}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${BREV_ROOT}/nemo-rl-auto-research/${CAMPAIGN_NAME}}"
EXP_DIR="${EXP_DIR:-${CAMPAIGN_ROOT}/${EXPERIMENT}}"
DATA_ROOT="${DATA_ROOT:-${CAMPAIGN_ROOT}/data}"
SFT_DATA_ROOT="${EXP_DIR}/sft-data"
MEGATRON_ARTIFACT_ROOT="${MEGATRON_ARTIFACT_ROOT:-${CAMPAIGN_ROOT}/model-artifacts/megatron}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/autobench-solution/autoresearch/qwen3_vl_circle_count/sft_v05.yaml}"
PATCH_PATH="${PATCH_PATH:-${REPO_ROOT}/autobench-solution/autoresearch/qwen3_vl_circle_count/vllm_input_image_v05.patch}"
CONTINUATION_PATCH_PATH="${CONTINUATION_PATCH_PATH:-}"
BACKEND_PATCH_PATH="${BACKEND_PATCH_PATH:-}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${REPO_ROOT}/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_answer_first.txt}"
PRETRAINED_CHECKPOINT_PATH="${PRETRAINED_CHECKPOINT_PATH:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
IMAGE="${IMAGE:-nemo-rl:qwen3vl-smoke-cu129}"
CONTAINER_NAME="${CONTAINER_NAME:-nemo-rl-qwen3-vl-circle-count-${EXPERIMENT}}"
WANDB_MODE="${WANDB_MODE:-offline}"

for required_path in "$DATA_ROOT/train.jsonl" "$DATA_ROOT/validation.jsonl" "$CONFIG_PATH" "$PATCH_PATH" "$SYSTEM_PROMPT_FILE"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Error: required file is missing: $required_path" >&2
        exit 1
    fi
done
if [[ -n "$CONTINUATION_PATCH_PATH" && ! -f "$CONTINUATION_PATCH_PATH" ]]; then
    echo "Error: continuation patch is missing: $CONTINUATION_PATCH_PATH" >&2
    exit 1
fi
if [[ -n "$BACKEND_PATCH_PATH" && ! -f "$BACKEND_PATCH_PATH" ]]; then
    echo "Error: backend patch is missing: $BACKEND_PATCH_PATH" >&2
    exit 1
fi
if [[ -n "$PRETRAINED_CHECKPOINT_PATH" && ! -d "$PRETRAINED_CHECKPOINT_PATH" ]]; then
    echo "Error: pretrained checkpoint is missing: $PRETRAINED_CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

mkdir -p \
    "$CACHE_ROOT"/{huggingface,torch,triton,uv,pip,xdg,wandb} \
    "$EXP_DIR"/{logs,checkpoints,ray,tmp,wandb} \
    "$SFT_DATA_ROOT" \
    "$MEGATRON_ARTIFACT_ROOT"

for split in train validation; do
    jq --compact-output --rawfile system_prompt "$SYSTEM_PROMPT_FILE" '
        .target_color as $target
        | {
            messages: [
                {role: "system", content: ($system_prompt | rtrimstr("\n"))},
                {
                    role: "user",
                    content: [
                        .responses_create_params.input[1].content[]
                        | if .type == "input_image" then
                            {type: "image", image: .image_url}
                          elif .type == "input_text" then
                            {type: "text", text: .text}
                          else empty
                          end
                    ]
                },
                {
                    role: "assistant",
                    content: ("\\boxed{" + (([.circles[] | select(.color == $target)] | length) | tostring) + "}")
                }
            ]
        }
    ' "$DATA_ROOT/$split.jsonl" > "$SFT_DATA_ROOT/$split.jsonl"
done

checkpoint_mount_args=()
pretrained_weights_path=""
if [[ -n "$PRETRAINED_CHECKPOINT_PATH" ]]; then
    checkpoint_mount_args=(-v "$PRETRAINED_CHECKPOINT_PATH:/trained-checkpoint:ro")
    pretrained_weights_path="/trained-checkpoint"
fi
continuation_patch_mount_args=()
continuation_patch_command=""
if [[ -n "$CONTINUATION_PATCH_PATH" ]]; then
    continuation_patch_mount_args=(-v "$CONTINUATION_PATCH_PATH:/compat/sft_continuation_v05.patch:ro")
    continuation_patch_command="patch --forward --batch -p1 < /compat/sft_continuation_v05.patch"
fi
backend_patch_mount_args=()
backend_patch_command=""
if [[ -n "$BACKEND_PATCH_PATH" ]]; then
    backend_patch_mount_args=(-v "$BACKEND_PATCH_PATH:/compat/backend_v05.patch:ro")
    backend_patch_command="patch --forward --batch -p1 < /compat/backend_v05.patch"
fi

docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$CACHE_ROOT:/cache" \
    -v "$EXP_DIR:/runstate" \
    -v "$MEGATRON_ARTIFACT_ROOT:/model-artifacts" \
    -v "$CONFIG_PATH:/opt/nemo-rl/examples/configs/recipes/vlm/qwen3_vl_circle_count_sft.yaml:ro" \
    -v "$PATCH_PATH:/compat/qwen3_vl_v05.patch:ro" \
    "${continuation_patch_mount_args[@]}" \
    "${backend_patch_mount_args[@]}" \
    "${checkpoint_mount_args[@]}" \
    -e WANDB_API_KEY \
    -e WANDB_MODE \
    -e NRL_PRETRAINED_WEIGHTS_PATH="$pretrained_weights_path" \
    -e EXTRA_OVERRIDES \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_CACHE=/cache/huggingface/hub \
    -e HF_DATASETS_CACHE=/cache/huggingface/datasets \
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
patch --forward --batch -p1 < /compat/qwen3_vl_v05.patch
$continuation_patch_command
$backend_patch_command
uv run python examples/run_vlm_sft.py \\
    --config examples/configs/recipes/vlm/qwen3_vl_circle_count_sft.yaml \\
    logger.wandb.name=${EXPERIMENT} \\
    \$EXTRA_OVERRIDES \\
    2>&1 | tee /runstate/logs/run.log
"
