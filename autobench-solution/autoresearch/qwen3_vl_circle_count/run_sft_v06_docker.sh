#!/usr/bin/env bash
# Launch the Qwen3-VL SFT recipe against the official NeMo RL v0.6 image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}"
export PATCH_PATH=""
export BACKEND_PATCH_PATH="${BACKEND_PATCH_PATH:-${SCRIPT_DIR}/dtensor_qwen3_vl_v06.patch}"
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor1tp1-ac-b128-mb4-200-v06.v1.yaml}"
export CONFIG_PARENT_PATH="${CONFIG_PARENT_PATH:-${SCRIPT_DIR}/vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor1tp1-b128-200.v1.yaml}"

exec "${SCRIPT_DIR}/run_sft_v05_docker.sh"
