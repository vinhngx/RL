#!/usr/bin/env bash
# Run the reference-compatible Qwen3-VL-2B Gym SFT recipe on NeMo-RL v0.6.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}"
export PATCH_PATH=""
export BACKEND_PATCH_PATH="${BACKEND_PATCH_PATH:-${SCRIPT_DIR}/dtensor_qwen3_vl_v06.patch}"
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor-lora-all-linear-r8-lr1e4-b128-mb8-200-gym6k-v06.yaml}"
export CONFIG_PARENT_PATH="${CONFIG_PARENT_PATH:-${SCRIPT_DIR}/vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor1tp1-b128-200.v1.yaml}"
export SFT_ANSWER_STYLE="${SFT_ANSWER_STYLE:-natural_box}"

exec "${SCRIPT_DIR}/run_sft_v05_docker.sh"
