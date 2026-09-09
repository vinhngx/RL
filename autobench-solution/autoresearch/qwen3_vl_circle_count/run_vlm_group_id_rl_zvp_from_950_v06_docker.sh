#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/home/ubuntu/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
export VLM_GROUP_IDENTITY=1
export EXP_NAME=vlm-group-id-rl-zvp-from-950
export CONFIG_FILE=grpo_vlm_group_id_rl_zvp_from_950_v06.yaml
export CONTAINER_NAME=qwen3-vl-2b-group-id-rl-zvp
exec bash "$SCRIPT/run_rl_zvp_from_950_v06_docker.sh"
