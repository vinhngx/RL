#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/home/ubuntu/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
EXP_NAME=vlm-group-id-rl-zvp-from-950 \
  bash "$SCRIPT/run_rl_zvp_from_950_eval_docker.sh"
