#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
PAIRS=${PAIRS:-128}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/circle-color-token-probe-from-950
mkdir -p "$ROOT"/{artifacts,logs,tmp}

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface -e TMPDIR="$ROOT/tmp" \
  "$IMAGE" bash -lc "
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
\$PY \$SCRIPT/analyze_circle_color_token_alignment.py \
  --data \$CAMPAIGN/fresh-paired-counterfactual-from-950/data/train.jsonl \
  --model \$CAMPAIGN/counterfactual-pair-grpo/merged-step2 \
  --prompt-file \$SCRIPT/prompt_gym_boxed.txt \
  --output \$CAMPAIGN/circle-color-token-probe-from-950/artifacts/circles.jsonl \
  --pairs '$PAIRS'
" 2>&1 | tee "$ROOT/logs/run.log"
