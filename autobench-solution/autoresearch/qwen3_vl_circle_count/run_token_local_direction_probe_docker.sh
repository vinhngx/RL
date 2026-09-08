#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/token-local-direction-probe-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
mkdir -p "$ROOT"/{artifacts,logs}

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
$PY $SCRIPT/analyze_counterfactual_token_direction.py \
  --data $CAMPAIGN/fresh-paired-counterfactual-from-950/data/train.jsonl \
  --model $CAMPAIGN/counterfactual-pair-grpo/merged-step2 \
  --prompt-file $SCRIPT/prompt_gym_boxed.txt \
  --output $CAMPAIGN/token-local-direction-probe-from-950/artifacts/pairs256.jsonl \
  --pairs 256
' 2>&1 | tee "$ROOT/logs/run.log"
