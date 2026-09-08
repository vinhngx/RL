#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/answer-margin-diagnostic-from-950
mkdir -p "$ROOT"/{artifacts,logs}

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
$PY $SCRIPT/analyze_answer_margins.py \
  --data /brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl \
  --predictions $CAMPAIGN/counterfactual-pair-grpo/evals/step-2-canonical.jsonl \
  --model $CAMPAIGN/counterfactual-pair-grpo/merged-step2 \
  --prompt-file $SCRIPT/prompt_gym_boxed.txt \
  --output $CAMPAIGN/answer-margin-diagnostic-from-950/artifacts/dev200-margins.jsonl \
  --batch-size 4
' 2>&1 | tee "$ROOT/logs/dev200.log"
