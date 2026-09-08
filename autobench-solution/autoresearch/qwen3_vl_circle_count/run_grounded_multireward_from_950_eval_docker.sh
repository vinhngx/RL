#!/usr/bin/env bash
set -euo pipefail

STEP=${1:?usage: $0 STEP [LIMIT]}
LIMIT=${2:-64}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/paper-grounded-multireward-from-950
BASE=$CAMPAIGN/counterfactual-pair-grpo/merged-step2
ADAPTER=$ROOT/checkpoints/step_$STEP/policy/weights/model
PROFILE=$BREV_ROOT/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
OUT=$ROOT/evals/step-$STEP-canonical-${LIMIT}.jsonl
PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python

mkdir -p "$ROOT/evals"
docker run --rm --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" -v "$HOST_REPO:/workspace/RL:ro" \
  nvcr.io/nvidia/nemo-rl:v0.6.0 bash -lc "
set -euo pipefail
$PYTHON /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/evaluate_hf_lora.py \
  --base /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/merged-step2 \
  --adapter /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/paper-grounded-multireward-from-950/checkpoints/step_$STEP/policy/weights/model \
  --profile /brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl \
  --prompt-file /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt \
  --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/paper-grounded-multireward-from-950/evals/step-$STEP-canonical-${LIMIT}.jsonl \
  --limit $LIMIT --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
"
