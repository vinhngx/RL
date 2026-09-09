#!/usr/bin/env bash
set -euo pipefail

STEP=${1:?usage: $0 STEP [LIMIT]}
LIMIT=${2:-200}
IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/counterfactual-boxed-token-vision-simpo-from-950
mkdir -p "$ROOT/evals" "$ROOT/logs"

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc "
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=\$CAMPAIGN/counterfactual-boxed-token-vision-simpo-from-950
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
\$PY \$SCRIPT/evaluate_hf_lora.py \
  --data /brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl \
  --model \$CAMPAIGN/counterfactual-pair-grpo/merged-step2 \
  --adapter \$ROOT/checkpoints/step_$STEP/policy/weights/model \
  --prompt-file \$SCRIPT/prompt_gym_boxed.txt \
  --output \$ROOT/evals/step-$STEP-canonical-$LIMIT.jsonl \
  --limit $LIMIT --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
" 2>&1 | tee "$ROOT/logs/canonical-step-$STEP-$LIMIT.log"
