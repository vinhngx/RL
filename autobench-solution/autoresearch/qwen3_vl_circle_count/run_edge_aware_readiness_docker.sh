#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
GYM_SOURCE=/data/ephemeral/nemo-rl/ubuntu/reference-rl-circle-count/3rdparty/Gym-workspace/Gym
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/edge-aware-exact-grpo

mkdir -p "$ROOT"/{data,evals,logs}
docker run --rm -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -v "$GYM_SOURCE:/gym:ro" "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/nemo_rl_venv/bin/python
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/edge-aware-exact-grpo
[[ -s $ROOT/data/train.jsonl ]] || $PY $SCRIPT/generate_edge_aware_circle_count.py \
  --generator /gym/resources_servers/circle_count/generate_data.py \
  --output $ROOT/data/train.jsonl --examples-per-count 200 --seed-offset 3800000
[[ -s $ROOT/data/readiness.jsonl ]] || $PY $SCRIPT/generate_edge_aware_circle_count.py \
  --generator /gym/resources_servers/circle_count/generate_data.py \
  --output $ROOT/data/readiness.jsonl --examples-per-count 40 --seed-offset 3900000
'

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/edge-aware-exact-grpo
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
$PY $SCRIPT/evaluate_hf_lora.py \
  --data $ROOT/data/readiness.jsonl \
  --model $CAMPAIGN/direction-matched-vlm-dpo-from-94/merged-parent \
  --adapter $CAMPAIGN/natural-dpo-delta-scale-search/adapters/scale-0p75 \
  --prompt-file $SCRIPT/prompt_gym_boxed.txt \
  --output $ROOT/evals/leader-readiness-200.jsonl \
  --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 --limit 200
' 2>&1 | tee "$ROOT/logs/readiness.log"
