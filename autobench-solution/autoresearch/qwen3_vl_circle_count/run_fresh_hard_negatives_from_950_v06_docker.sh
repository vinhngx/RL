#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
GYM_SOURCE=/data/ephemeral/nemo-rl/ubuntu/reference-rl-circle-count/3rdparty/Gym-workspace/Gym
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/fresh-hard-negatives-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_fresh_hard_negatives_from_950_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
if ! rg -q "messages_for_chat_template = \[\]" "$RUNTIME_REPO/nemo_rl/data/processors.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/vlm_system_prompt_v06.patch"
fi
mkdir -p "$ROOT"/{artifacts,checkpoints,data,logs,ray,tmp,wandb}

# Build a fully disjoint natural Gym pool for counts that dominate the current
# residual errors, score it with the 190/200 parent, and retain only failures.
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -v "$GYM_SOURCE:/gym:ro" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/fresh-hard-negatives-from-950
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
POOL=$ROOT/data/candidate-pool.jsonl
PREDICTIONS=$ROOT/artifacts/candidate-pool-leader-greedy.jsonl
if [[ ! -s $POOL ]]; then
  $PY $SCRIPT/generate_grpo_count_curriculum.py \
    --generator /gym/resources_servers/circle_count/generate_data.py \
    --output $POOL --counts 5 6 7 8 9 10 --per-count 512 \
    --seed-start 5100000
fi
if [[ ! -f $ROOT/artifacts/candidate-pool-leader-greedy.summary.json ]]; then
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $POOL --model $CAMPAIGN/counterfactual-pair-grpo/merged-step2 \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt \
    --output $PREDICTIONS --batch-size 16 --max-new-tokens 128 \
    --temperature 0 --seed 42
fi
$PY $SCRIPT/mine_circle_count_failures.py \
  --data $POOL --predictions $PREDICTIONS \
  --output $ROOT/data/train-mined-failures.jsonl --minimum-rows 4096 --seed 89
' 2>&1 | tee "$ROOT/logs/mining.log"

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -v "$RUNTIME_REPO:/opt/nemo-rl:ro" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
cd /opt/nemo-rl
PYTHONPATH=/opt/nemo-rl $PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/verify_vlm_system_prompt_parity.py \
  --model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/merged-step2 \
  --data /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-hard-negatives-from-950/data/train-mined-failures.jsonl \
  --system-prompt /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt
' | tee "$ROOT/logs/prompt-parity.log"

docker run --rm --name qwen3-vl-2b-fresh-hard-negative-grpo \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BREV_ROOT:/brev" -v "$ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY -e HF_TOKEN \
  -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
  -e UV_CACHE_DIR=/brev/cache/uv -e TRITON_CACHE_DIR=/brev/cache/triton \
  -e XDG_CACHE_HOME=/brev/cache/xdg -e WANDB_CACHE_DIR=/brev/cache/wandb \
  -e WANDB_DIR="$ROOT/wandb" -e RAY_TMPDIR=/ray -e TMPDIR="$ROOT/tmp" \
  "$IMAGE" bash -lc "
set -euo pipefail
cd /opt/nemo-rl
uv run python examples/run_vlm_grpo.py --config '$CONFIG'
" 2>&1 | tee "$ROOT/logs/run.log"
