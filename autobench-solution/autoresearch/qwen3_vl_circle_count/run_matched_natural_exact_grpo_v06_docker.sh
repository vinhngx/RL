#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/matched-natural-exact-grpo
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_matched_natural_exact_from_945_v06.yaml
POOL=$CAMPAIGN/natural-balanced-vlm-dpo-from-935/data/natural-pool-8192.jsonl
PARENT=$CAMPAIGN/matched-prompt-model-mined-grpo/merged-parent

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
if ! rg -q "messages_for_chat_template = \[\]" "$RUNTIME_REPO/nemo_rl/data/processors.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/vlm_system_prompt_v06.patch"
fi
[[ $(wc -l < "$POOL") -eq 8192 ]]
[[ -f "$PARENT/model.safetensors.index.json" ]]
mkdir -p "$ROOT"/{checkpoints,logs,ray,tmp,wandb}

docker run --rm --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$HOST_REPO:/workspace/RL:ro" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
$PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/verify_vlm_system_prompt_parity.py \
  --model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/matched-prompt-model-mined-grpo/merged-parent \
  --data /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/natural-balanced-vlm-dpo-from-935/data/natural-pool-8192.jsonl \
  --system-prompt /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt
' | tee "$ROOT/logs/prompt-parity.log"

docker run --rm --name qwen3-vl-2b-matched-natural-exact-grpo \
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
