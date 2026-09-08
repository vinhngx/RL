#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/full-rank-vision-natural-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_full_rank_vision_natural_from_950_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
if ! rg -q "messages_for_chat_template = \[\]" "$RUNTIME_REPO/nemo_rl/data/processors.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/vlm_system_prompt_v06.patch"
fi
if ! rg -q "trainable_param_patterns matched no parameters" "$RUNTIME_REPO/nemo_rl/models/automodel/setup.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/vision_full_rank_training_v06.patch"
fi
mkdir -p "$ROOT"/{checkpoints,logs,ray,tmp,wandb}

docker run --rm --gpus all --ipc=host \
  -v "$BREV_ROOT:/brev" -v "$RUNTIME_REPO:/opt/nemo-rl" \
  -v "$HOST_REPO:/workspace/RL:ro" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
cd /opt/nemo-rl
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
$PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/verify_vlm_system_prompt_parity.py \
  --model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/merged-step2 \
  --data /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-hard-negatives-from-950/data/candidate-pool.jsonl \
  --system-prompt /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt
$PY - <<'PY'
from nemo_rl.utils.config import load_config
cfg = load_config("/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_full_rank_vision_natural_from_950_v06.yaml")
assert cfg.policy.dtensor_cfg.lora_cfg.enabled is False
assert cfg.policy.dtensor_cfg.activation_checkpointing is True
assert list(cfg.policy.dtensor_cfg.trainable_param_patterns) == ["*visual*"]
assert cfg.policy.dynamic_batching.train_mb_tokens == 1152
assert cfg.policy.optimizer.kwargs.lr == 2e-8
assert cfg.checkpointing.save_consolidated is True
print("full_rank_vision=True train_mb_tokens=1152 lr=2e-8 steps=4")
PY
' | tee "$ROOT/logs/preflight.log"

docker run --rm --name qwen3-vl-2b-full-rank-vision-natural-from-950 \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BREV_ROOT:/brev" -v "$ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY -e HF_TOKEN -e PYTORCH_ALLOC_CONF=expandable_segments:True \
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
