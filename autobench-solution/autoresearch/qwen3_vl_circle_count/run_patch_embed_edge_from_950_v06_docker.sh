#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/patch-embed-edge-grpo-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_patch_embed_edge_from_950_v06.yaml
MAX_STEPS=${MAX_STEPS:-1}

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
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc "
set -euo pipefail
cd /opt/nemo-rl
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
\$PY - <<'PY'
from nemo_rl.utils.config import load_config
cfg = load_config('$CONFIG')
assert cfg.policy.dtensor_cfg.lora_cfg.enabled is False
assert list(cfg.policy.dtensor_cfg.trainable_param_patterns) == [
    '*visual.patch_embed*', '*visual.pos_embed*'
]
assert cfg.policy.dynamic_batching.train_mb_tokens == 1152
assert cfg.policy.optimizer.kwargs.lr == 1e-7
assert cfg.checkpointing.save_consolidated is True
assert cfg.data.default.system_prompt_file.endswith('prompt_gym_boxed.txt')
print('patch_embed_and_pos_only=True lr=1e-7 max_steps=$MAX_STEPS')
PY
" | tee "$ROOT/logs/preflight.log"

docker run --rm --name qwen3-vl-2b-patch-embed-edge-from-950 \
  --gpus all --ipc=host --cap-add=SYS_PTRACE \
  --ulimit memlock=-1 --ulimit stack=67108864 \
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
uv run python examples/run_vlm_grpo.py --config '$CONFIG' grpo.max_num_steps=$MAX_STEPS
" 2>&1 | tee "$ROOT/logs/run-max-$MAX_STEPS.log"
