#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/system-matched-direction-dpo
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
PATCH=$SCRIPT/dpo_vlm_multimodal_v06.patch
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/dpo_system_matched_direction_from_945_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
mkdir -p "$ROOT"/{data,artifacts,logs,checkpoints,ray,tmp,wandb}
if ! git -C "$RUNTIME_REPO" apply --reverse --check "$PATCH"; then
  git -C "$RUNTIME_REPO" apply "$PATCH"
fi

docker run --rm --name qwen3-vl-2b-system-matched-direction-dpo \
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
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/system-matched-direction-dpo
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
BASE=\$CAMPAIGN/matched-prompt-model-mined-grpo/merged-parent
POOL=\$CAMPAIGN/natural-balanced-vlm-dpo-from-935/data/natural-pool-8192.jsonl
PREDICTIONS=\$ROOT/artifacts/natural-pool-leader945-greedy.jsonl
if [[ ! -f \$PREDICTIONS ]]; then
  \$PY \$SCRIPT/evaluate_hf_lora.py \
    --data \$POOL --model \$BASE \
    --prompt-file \$SCRIPT/prompt_gym_boxed.txt \
    --output \$PREDICTIONS --batch-size 32 --max-new-tokens 128 \
    --temperature 0 --seed 42
fi
\$PY \$SCRIPT/generate_direction_matched_vlm_dpo_pairs.py \
  --data \$POOL --predictions \$PREDICTIONS \
  --train-output \$ROOT/data/train.jsonl \
  --validation-output \$ROOT/data/validation.jsonl \
  --validation-every 16 --system-prompt-file \$SCRIPT/prompt_gym_boxed.txt
cd /opt/nemo-rl
uv run python \$SCRIPT/run_vlm_dpo_v06.py --config '$CONFIG'
" 2>&1 | tee "$ROOT/logs/run.log"
