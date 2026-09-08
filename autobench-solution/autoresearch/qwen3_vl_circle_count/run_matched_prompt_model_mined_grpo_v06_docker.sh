#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/matched-prompt-model-mined-grpo
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_matched_prompt_model_mined_from_945_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
if ! rg -q "messages_for_chat_template = \[\]" "$RUNTIME_REPO/nemo_rl/data/processors.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/vlm_system_prompt_v06.patch"
fi

mkdir -p "$ROOT"/{artifacts,checkpoints,data,logs,ray,tmp,wandb,merged-parent}

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/matched-prompt-model-mined-grpo
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
BASE=$CAMPAIGN/direction-matched-vlm-dpo-from-94/merged-parent
ADAPTER=$CAMPAIGN/edge-aware-exact-grpo/checkpoints/step_2/policy/weights/model
POOL=$CAMPAIGN/model-mined-hard-negatives-from-93/data/candidate-pool.jsonl
if [[ ! -f $ROOT/merged-parent/model.safetensors.index.json ]]; then
  $PY $SCRIPT/merge_qwen3vl_lora.py \
    --base-model $BASE --adapter $ADAPTER --output $ROOT/merged-parent
fi
if [[ ! -f $ROOT/artifacts/candidate-pool-leader-greedy.summary.json ]]; then
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $POOL --model $ROOT/merged-parent \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt \
    --output $ROOT/artifacts/candidate-pool-leader-greedy.jsonl \
    --batch-size 16 --max-new-tokens 128 --temperature 0 --seed 42
fi
$PY $SCRIPT/mine_circle_count_failures.py \
  --data $POOL \
  --predictions $ROOT/artifacts/candidate-pool-leader-greedy.jsonl \
  --output $ROOT/data/train-mined-failures.jsonl --minimum-rows 2048
'

docker run --rm --name qwen3-vl-2b-matched-prompt-model-mined-grpo \
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
