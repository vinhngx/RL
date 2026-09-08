#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
GYM_SOURCE=/data/ephemeral/nemo-rl/ubuntu/reference-rl-circle-count/3rdparty/Gym-workspace/Gym
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/spatial-invariance-from-950
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_spatial_invariance_from_950_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
if ! rg -q "messages_for_chat_template = \[\]" "$RUNTIME_REPO/nemo_rl/data/processors.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/vlm_system_prompt_v06.patch"
fi
if ! rg -q 'expected_delta = int' "$RUNTIME_REPO/nemo_rl/algorithms/grpo.py"; then
  git -C "$RUNTIME_REPO" apply "$SCRIPT/spatial_invariance_pair_reward_v06.patch"
fi
mkdir -p "$ROOT"/{data,checkpoints,logs,ray,tmp,wandb}

docker run --rm -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -v "$GYM_SOURCE:/gym:ro" "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/nemo_rl_venv/bin/python
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/spatial-invariance-from-950
[[ -s $ROOT/data/train.jsonl ]] || $PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/generate_edge_aware_circle_count.py \
  --generator /gym/resources_servers/circle_count/generate_data.py \
  --output $ROOT/data/train.jsonl --min-target-count 5 --max-target-count 9 \
  --examples-per-count 256 --seed-offset 5200000 --preserve-pairs
[[ $(wc -l < $ROOT/data/train.jsonl) -eq 2560 ]]
$PY - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/spatial-invariance-from-950/data/train.jsonl").open()]
for i in range(0, len(rows), 2):
    a, b = rows[i:i + 2]
    assert (a["spatial_mode"], b["spatial_mode"]) == ("edge", "center")
    assert a["target_color"] == b["target_color"]
    assert sum(c["color"] == a["target_color"] for c in a["circles"]) == sum(c["color"] == b["target_color"] for c in b["circles"])
print(f"verified_pairs={len(rows) // 2}")
PY
'

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -v "$RUNTIME_REPO:/opt/nemo-rl:ro" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
cd /opt/nemo-rl
PYTHONPATH=/opt/nemo-rl $PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/verify_vlm_system_prompt_parity.py \
  --model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/merged-step2 \
  --data /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/spatial-invariance-from-950/data/train.jsonl \
  --system-prompt /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt
'

docker run --rm --name qwen3-vl-2b-spatial-invariance-from-950 \
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
