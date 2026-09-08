#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
RUNTIME_REPO=/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/dihedral-invariance-from-950
SOURCE=$CAMPAIGN/fresh-hard-negatives-from-950/data/candidate-pool.jsonl
SCRIPT=$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count
CONFIG=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/grpo_dihedral_invariance_from_950_v06.yaml

if [[ -f "$HOST_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOST_REPO/.env"
  set +a
fi
mkdir -p "$ROOT"/{data,checkpoints,logs,ray,tmp,wandb}

docker run --rm -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/nemo_rl_venv/bin/python
ROOT=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dihedral-invariance-from-950
SOURCE=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-hard-negatives-from-950/data/candidate-pool.jsonl
[[ -s $ROOT/data/train.jsonl ]] || $PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/generate_dihedral_invariance_pairs.py \
  --input $SOURCE --output $ROOT/data/train.jsonl --pairs 2048
[[ $(wc -l < $ROOT/data/train.jsonl) -eq 4096 ]]
$PY - <<'PY'
import json
from pathlib import Path
rows=[json.loads(line) for line in Path("/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dihedral-invariance-from-950/data/train.jsonl").open()]
for i in range(0,len(rows),2):
    a,b=rows[i:i+2]
    assert a["pair_id"] == b["pair_id"] and a["target_color"] == b["target_color"]
    assert sum(c["color"] == a["target_color"] for c in a["circles"]) == sum(c["color"] == b["target_color"] for c in b["circles"])
print(f"verified_dihedral_pairs={len(rows)//2}")
PY
'

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -v "$RUNTIME_REPO:/opt/nemo-rl:ro" -e HF_HOME=/brev/cache/huggingface \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
cd /opt/nemo-rl
PYTHONPATH=/opt/nemo-rl $PY /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/verify_vlm_system_prompt_parity.py \
  --model /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/merged-step2 \
  --data /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dihedral-invariance-from-950/data/train.jsonl \
  --system-prompt /workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt
'

docker run --rm --name qwen3-vl-2b-dihedral-invariance-from-950 \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BREV_ROOT:/brev" -v "$ROOT/ray:/ray" \
  -v "$RUNTIME_REPO:/opt/nemo-rl" -v "$HOST_REPO:/workspace/RL:ro" \
  -e WANDB_API_KEY -e HF_TOKEN -e HF_HOME=/brev/cache/huggingface \
  -e HF_HUB_CACHE=/brev/cache/huggingface/hub -e HF_DATASETS_CACHE=/brev/cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers -e UV_CACHE_DIR=/brev/cache/uv \
  -e TRITON_CACHE_DIR=/brev/cache/triton -e XDG_CACHE_HOME=/brev/cache/xdg \
  -e WANDB_CACHE_DIR=/brev/cache/wandb -e WANDB_DIR="$ROOT/wandb" \
  -e RAY_TMPDIR=/ray -e TMPDIR="$ROOT/tmp" \
  "$IMAGE" bash -lc "
set -euo pipefail
cd /opt/nemo-rl
uv run python examples/run_vlm_grpo.py --config '$CONFIG'
" 2>&1 | tee "$ROOT/logs/run.log"
