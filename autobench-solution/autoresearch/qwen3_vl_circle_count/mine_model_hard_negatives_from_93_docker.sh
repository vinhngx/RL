#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}"
DATA_ROOT="${DATA_ROOT:-/data/ephemeral/nemo-rl/ubuntu}"
REPO="${REPO:-/home/ubuntu/RL}"
CAMPAIGN="$DATA_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98"
ROOT="$CAMPAIGN/model-mined-hard-negatives-from-93"

mkdir -p "$ROOT"/{data,artifacts,logs}

docker run --rm --gpus all --ipc=host \
    -v "$REPO:/workspace/RL:ro" \
    -v "$DATA_ROOT:/brev" \
    -e HF_HOME=/brev/cache/huggingface \
    -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
    "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT="$CAMPAIGN/model-mined-hard-negatives-from-93"
POOL="$ROOT/data/candidate-pool.jsonl"
PREDICTIONS="$ROOT/artifacts/candidate-pool-leader-greedy.jsonl"
"$PY" "$SCRIPT/generate_balanced_circle_count.py" \
    --generator /brev/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \
    --output "$POOL" \
    --count-quotas 3:256,5:512,7:768,8:768,9:512,11:512,13:384,14:384 \
    --color-cycle red,blue,green,yellow,purple,orange,cyan,pink \
    --min-total-circles 14 --max-total-circles 20 --seed-offset 1800000
"$PY" "$SCRIPT/evaluate_hf_lora.py" \
    --data "$POOL" \
    --prompt-file "$SCRIPT/prompt_gym_boxed.txt" \
    --output "$PREDICTIONS" \
    --model "$CAMPAIGN/dynamic-sampling-exact-from-grpo8/merged-step2" \
    --batch-size 8 --max-new-tokens 128 --temperature 0 --seed 42
"$PY" "$SCRIPT/mine_circle_count_failures.py" \
    --data "$POOL" --predictions "$PREDICTIONS" \
    --output "$ROOT/data/train-mined-failures.jsonl" --minimum-rows 2048
' 2>&1 | tee "$ROOT/logs/mining.log"
