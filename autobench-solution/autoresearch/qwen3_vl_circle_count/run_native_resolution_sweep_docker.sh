#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/native-resolution-sweep

mkdir -p "$ROOT/evals" "$ROOT/logs"
docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface \
  "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/native-resolution-sweep
MODEL=$CAMPAIGN/direction-matched-vlm-dpo-from-94/merged-parent
ADAPTER=$CAMPAIGN/natural-dpo-delta-scale-search/adapters/scale-0p75
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for PIXELS in 655360 1048576 1441792 2097152; do
  OUT=$ROOT/evals/pixels-$PIXELS-smoke64.jsonl
  [[ -f ${OUT%.jsonl}.summary.json ]] && continue
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $MODEL --adapter $ADAPTER \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $OUT \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 \
    --image-pixels $PIXELS --limit 64
done
' 2>&1 | tee "$ROOT/logs/sweep.log"
