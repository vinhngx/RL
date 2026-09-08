#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/full-rank-vision-sft-from-950
PARENT=$CAMPAIGN/counterfactual-pair-grpo/merged-step2

mkdir -p "$ROOT/evals"
for STEP in 2 4 6 8; do
  RAW=$ROOT/checkpoints/step_$STEP/policy/weights/model/shard-00001-model-00001-of-00001.safetensors
  MODEL=$ROOT/hf-step-$STEP
  [[ -f "$RAW" ]] || continue
  mkdir -p "$MODEL"
  cp "$PARENT/config.json" "$MODEL/config.json"
  [[ ! -f "$PARENT/generation_config.json" ]] || cp "$PARENT/generation_config.json" "$MODEL/generation_config.json"
  ln -sfn "../checkpoints/step_$STEP/policy/weights/model/shard-00001-model-00001-of-00001.safetensors" "$MODEL/model.safetensors"
done

docker run --rm --gpus all --ipc=host \
  -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
  -e HF_HOME=/brev/cache/huggingface "$IMAGE" bash -lc '
set -euo pipefail
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
CAMPAIGN=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/full-rank-vision-sft-from-950
PARENT=$CAMPAIGN/counterfactual-pair-grpo/merged-step2
DATA=/brev/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
for STEP in 2 4 6 8; do
  MODEL=$ROOT/hf-step-$STEP
  SMOKE=$ROOT/evals/step-$STEP-smoke64.jsonl
  $PY $SCRIPT/evaluate_hf_lora.py \
    --data $DATA --model $MODEL --processor-model $PARENT \
    --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $SMOKE \
    --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42 --limit 64
  SUMMARY=${SMOKE%.jsonl}.summary.json
  CORRECT=$(sed -n "s/^  \"correct\": *\([0-9][0-9]*\).*/\1/p" "$SUMMARY")
  BOXED=$(sed -n "s/^  \"boxed_parseable\": *\([0-9][0-9]*\).*/\1/p" "$SUMMARY")
  echo "step=$STEP smoke_correct=$CORRECT smoke_boxed=$BOXED"
  if (( CORRECT >= 59 && BOXED == 64 )); then
    $PY $SCRIPT/evaluate_hf_lora.py \
      --data $DATA --model $MODEL --processor-model $PARENT \
      --prompt-file $SCRIPT/prompt_gym_boxed.txt --output $ROOT/evals/step-$STEP-canonical.jsonl \
      --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42
  fi
done
' 2>&1 | tee "$ROOT/logs/canonical-eval.log"
