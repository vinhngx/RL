#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}
HOST_REPO=/home/ubuntu/RL
BREV_ROOT=/data/ephemeral/nemo-rl/ubuntu
GYM_SOURCE=$BREV_ROOT/reference-rl-circle-count/3rdparty/Gym-workspace/Gym
CAMPAIGN=$BREV_ROOT/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98
ROOT=$CAMPAIGN/fresh-final-1k-best-checkpoints
DATA=$ROOT/data/profile-seed30000000-n1000.jsonl
RESULTS=$ROOT/evals
SCRIPT=/workspace/RL/autobench-solution/autoresearch/qwen3_vl_circle_count
SEED_OFFSET=30000000
SAMPLE_COUNT=1000

mkdir -p "$ROOT"/{data,evals,logs,tmp}

if [[ ! -s "$DATA" ]]; then
  docker run --rm \
    -v "$GYM_SOURCE:/gym:ro" -v "$ROOT:/out" \
    "$IMAGE" bash -lc "/opt/nemo_rl_venv/bin/python \
      /gym/resources_servers/circle_count/generate_data.py \
      --n $SAMPLE_COUNT --seed-offset $SEED_OFFSET \
      --out /out/data/profile-seed30000000-n1000.jsonl"
fi
[[ $(wc -l < "$DATA") -eq $SAMPLE_COUNT ]]

DATA_SHA256=$(sha256sum "$DATA" | awk '{print $1}')
python3 - "$ROOT/manifest.json" "$DATA" "$DATA_SHA256" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, data_path, sha256 = sys.argv[1:]
manifest = {
    "dataset": data_path,
    "generator": "/data/ephemeral/nemo-rl/ubuntu/reference-rl-circle-count/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py",
    "sample_count": 1000,
    "seed_first": 30000000,
    "seed_last": 30000999,
    "sha256": sha256,
}
path = Path(manifest_path)
if path.exists() and json.loads(path.read_text()) != manifest:
    raise SystemExit("existing frozen dataset manifest does not match")
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

labels=(
  sft_step150_dev184
  dynamic_grpo_dev186
  strong_dpo_dev187
  natural_scaled_dpo_dev188
  edge_grpo_dev189
  counterfactual_grpo_dev190
)
models=(
  /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-match-external/github-dataset-verbatim/merged-step150
  /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/dynamic-sampling-exact-from-grpo8/merged-step2
  /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/iterative-vlm-dpo-from-935/merged-parent
  /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/direction-matched-vlm-dpo-from-94/merged-parent
  /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/matched-prompt-model-mined-grpo/merged-parent
  /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/merged-step2
)

for index in "${!labels[@]}"; do
  label=${labels[$index]}
  model=${models[$index]}
  output=$RESULTS/$label.jsonl
  [[ -s "$output" && -s "${output%.jsonl}.summary.json" ]] && continue
  docker run --rm --name "qwen3-vl-2b-final1k-$index" --gpus all --ipc=host \
    -v "$HOST_REPO:/workspace/RL:ro" -v "$BREV_ROOT:/brev" \
    -e HF_HOME=/brev/cache/huggingface \
    -e HF_HUB_CACHE=/brev/cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/brev/cache/huggingface/transformers \
    -e XDG_CACHE_HOME=/brev/cache/xdg -e TMPDIR=/brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-final-1k-best-checkpoints/tmp \
    "$IMAGE" bash -lc "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
      $SCRIPT/evaluate_hf_lora.py \
      --data /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-final-1k-best-checkpoints/data/profile-seed30000000-n1000.jsonl \
      --model '$model' --prompt-file $SCRIPT/prompt_gym_boxed.txt \
      --output /brev/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-final-1k-best-checkpoints/evals/$label.jsonl \
      --batch-size 4 --max-new-tokens 128 --temperature 0 --seed 42" \
    2>&1 | tee "$ROOT/logs/$label.log"
done

python3 "$HOST_REPO/autobench-solution/autoresearch/qwen3_vl_circle_count/summarize_fresh_checkpoint_eval.py" \
  --results-root "$RESULTS" --output "$ROOT/comparison.json"
cat "$ROOT/comparison.md"
