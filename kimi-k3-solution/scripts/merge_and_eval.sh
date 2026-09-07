# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Merge a LoRA checkpoint and gym-eval it on the 200-task profile set.
# Usage inside container: CKPT=<policy/weights/model path> TAG=<name> bash merge_and_eval.sh
set -exo pipefail
export HF_HOME=/brev/cache/huggingface
export RAY_TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false

OUT_DIR=/brev/circle-count-gym/evals/$TAG
MERGED=/brev/circle-count-gym/merged-$TAG
mkdir -p $OUT_DIR

cd /opt/nemo-rl
if [ ! -d "$MERGED" ]; then
  CKPT=$CKPT OUT=$MERGED uv run python /brev/circle-count-gym/merge_lora.py
fi

UV="uv run --directory /opt/nemo-rl --extra vllm --extra nemo_gym --"
GYM_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym

nohup $UV vllm serve $MERGED --served-model-name $TAG \
  --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
  > $OUT_DIR/vllm_server.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then echo "vLLM ready"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died"; tail -20 $OUT_DIR/vllm_server.log; exit 1; fi
  sleep 10
done

cd $GYM_DIR
nohup $UV ng_run \
  "+config_paths=[resources_servers/circle_count/configs/circle_count.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +policy_base_url=http://127.0.0.1:8000/v1 +policy_api_key=dummy +policy_model_name=$TAG \
  > $OUT_DIR/gym_servers.log 2>&1 &
GYM_PID=$!
sleep 50
for i in $(seq 1 30); do
  if $UV ng_status 2>/dev/null | grep -q circle_count; then echo "gym ready"; break; fi
  if ! kill -0 $GYM_PID 2>/dev/null; then echo "gym died"; tail -10 $OUT_DIR/gym_servers.log; exit 1; fi
  sleep 10
done

$UV ng_collect_rollouts \
  +agent_name=circle_count_simple_agent \
  +input_jsonl_fpath=/brev/circle-count-gym/data/profile.jsonl \
  +output_jsonl_fpath=$OUT_DIR/rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 1024, temperature: 1.0}"

kill $VLLM_PID $GYM_PID 2>/dev/null || true
sleep 5
echo "EVAL_DONE $TAG"
