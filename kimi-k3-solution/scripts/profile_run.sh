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
# Profiles out-of-box accuracy of Qwen/Qwen3-VL-2B-Instruct on circle_count gym.
# Runs inside nemo-rl:v060-smoke with /brev mount and /opt/nemo-rl worktree mount.
set -exo pipefail

export HF_HOME=/brev/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export RAY_TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false

UV="uv run --directory /opt/nemo-rl --extra vllm --extra nemo_gym --"
GYM_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
EXP=/brev/circle-count-gym
cd /opt/nemo-rl

# 1. vLLM OpenAI server
nohup $UV vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --port 8000 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  > $EXP/vllm_server.log 2>&1 &
VLLM_PID=$!
echo "vllm pid: $VLLM_PID"

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "vLLM ready"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM process died"; tail -30 $EXP/vllm_server.log; exit 1
  fi
  sleep 10
done

# 2. Gym head server + circle_count resources server + agent + vllm_model proxy
cd $GYM_DIR
nohup $UV ng_run \
  "+config_paths=[resources_servers/circle_count/configs/circle_count.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +policy_base_url=http://127.0.0.1:8000/v1 \
  +policy_api_key=dummy \
  +policy_model_name=Qwen/Qwen3-VL-2B-Instruct \
  > $EXP/gym_servers.log 2>&1 &
GYM_PID=$!
echo "gym pid: $GYM_PID"
sleep 60

for i in $(seq 1 30); do
  if $UV ng_status 2>/dev/null | grep -q circle_count; then
    echo "gym ready"
    break
  fi
  if ! kill -0 $GYM_PID 2>/dev/null; then
    echo "gym server died"; tail -30 $EXP/gym_servers.log; exit 1
  fi
  sleep 10
done

# 3. Collect rollouts
$UV ng_collect_rollouts \
  +agent_name=circle_count_simple_agent \
  +input_jsonl_fpath=$EXP/data/profile.jsonl \
  +output_jsonl_fpath=$EXP/profile/rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 1024, temperature: 1.0}"

# 4. Reward profile + aggregate
$UV ng_reward_profile \
  +input_jsonl_fpath=$EXP/data/profile.jsonl \
  +rollouts_jsonl_fpath=$EXP/profile/rollouts.jsonl \
  +output_jsonl_fpath=$EXP/profile/profiled.jsonl \
  +pass_threshold=1.0

$UV python $GYM_DIR/scripts/print_aggregate_results.py \
  +jsonl_fpath=$EXP/profile/profiled.jsonl || true

echo "PROFILE_DONE"
