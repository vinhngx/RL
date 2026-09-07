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
# GRPO training run for circle-count Qwen3-VL-2B. Runs inside nemo-rl:v060-smoke.
# Usage: pass extra hydra overrides as $@.
set -exo pipefail

export HF_HOME=/brev/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export RAY_TMPDIR=/ray
export TMPDIR=/brev/smoke-vlm/qwen3-vl-2b-grpo-clevr/tmp
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

cd /opt/nemo-rl
uv run -- python examples/run_vlm_grpo.py \
  --config examples/configs/vlm_grpo_circle_count_qwen3vl_2B.yaml \
  "$@"
