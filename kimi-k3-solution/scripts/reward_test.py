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

from nemo_rl.environments.rewards import boxed_numeric_reward

cases = [
    ("7", "The image has 7 red circles. \\boxed{7}"),
    ("7", "Counting... \\boxed{3}"),
    ("7", "no box here"),
    ("12", "\\boxed{12}"),
    ("12", "\\boxed{012}"),
]
for gt, resp in cases:
    print(repr(resp[-25:]), "gt", gt, "->", boxed_numeric_reward(gt, resp))
