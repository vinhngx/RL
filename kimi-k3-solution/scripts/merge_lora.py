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

"""Merge the circle-count LoRA adapter into Qwen3-VL-2B for faithful eval serve."""
import os

import torch
from safetensors.torch import load_file

CKPT = os.environ.get(
    "CKPT",
    "/brev/circle-count-gym/ckpts/circlecount_Qwen/Qwen3-VL-2B-Instruct/step_110/policy/weights/model",
)
BASE = "Qwen/Qwen3-VL-2B-Instruct"
OUT = os.environ.get("OUT", "/brev/circle-count-gym/merged-step110")

adapter = load_file(os.path.join(CKPT, "adapter_model.safetensors"))
pairs = {}
for key in adapter:
    if key.endswith(".lora_A.weight"):
        module = key[: -len(".lora_A.weight")].removeprefix("base_model.model.")
        pairs[module] = (adapter[key], adapter[key[: -len("lora_A.weight")] + "lora_B.weight"])
print(f"{len(pairs)} LoRA modules")

scale = 32 / 8  # lora_alpha / r

from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
sd = model.state_dict()
missing = [m for m in pairs if m + ".weight" not in sd]
if missing:
    raise SystemExit(f"LoRA modules missing from HF state dict: {missing[:8]}")

with torch.no_grad():
    for module, (A, B) in pairs.items():
        w = sd[module + ".weight"]
        delta = (B.to(torch.float32) @ A.to(torch.float32)) * scale
        w.add_(delta.to(w.dtype))

model.save_pretrained(OUT, safe_serialization=True)
AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
print("merged model at", OUT)
