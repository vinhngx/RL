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
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _parse_json_counts(response_text: str) -> dict[str, Any] | None:
    stripped = response_text.strip()
    candidates = [stripped]
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced_match:
        candidates.append(fenced_match.group(1))
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_counts(
    parsed_counts: dict[str, Any] | None, expected_counts: dict[str, int]
) -> dict[str, int] | None:
    if parsed_counts is None:
        return None
    lowered_counts = {
        str(color).lower(): value for color, value in parsed_counts.items()
    }
    normalized = {}
    for color in expected_counts:
        value = lowered_counts.get(color.lower())
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            normalized[color] = value
            continue
        if isinstance(value, str) and value.strip().isdigit():
            normalized[color] = int(value)
            continue
        return None
    return normalized


def _score_response(
    response_text: str, expected_counts: dict[str, int]
) -> tuple[float, bool, dict[str, int] | None]:
    parsed_counts = _normalize_counts(
        _parse_json_counts(response_text), expected_counts
    )
    if parsed_counts is None:
        return 0.0, False, None
    correct_colors = sum(
        parsed_counts.get(color, -1) == count
        for color, count in expected_counts.items()
    )
    reward = correct_colors / len(expected_counts) if expected_counts else 0.0
    return reward, reward == 1.0, parsed_counts


def _load_rows(data_path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with data_path.open() as data_file:
        for line in data_file:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _expected_counts(row: dict[str, Any]) -> dict[str, int]:
    return json.loads(row["messages"][-1]["content"])


def _user_images(row: dict[str, Any]) -> list[Image.Image]:
    user_content = row["messages"][1]["content"]
    image_paths = [item["image"] for item in user_content if item["type"] == "image"]
    return [Image.open(image_path).convert("RGB") for image_path in image_paths]


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row["messages"][:-1]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(args.data_path, args.limit)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    rewards = []
    exact_matches = []
    started_at = time.time()
    with args.output_path.open("w") as output_file:
        for index, row in enumerate(tqdm(rows, desc="Evaluating star_count")):
            messages = _prompt_messages(row)
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images = _user_images(row)
            inputs = processor(text=[prompt], images=images, return_tensors="pt").to(
                model.device
            )
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            response_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            expected_counts = _expected_counts(row)
            reward, exact_match, parsed_counts = _score_response(
                response_text, expected_counts
            )
            rewards.append(reward)
            exact_matches.append(exact_match)
            output_file.write(
                json.dumps(
                    {
                        "index": index,
                        "reward": reward,
                        "exact_match": exact_match,
                        "expected_counts": expected_counts,
                        "parsed_counts": parsed_counts,
                        "response_text": response_text,
                    }
                )
                + "\n"
            )

    elapsed_sec = time.time() - started_at
    accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    metrics = {
        "model": args.model,
        "data_path": str(args.data_path),
        "num_examples": len(rows),
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "elapsed_sec": elapsed_sec,
        "examples_per_sec": len(rows) / elapsed_sec if elapsed_sec > 0 else 0.0,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on star_count SFT-format JSONL."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
