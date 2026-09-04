# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Evaluate a Qwen3-VL PEFT adapter on raw Circle Count JSONL examples."""

import argparse
import base64
import io
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


BOXED_COUNT = re.compile(r"\\boxed\{(\d+)\}")
BARE_COUNT = re.compile(r"^\s*(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def decode_image(image_url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    if not image_url.startswith(prefix):
        raise ValueError("Expected a base64 PNG data URL")
    return Image.open(
        io.BytesIO(base64.b64decode(image_url.removeprefix(prefix)))
    ).convert("RGB")


def unpack_example(example: dict) -> tuple[Image.Image, str, int, str]:
    target_color = example["target_color"]
    expected = sum(
        circle["color"] == target_color for circle in example["circles"]
    )
    user_content = example["responses_create_params"]["input"][1]["content"]
    image_url = next(
        item["image_url"] for item in user_content if item["type"] == "input_image"
    )
    question = next(
        item["text"] for item in user_content if item["type"] == "input_text"
    )
    return decode_image(image_url), question, expected, target_color


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - margin, center + margin


def main() -> None:
    args = parse_args()
    with args.data.open() as source:
        examples = [json.loads(line) for line in source]
    if args.limit is not None:
        examples = examples[: args.limit]
    system_prompt = args.prompt_file.read_text().strip()

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_start in range(0, len(examples), args.batch_size):
            batch_examples = examples[batch_start : batch_start + args.batch_size]
            batch = [unpack_example(example) for example in batch_examples]
            images = [item[0] for item in batch]
            messages = [
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
                for image, question, _, _ in batch
            ]
            texts = [
                processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for message in messages
            ]
            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            responses = processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for index, (response, (_, question, expected, color)) in enumerate(
                zip(responses, batch, strict=True), start=batch_start
            ):
                boxed_match = BOXED_COUNT.search(response)
                bare_match = BARE_COUNT.fullmatch(response)
                boxed_predicted = (
                    int(boxed_match.group(1)) if boxed_match else None
                )
                predicted = (
                    boxed_predicted
                    if boxed_predicted is not None
                    else int(bare_match.group(1)) if bare_match else None
                )
                predictions.append(
                    {
                        "index": index,
                        "question": question,
                        "target_color": color,
                        "expected_count": expected,
                        "predicted_count": predicted,
                        "correct": predicted == expected,
                        "boxed_predicted_count": boxed_predicted,
                        "boxed_correct": boxed_predicted == expected,
                        "response": response,
                    }
                )
            correct = sum(prediction["correct"] for prediction in predictions)
            print(
                f"evaluated={len(predictions)}/{len(examples)} "
                f"correct={correct} accuracy={correct / len(predictions):.4f}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    correct = sum(prediction["correct"] for prediction in predictions)
    parseable = sum(
        prediction["predicted_count"] is not None for prediction in predictions
    )
    boxed_correct = sum(prediction["boxed_correct"] for prediction in predictions)
    boxed_parseable = sum(
        prediction["boxed_predicted_count"] is not None for prediction in predictions
    )
    lower, upper = wilson_interval(correct, len(predictions))
    by_count: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_color: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for prediction in predictions:
        for key, value in (
            (by_count, prediction["expected_count"]),
            (by_color, prediction["target_color"]),
        ):
            key[value]["total"] += 1
            key[value]["correct"] += int(prediction["correct"])
    summary = {
        "total": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "wilson_95": [lower, upper],
        "parseable": parseable,
        "boxed_correct": boxed_correct,
        "boxed_accuracy": boxed_correct / len(predictions),
        "boxed_parseable": boxed_parseable,
        "elapsed_seconds": elapsed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "by_expected_count": dict(sorted(by_count.items())),
        "by_target_color": dict(sorted(by_color.items())),
    }
    with args.output.open("w") as destination:
        for prediction in predictions:
            destination.write(json.dumps(prediction, separators=(",", ":")) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
