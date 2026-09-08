# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Measure local boxed-count decision margins for a Qwen3-VL checkpoint.

The generated response fixes the language prefix. For each neighboring count,
this script compares the logits at the first digit token where the expected and
alternative counts diverge. This is the exact local branch that greedy decoding
must choose, without introducing a different prompt or inference-time helper.
"""

import argparse
import base64
import io
import json
import math
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


BOXED_COUNT = re.compile(r"\\boxed\{(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor-model")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
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


def first_divergent_tokens(
    tokenizer, expected: int, alternative: int
) -> tuple[str, int, int]:
    expected_tokens = tokenizer.encode(str(expected), add_special_tokens=False)
    alternative_tokens = tokenizer.encode(str(alternative), add_special_tokens=False)
    common = 0
    for expected_token, alternative_token in zip(
        expected_tokens, alternative_tokens, strict=False
    ):
        if expected_token != alternative_token:
            break
        common += 1
    if common == len(expected_tokens) or common == len(alternative_tokens):
        raise ValueError(
            f"Counts {expected} and {alternative} do not diverge before one ends"
        )
    common_text = tokenizer.decode(
        expected_tokens[:common], clean_up_tokenization_spaces=False
    )
    return common_text, expected_tokens[common], alternative_tokens[common]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    examples = [json.loads(line) for line in args.data.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    if args.limit is not None:
        examples = examples[: args.limit]
        predictions = predictions[: args.limit]
    if len(examples) != len(predictions):
        raise ValueError(
            f"Data/prediction length mismatch: {len(examples)} != {len(predictions)}"
        )

    prompt = args.prompt_file.read_text().strip()
    processor = AutoProcessor.from_pretrained(args.processor_model or args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()

    rows: list[dict] = []
    jobs: list[dict] = []
    for row_index, (example, prediction) in enumerate(
        zip(examples, predictions, strict=True)
    ):
        image, question, expected, color = unpack_example(example)
        if prediction.get("index", row_index) != row_index:
            raise ValueError(f"Prediction index mismatch at row {row_index}")
        if prediction["expected_count"] != expected:
            raise ValueError(f"Expected-count mismatch at row {row_index}")
        response = prediction["response"]
        match = BOXED_COUNT.search(response)
        if match is None:
            raise ValueError(f"No boxed count in prediction {row_index}: {response!r}")
        response_prefix = response[: match.start(1)]
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        row = {
            "index": row_index,
            "target_color": color,
            "expected_count": expected,
            "generated_count": prediction["boxed_predicted_count"],
            "generated_correct": prediction["boxed_correct"],
            "response_prefix": response_prefix,
            "comparisons": [],
        }
        rows.append(row)
        alternatives = {count for count in (expected - 1, expected + 1) if count >= 0}
        generated = prediction["boxed_predicted_count"]
        if generated is not None and generated != expected:
            alternatives.add(generated)
        for alternative in sorted(alternatives):
            common_text, expected_token, alternative_token = first_divergent_tokens(
                processor.tokenizer, expected, alternative
            )
            jobs.append(
                {
                    "row_index": row_index,
                    "alternative": alternative,
                    "image": image,
                    "text": prompt_text + response_prefix + common_text,
                    "expected_token": expected_token,
                    "alternative_token": alternative_token,
                }
            )

    with torch.inference_mode():
        for start in range(0, len(jobs), args.batch_size):
            batch_jobs = jobs[start : start + args.batch_size]
            inputs = processor(
                text=[job["text"] for job in batch_jobs],
                images=[job["image"] for job in batch_jobs],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            logits = model(**inputs).logits[:, -1, :].float()
            for batch_index, job in enumerate(batch_jobs):
                expected_logit = logits[batch_index, job["expected_token"]].item()
                alternative_logit = logits[
                    batch_index, job["alternative_token"]
                ].item()
                margin = expected_logit - alternative_logit
                rows[job["row_index"]]["comparisons"].append(
                    {
                        "alternative_count": job["alternative"],
                        "expected_token": job["expected_token"],
                        "alternative_token": job["alternative_token"],
                        "expected_logit": expected_logit,
                        "alternative_logit": alternative_logit,
                        "margin": margin,
                        "pair_probability": 1.0 / (1.0 + math.exp(-margin)),
                    }
                )
            print(f"scored={min(start + len(batch_jobs), len(jobs))}/{len(jobs)}", flush=True)

    for row in rows:
        row["minimum_adjacent_margin"] = min(
            comparison["margin"] for comparison in row["comparisons"]
        )
        generated = row["generated_count"]
        row["generated_margin"] = next(
            (
                comparison["margin"]
                for comparison in row["comparisons"]
                if comparison["alternative_count"] == generated
            ),
            None,
        )

    correct_margins = [
        row["minimum_adjacent_margin"] for row in rows if row["generated_correct"]
    ]
    error_margins = [
        row["generated_margin"]
        for row in rows
        if not row["generated_correct"] and row["generated_margin"] is not None
    ]
    summary = {
        "total": len(rows),
        "generated_correct": sum(row["generated_correct"] for row in rows),
        "generated_errors": sum(not row["generated_correct"] for row in rows),
        "correct_min_adjacent_margin": {
            "min": min(correct_margins),
            "p10": percentile(correct_margins, 0.1),
            "median": percentile(correct_margins, 0.5),
        },
        "error_correct_vs_generated_margin": {
            "min": min(error_margins),
            "median": percentile(error_margins, 0.5),
            "max": max(error_margins),
            "within_one_logit": sum(margin >= -1.0 for margin in error_margins),
            "wrong_branch_preferred": sum(margin < 0 for margin in error_margins),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
