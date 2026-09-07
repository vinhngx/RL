#!/usr/bin/env python3
"""Build image-conditioned chosen/rejected pairs from a scored Circle Count pool."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BOXED = re.compile(r"\\boxed\{\d+\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-every", type=int, default=16)
    parser.add_argument("--failure-repeats", type=int, default=1)
    parser.add_argument("--anchor-every", type=int, default=1)
    return parser.parse_args()


def media_and_question(row: dict) -> tuple[str, str]:
    user = next(
        message
        for message in row["responses_create_params"]["input"]
        if message["role"] == "user"
    )
    image = next(
        item["image_url"] for item in user["content"] if item["type"] == "input_image"
    )
    question = next(
        item["text"] for item in user["content"] if item["type"] == "input_text"
    )
    return image, question


def replace_box(response: str, count: int) -> str:
    replacement = rf"\boxed{{{count}}}"
    if BOXED.search(response):
        return BOXED.sub(lambda _: replacement, response)
    return replacement


def main() -> None:
    args = parse_args()
    if args.validation_every < 2:
        raise ValueError("--validation-every must be at least 2")
    if args.failure_repeats < 1:
        raise ValueError("--failure-repeats must be at least 1")
    if args.anchor_every < 1:
        raise ValueError("--anchor-every must be at least 1")

    rows = [json.loads(line) for line in args.data.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    if len(rows) != len(predictions):
        raise ValueError("data and prediction lengths differ")

    outputs = {"train": [], "validation": []}
    failures = 0
    synthetic_negatives = 0
    for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        expected = sum(
            circle["color"] == row["target_color"] for circle in row["circles"]
        )
        predicted = prediction["boxed_predicted_count"]
        if predicted is None:
            raise ValueError(f"prediction {index} has no boxed answer")

        image, question = media_and_question(row)
        actual_response = prediction["response"]
        chosen = replace_box(actual_response, expected)
        is_failure = predicted != expected
        if is_failure:
            rejected = actual_response
            failures += 1
        else:
            # Balance adjacent alternatives so replay anchors do not create a
            # one-directional global count bias.
            offset = -1 if expected > 0 and index % 2 else 1
            rejected = replace_box(actual_response, expected + offset)
            synthetic_negatives += 1

        if chosen == rejected:
            raise ValueError(f"preference pair {index} has identical completions")
        preference = {
            "context": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "completions": [
                {
                    "rank": 0,
                    "completion": [{"role": "assistant", "content": chosen}],
                },
                {
                    "rank": 1,
                    "completion": [{"role": "assistant", "content": rejected}],
                },
            ],
        }
        is_validation = index % args.validation_every == 0
        split = "validation" if is_validation else "train"
        if is_validation:
            repeats = 1
        elif is_failure:
            repeats = args.failure_repeats
        else:
            repeats = int(index % args.anchor_every == 0)
        outputs[split].extend([preference] * repeats)

    for split, output in (
        ("train", args.train_output),
        ("validation", args.validation_output),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as destination:
            for row in outputs[split]:
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(
        json.dumps(
            {
                "train": len(outputs["train"]),
                "validation": len(outputs["validation"]),
                "model_failures": failures,
                "synthetic_anchor_negatives": synthetic_negatives,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
