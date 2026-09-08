#!/usr/bin/env python3
"""Build deployment-matched SFT data from on-policy failures and matched anchors."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def expected_count(row: dict) -> int:
    return sum(circle["color"] == row["target_color"] for circle in row["circles"])


def to_sft(row: dict) -> dict:
    inputs = row["responses_create_params"]["input"]
    system = next(item["content"] for item in inputs if item["role"] == "system")
    user = next(item["content"] for item in inputs if item["role"] == "user")
    image = next(item["image_url"] for item in user if item["type"] == "input_image")
    question = next(item["text"] for item in user if item["type"] == "input_text")
    count = expected_count(row)
    return {
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": rf"\boxed{{{count}}}"},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.data.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    if len(examples) != len(predictions):
        raise ValueError("example/prediction row mismatch")
    failures, anchors = [], []
    for index, (row, prediction) in enumerate(zip(examples, predictions, strict=True)):
        if prediction["index"] != index:
            raise ValueError(f"prediction index mismatch at {index}")
        (anchors if prediction["correct"] else failures).append(row)
    if not failures or not anchors:
        raise ValueError("both failures and correct anchors are required")

    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    count_buckets: dict[int, list[dict]] = defaultdict(list)
    for row in anchors:
        count = expected_count(row)
        buckets[(count, row["target_color"])].append(row)
        count_buckets[count].append(row)

    rng = random.Random(args.seed)
    rng.shuffle(failures)
    for values in buckets.values():
        rng.shuffle(values)
    for values in count_buckets.values():
        rng.shuffle(values)
    rows = []
    for index in range(args.pairs):
        failure = failures[index % len(failures)]
        count = expected_count(failure)
        candidates = buckets.get((count, failure["target_color"])) or count_buckets.get(count) or anchors
        anchor = candidates[(index // len(failures)) % len(candidates)]
        rows.extend((to_sft(failure), to_sft(anchor)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"failures={len(failures)} anchors={len(anchors)} pairs={args.pairs} rows={len(rows)}")


if __name__ == "__main__":
    main()
