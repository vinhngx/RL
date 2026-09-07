# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Build a GRPO dataset from model-scored Circle Count failures."""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.minimum_rows <= 0:
        raise ValueError("--minimum-rows must be positive")

    with args.data.open() as source:
        examples = [json.loads(line) for line in source]
    with args.predictions.open() as source:
        predictions = [json.loads(line) for line in source]
    if len(examples) != len(predictions):
        raise ValueError(
            f"row mismatch: {len(examples)} examples and "
            f"{len(predictions)} predictions"
        )

    failures = []
    for index, (example, prediction) in enumerate(
        zip(examples, predictions, strict=True)
    ):
        if prediction["index"] != index:
            raise ValueError(f"prediction index mismatch at row {index}")
        if prediction["correct"]:
            continue
        example["mining_predicted_count"] = prediction["predicted_count"]
        failures.append(example)
    if not failures:
        raise ValueError("no model failures found")

    rng = random.Random(args.seed)
    rows = []
    while len(rows) < args.minimum_rows:
        epoch = failures.copy()
        rng.shuffle(epoch)
        rows.extend(epoch)
    rows = rows[: args.minimum_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for example in rows:
            output.write(json.dumps(example, separators=(",", ":")) + "\n")
    print(
        f"mined {len(failures)}/{len(examples)} unique failures; "
        f"wrote {len(rows)} training rows"
    )


if __name__ == "__main__":
    main()
