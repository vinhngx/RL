#!/usr/bin/env python3
"""Generate deployment-matched SFT rows directly from the Circle Count Gym."""

import argparse
import importlib.util
import json
from pathlib import Path


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("circle_count_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_sft(row: dict) -> dict:
    inputs = row["responses_create_params"]["input"]
    system = next(item["content"] for item in inputs if item["role"] == "system")
    user = next(item["content"] for item in inputs if item["role"] == "user")
    image = next(item["image_url"] for item in user if item["type"] == "input_image")
    question = next(item["text"] for item in user if item["type"] == "input_text")
    count = sum(circle["color"] == row["target_color"] for circle in row["circles"])
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
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    args = parser.parse_args()
    if args.n < 1:
        raise ValueError("--n must be positive")

    generator = load_generator(args.generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for offset in range(args.n):
            row = generator.make_example(args.seed_offset + offset)
            output.write(json.dumps(to_sft(row), separators=(",", ":")) + "\n")
            if (offset + 1) % 1000 == 0:
                print(f"generated={offset + 1}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
