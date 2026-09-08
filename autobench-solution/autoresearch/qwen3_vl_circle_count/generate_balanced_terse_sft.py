#!/usr/bin/env python3
"""Generate count-balanced circle-count SFT rows with deployment-matched answers."""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("circle_count_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_sft(example: dict, count: int) -> dict:
    inputs = example["responses_create_params"]["input"]
    system = next(item["content"] for item in inputs if item["role"] == "system")
    user = next(item["content"] for item in inputs if item["role"] == "user")
    image = next(item["image_url"] for item in user if item["type"] == "input_image")
    question = next(item["text"] for item in user if item["type"] == "input_text")
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
    parser.add_argument("--per-count", type=int, required=True)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, required=True)
    args = parser.parse_args()
    if args.min_count > args.max_count or args.per_count < 1:
        raise ValueError("invalid count range or quota")

    generator = load_generator(args.generator)
    wanted = set(range(args.min_count, args.max_count + 1))
    counts: Counter[int] = Counter()
    rows = []
    seed = args.seed_offset
    while any(counts[count] < args.per_count for count in wanted):
        example = generator.make_example(seed)
        seed += 1
        target = example["target_color"]
        count = sum(circle["color"] == target for circle in example["circles"])
        if count not in wanted or counts[count] >= args.per_count:
            continue
        rows.append(to_sft(example, count))
        counts[count] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"rows={len(rows)} counts={dict(sorted(counts.items()))} seeds_scanned={seed-args.seed_offset}")


if __name__ == "__main__":
    main()
