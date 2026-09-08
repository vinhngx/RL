#!/usr/bin/env python3
"""Convert a contiguous, pair-aligned slice of Gym rows to terse SFT."""

import argparse
import json
from pathlib import Path


def to_sft(row: dict) -> dict:
    inputs = row["responses_create_params"]["input"]
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
            {"role": "assistant", "content": rf"\boxed{{{row['expected_count']}}}"},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--pairs", type=int, required=True)
    args = parser.parse_args()
    if args.start_pair < 0 or args.pairs < 1:
        raise ValueError("pair slice must be non-negative and non-empty")

    rows = [json.loads(line) for line in args.input.open()]
    selected = rows[2 * args.start_pair : 2 * (args.start_pair + args.pairs)]
    if len(selected) != 2 * args.pairs:
        raise ValueError("input does not contain the requested pair slice")
    for offset in range(0, len(selected), 2):
        first, second = selected[offset : offset + 2]
        if first["pair_id"] != second["pair_id"]:
            raise ValueError(f"unaligned pair at selected row {offset}")
        if {first["counterfactual_variant"], second["counterfactual_variant"]} != {
            "base",
            "plus_one",
        }:
            raise ValueError(f"invalid variants for pair {first['pair_id']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in selected:
            output.write(json.dumps(to_sft(row), separators=(",", ":")) + "\n")
    print(
        f"pairs={args.pairs} rows={len(selected)} "
        f"pair_ids={selected[0]['pair_id']}..{selected[-1]['pair_id']}"
    )


if __name__ == "__main__":
    main()
