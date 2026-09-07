#!/usr/bin/env python3
"""Convert raw Circle Count rows into terse and quadrant-process SFT examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TERSE_SYSTEM = (
    "You are a visual assistant. Count the number of circles of the specified "
    r"color in the image. Output your final answer in \boxed{} format, e.g. \boxed{3}."
)
PROCESS_SYSTEM = (
    "Count circles of the color requested by the user. Mentally divide the image "
    "at its horizontal and vertical center. Count target-color circle centers in "
    "the top-left, top-right, bottom-left, and bottom-right regions. Respond exactly "
    r"as TL=a, TR=b, BL=c, BR=d; \boxed{N}, where N is their sum."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1000)
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Emit only structured examples (useful for validation).",
    )
    return parser.parse_args()


def user_message(row: dict[str, Any]) -> dict[str, Any]:
    source = row["responses_create_params"]["input"][1]["content"]
    content = []
    for item in source:
        if item["type"] == "input_image":
            content.append({"type": "image", "image": item["image_url"]})
        elif item["type"] == "input_text":
            content.append({"type": "text", "text": item["text"]})
    return {"role": "user", "content": content}


def quadrant_counts(row: dict[str, Any], midpoint: float) -> dict[str, int]:
    counts = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for circle in row["circles"]:
        if circle["color"] != row["target_color"]:
            continue
        vertical = "T" if circle["y"] < midpoint else "B"
        horizontal = "L" if circle["x"] < midpoint else "R"
        counts[vertical + horizontal] += 1
    return counts


def example(system: str, user: dict[str, Any], response: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            user,
            {"role": "assistant", "content": response},
        ]
    }


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.open()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with args.output.open("w") as destination:
        for row in rows:
            user = user_message(row)
            counts = quadrant_counts(row, args.image_size / 2)
            total = sum(counts.values())
            process_response = (
                f"TL={counts['TL']}, TR={counts['TR']}, "
                f"BL={counts['BL']}, BR={counts['BR']}; "
                rf"\boxed{{{total}}}"
            )
            examples = [example(PROCESS_SYSTEM, user, process_response)]
            if not args.process_only:
                terse_response = (
                    f"There are {total} {row['target_color']} circles in the image. "
                    rf"\boxed{{{total}}}"
                )
                examples.insert(0, example(TERSE_SYSTEM, user, terse_response))
            for output in examples:
                destination.write(json.dumps(output, separators=(",", ":")) + "\n")
                emitted += 1
    print(json.dumps({"input": len(rows), "output": emitted}, sort_keys=True))


if __name__ == "__main__":
    main()
