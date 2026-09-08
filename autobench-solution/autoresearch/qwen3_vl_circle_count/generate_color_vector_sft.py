#!/usr/bin/env python3
"""Convert raw Circle Count rows into exact all-color-vector SFT examples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")
SYSTEM = (
    "Count the circles of every color in the image. Report exactly one count for "
    "each color using V=(red:N, orange:N, yellow:N, green:N, cyan:N, blue:N, "
    r"purple:N, pink:N). Then answer the user's requested color in \boxed{N} format."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open() as source, args.output.open("w") as destination:
        for line in source:
            row = json.loads(line)
            observed = Counter(circle["color"] for circle in row["circles"])
            counts = {color: observed[color] for color in COLORS}
            vector = ", ".join(f"{color}:{counts[color]}" for color in COLORS)
            target_count = counts[row["target_color"]]
            output = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    user_message(row),
                    {
                        "role": "assistant",
                        "content": rf"V=({vector}). \boxed{{{target_count}}}",
                    },
                ]
            }
            destination.write(json.dumps(output, separators=(",", ":")) + "\n")
            written += 1
    print(json.dumps({"input": written, "output": written}, sort_keys=True))


if __name__ == "__main__":
    main()
