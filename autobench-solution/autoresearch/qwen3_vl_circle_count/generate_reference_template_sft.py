#!/usr/bin/env python3
"""Convert Gym circle-count rows to the response templates learned by the reference model."""

import argparse
import json
import random
from pathlib import Path


RESPONSE_TEMPLATES = (
    "Counting the {color} circles: I'll scan the image systematically. "
    "After checking the whole image, I count exactly \\boxed{{{count}}} {color} circles.",
    "Let me look at the image carefully and count the {color} circles. "
    "Scanning from the top left to the bottom right, I find them one by one. "
    "The total count is \\boxed{{{count}}}.",
    "Looking at this image, I count the {color} circles. Going through them carefully, "
    "the count of {color} circles is \\boxed{{{count}}}.",
    "I need to count the {color} circles in this image. Carefully going through each "
    "region of the image, I can identify the {color} circles. There are "
    "\\boxed{{{count}}} {color} circles.",
    "To count the {color} circles, I scan row by row. In the image I can spot each "
    "{color} circle clearly. Final answer: \\boxed{{{count}}}.",
)


def convert(raw_path: Path, output_path: Path, seed_offset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open() as source, output_path.open("w") as destination:
        for index, line in enumerate(source):
            row = json.loads(line)
            inputs = row["responses_create_params"]["input"]
            system_prompt = next(item["content"] for item in inputs if item["role"] == "system")
            user_content = next(item["content"] for item in inputs if item["role"] == "user")
            image_url = next(item["image_url"] for item in user_content if item["type"] == "input_image")
            question = next(item["text"] for item in user_content if item["type"] == "input_text")
            color = row["target_color"]
            count = sum(circle["color"] == color for circle in row["circles"])
            template = random.Random(seed_offset + index).choice(RESPONSE_TEMPLATES)
            converted = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_url},
                            {"type": "text", "text": question},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": template.format(color=color, count=count),
                    },
                ]
            }
            destination.write(json.dumps(converted, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-offset", type=int, required=True)
    args = parser.parse_args()
    convert(args.raw, args.output, args.seed_offset)


if __name__ == "__main__":
    main()
