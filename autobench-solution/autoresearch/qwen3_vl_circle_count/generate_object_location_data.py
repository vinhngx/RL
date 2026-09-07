#!/usr/bin/env python3
"""Build object-location SFT rows or GRPO rows from Circle Count data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Count circles of the color requested by the user. Use a 10 by 10 grid over "
    "the image, with x=0..9 from left to right and y=0..9 from top to bottom. "
    "List the approximate grid cell of every target-circle center exactly as "
    r"P=(x,y),(x,y),... and finish with the number listed in \boxed{N} format."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("sft", "grpo"), required=True)
    parser.add_argument(
        "--coordinate-space",
        choices=("grid", "pixel"),
        default="grid",
        help="Coordinate representation stored in GRPO process ground truth.",
    )
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=1000)
    return parser.parse_args()


def target_points(
    row: dict[str, Any], *, grid_size: int, image_size: int, coordinate_space: str
) -> list[tuple[int, int]]:
    points = []
    for circle in row["circles"]:
        if circle["color"] != row["target_color"]:
            continue
        if coordinate_space == "pixel":
            x, y = int(circle["x"]), int(circle["y"])
        else:
            x = min(grid_size - 1, circle["x"] * grid_size // image_size)
            y = min(grid_size - 1, circle["y"] * grid_size // image_size)
        points.append((x, y))
    return sorted(points, key=lambda point: (point[1], point[0]))


def sft_row(row: dict[str, Any], points: list[tuple[int, int]]) -> dict[str, Any]:
    source = row["responses_create_params"]["input"][1]["content"]
    user_content = []
    for item in source:
        if item["type"] == "input_image":
            user_content.append({"type": "image", "image": item["image_url"]})
        elif item["type"] == "input_text":
            user_content.append({"type": "text", "text": item["text"]})
    point_text = ",".join(f"({x},{y})" for x, y in points)
    response = rf"P={point_text}; \boxed{{{len(points)}}}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def main() -> None:
    args = parse_args()
    if args.grid_size < 2 or args.grid_size > 10:
        raise ValueError("--grid-size must be between 2 and 10")
    rows = [json.loads(line) for line in args.input.open()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for row in rows:
            points = target_points(
                row,
                grid_size=args.grid_size,
                image_size=args.image_size,
                coordinate_space=args.coordinate_space,
            )
            if args.mode == "sft":
                output = sft_row(row, points)
            else:
                output = copy.deepcopy(row)
                output["process_ground_truth"] = json.dumps(
                    {
                        "coordinate_space": args.coordinate_space,
                        "points": points,
                        "total": len(points),
                    },
                    separators=(",", ":"),
                )
            destination.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"input": len(rows), "output": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
