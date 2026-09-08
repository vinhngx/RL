#!/usr/bin/env python3
"""Attach dense all-color count supervision to Circle Count GRPO rows."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open() as source, args.output.open("w") as destination:
        for line in source:
            row = json.loads(line)
            counts = Counter(circle["color"] for circle in row["circles"])
            dense_counts = {color: counts[color] for color in COLORS}
            output = copy.deepcopy(row)
            output["process_ground_truth"] = json.dumps(
                {
                    "color_counts": dense_counts,
                    "target_color": row["target_color"],
                    "total": dense_counts[row["target_color"]],
                },
                separators=(",", ":"),
            )
            destination.write(json.dumps(output, separators=(",", ":")) + "\n")
            written += 1
    print(json.dumps({"input": written, "output": written}, sort_keys=True))


if __name__ == "__main__":
    main()
