#!/usr/bin/env python3
"""Add dense ordinal count supervision to raw Circle Count rows."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


THRESHOLDS = tuple(range(5, 11))


def make_ordinal_example(row: dict) -> dict:
    """Return a copy whose prompt and ground truth expose count thresholds."""
    output = copy.deepcopy(row)
    target_color = output["target_color"]
    total = sum(circle["color"] == target_color for circle in output["circles"])
    expected = {
        "kind": "ordinal_count",
        **{f"ge{threshold}": int(total >= threshold) for threshold in THRESHOLDS},
        "total": total,
    }
    for message in output["responses_create_params"]["input"]:
        if message["role"] != "user":
            continue
        for content in message["content"]:
            if content["type"] != "input_text":
                continue
            content["text"] = (
                f"How many {target_color} circles are in the image? Report "
                "six binary threshold checks in exactly this order: "
                "GE5=, GE6=, GE7=, GE8=, GE9=, GE10=. Write 1 when the "
                "count is at least that threshold and 0 otherwise. Then "
                r"finish with the exact total in \boxed{} format. For "
                "example, if the total is 7, write exactly: GE5=1 GE6=1 "
                r"GE7=1 GE8=0 GE9=0 GE10=0 \boxed{7}. Never leave a "
                "threshold value blank."
            )
    output["process_ground_truth"] = json.dumps(expected, separators=(",", ":"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count_histogram: dict[int, int] = {}
    written = 0
    with args.input.open() as source, args.output.open("w") as destination:
        for line in source:
            output = make_ordinal_example(json.loads(line))
            expected = json.loads(output["process_ground_truth"])
            total = int(expected["total"])
            count_histogram[total] = count_histogram.get(total, 0) + 1
            destination.write(json.dumps(output, separators=(",", ":")) + "\n")
            written += 1
    print(
        json.dumps(
            {"count_histogram": count_histogram, "written": written},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
