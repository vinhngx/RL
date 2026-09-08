#!/usr/bin/env python3
"""Measure dense color-vector behavior from an HF evaluation JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")
BOX = re.compile(r"\\boxed\{(\d+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = [json.loads(line) for line in args.data.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    component_correct = 0
    vector_parseable = 0
    vector_exact = 0
    box_correct = 0
    reward_sum = 0.0
    for prediction in predictions:
        row = source[prediction["index"]]
        observed = Counter(circle["color"] for circle in row["circles"])
        response = prediction["response"]
        parsed: dict[str, int] = {}
        for color in COLORS:
            match = re.search(
                rf"\b{color}\s*[:=]\s*(\d+)\b", response, re.IGNORECASE
            )
            if match:
                parsed[color] = int(match.group(1))
        exact = sum(parsed.get(color) == observed[color] for color in COLORS)
        target = observed[row["target_color"]]
        box_match = BOX.search(response)
        target_correct = bool(box_match and int(box_match.group(1)) == target)
        component_correct += exact
        vector_parseable += len(parsed) == len(COLORS)
        vector_exact += exact == len(COLORS)
        box_correct += target_correct
        reward_sum += 0.75 * exact / len(COLORS) + 0.25 * target_correct
    total = len(predictions)
    summary = {
        "total": total,
        "vector_parseable": vector_parseable,
        "vector_parse_rate": vector_parseable / total,
        "component_correct": component_correct,
        "component_total": total * len(COLORS),
        "component_accuracy": component_correct / (total * len(COLORS)),
        "vector_exact": vector_exact,
        "vector_exact_accuracy": vector_exact / total,
        "boxed_correct": box_correct,
        "boxed_accuracy": box_correct / total,
        "mean_reward": reward_sum / total,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
