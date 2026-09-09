#!/usr/bin/env python3
"""Convert adjacent Circle Count image pairs into grouped VLM preferences.

Each source pair contains a base image with k target circles and the same image
with one target circle added.  Two adjacent preference rows are emitted so a
DPO microbatch of two becomes, after NeMo's interleaving collator:

    base chosen, base rejected, plus-one chosen, plus-one rejected

This ordering is the contract used by the counterfactual vision auxiliary loss.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument(
        "--validation-pairs-per-boundary",
        type=int,
        default=4,
        help="Number of whole adjacent-image pairs held out for each k->k+1 boundary.",
    )
    return parser.parse_args()


def expected_count(row: dict[str, Any]) -> int:
    declared = int(row["expected_count"])
    derived = sum(
        circle["color"] == row["target_color"] for circle in row["circles"]
    )
    if declared != derived:
        raise ValueError(f"declared count {declared} differs from derived {derived}")
    return declared


def media_and_question(row: dict[str, Any]) -> tuple[str, str]:
    user = next(
        message
        for message in row["responses_create_params"]["input"]
        if message["role"] == "user"
    )
    image = next(
        part["image_url"]
        for part in user["content"]
        if part["type"] == "input_image"
    )
    question = next(
        part["text"] for part in user["content"] if part["type"] == "input_text"
    )
    return image, question


def preference(
    row: dict[str, Any],
    chosen_count: int,
    rejected_count: int,
    system_prompt: str,
    counterfactual_center: tuple[float, float],
) -> dict[str, Any]:
    image, question = media_and_question(row)
    return {
        "counterfactual_center": list(counterfactual_center),
        "context": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ],
        "completions": [
            {
                "rank": 0,
                "completion": [
                    {
                        "role": "assistant",
                        "content": rf"\boxed{{{chosen_count}}}",
                    }
                ],
            },
            {
                "rank": 1,
                "completion": [
                    {
                        "role": "assistant",
                        "content": rf"\boxed{{{rejected_count}}}",
                    }
                ],
            },
        ],
    }


def changed_circle_center(
    base: dict[str, Any], upper: dict[str, Any]
) -> tuple[float, float]:
    """Return the normalized center of the sole non-target -> target change."""
    if len(base["circles"]) != len(upper["circles"]):
        raise ValueError("counterfactual images have different circle counts")
    changed = []
    for lower_circle, upper_circle in zip(
        base["circles"], upper["circles"], strict=True
    ):
        lower_geometry = (
            lower_circle["x"],
            lower_circle["y"],
            lower_circle["radius"],
        )
        upper_geometry = (
            upper_circle["x"],
            upper_circle["y"],
            upper_circle["radius"],
        )
        if lower_geometry != upper_geometry:
            raise ValueError("counterfactual circle geometry changed")
        if lower_circle["color"] != upper_circle["color"]:
            changed.append((lower_circle, upper_circle))
    if len(changed) != 1:
        raise ValueError(f"expected one changed circle, found {len(changed)}")
    lower_circle, upper_circle = changed[0]
    if upper_circle["color"] != upper["target_color"]:
        raise ValueError("changed circle did not become the target color")
    image_size = 1000.0
    return lower_circle["x"] / image_size, lower_circle["y"] / image_size


def interleave_boundaries(
    grouped: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    queues = {boundary: deque(pairs) for boundary, pairs in sorted(grouped.items())}
    ordered = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                ordered.append(queue.popleft())
    return ordered


def main() -> None:
    args = parse_args()
    if args.validation_pairs_per_boundary < 1:
        raise ValueError("--validation-pairs-per-boundary must be positive")
    system_prompt = args.system_prompt_file.read_text().strip()
    if not system_prompt:
        raise ValueError("system prompt is empty")

    rows = [json.loads(line) for line in args.input.open()]
    if len(rows) % 2:
        raise ValueError("input must contain an even number of rows")

    grouped: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for offset in range(0, len(rows), 2):
        base, upper = rows[offset : offset + 2]
        if base["pair_id"] != upper["pair_id"]:
            raise ValueError(f"pair mismatch at rows {offset} and {offset + 1}")
        if (base["counterfactual_variant"], upper["counterfactual_variant"]) != (
            "base",
            "plus_one",
        ):
            raise ValueError(f"variant order mismatch for pair {base['pair_id']}")
        lower_count = expected_count(base)
        if expected_count(upper) != lower_count + 1:
            raise ValueError(f"counts are not adjacent for pair {base['pair_id']}")
        grouped[lower_count].append((base, upper))

    train_groups: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    validation_groups: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for boundary, pairs in sorted(grouped.items()):
        holdout = args.validation_pairs_per_boundary
        if len(pairs) <= holdout:
            raise ValueError(f"not enough pairs for boundary {boundary}->{boundary + 1}")
        validation_groups[boundary] = pairs[:holdout]
        train_groups[boundary] = pairs[holdout:]

    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, split_groups in (
        ("train", train_groups),
        ("validation", validation_groups),
    ):
        for base, upper in interleave_boundaries(split_groups):
            lower_count = expected_count(base)
            upper_count = lower_count + 1
            center = changed_circle_center(base, upper)
            outputs[split].append(
                preference(
                    base,
                    lower_count,
                    upper_count,
                    system_prompt,
                    center,
                )
            )
            outputs[split].append(
                preference(
                    upper,
                    upper_count,
                    lower_count,
                    system_prompt,
                    center,
                )
            )

    for split, output_path in (
        ("train", args.train_output),
        ("validation", args.validation_output),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as destination:
            for row in outputs[split]:
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(
        json.dumps(
            {
                "boundaries": {
                    f"{count}->{count + 1}": len(pairs)
                    for count, pairs in sorted(grouped.items())
                },
                "source_pairs": len(rows) // 2,
                "train_preferences": len(outputs["train"]),
                "validation_preferences": len(outputs["validation"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
