#!/usr/bin/env python3
"""Build direction-neutral, image-conditioned Circle Count DPO pairs.

Every real model failure is coupled to one unique, initially-correct anchor.  The
anchor's synthetic rejected answer moves in the opposite numeric direction from
the failure.  Keeping both examples in the same split prevents a global
"increase/decrease the count" shortcut while preserving the real error signal.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


BOXED = re.compile(r"\\boxed\{\d+\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-every", type=int, default=16)
    parser.add_argument("--system-prompt-file", type=Path)
    return parser.parse_args()


def media_and_question(row: dict[str, Any]) -> tuple[str, str]:
    user = next(
        message
        for message in row["responses_create_params"]["input"]
        if message["role"] == "user"
    )
    image = next(
        item["image_url"] for item in user["content"] if item["type"] == "input_image"
    )
    question = next(
        item["text"] for item in user["content"] if item["type"] == "input_text"
    )
    return image, question


def replace_box(response: str, count: int) -> str:
    replacement = rf"\boxed{{{count}}}"
    if BOXED.search(response):
        return BOXED.sub(lambda _: replacement, response)
    return replacement


def expected_count(row: dict[str, Any]) -> int:
    return sum(circle["color"] == row["target_color"] for circle in row["circles"])


def preference(
    row: dict[str, Any], chosen: str, rejected: str, system_prompt: str | None
) -> dict[str, Any]:
    if chosen == rejected:
        raise ValueError("chosen and rejected completions are identical")
    image, question = media_and_question(row)
    context: list[dict[str, Any]] = []
    if system_prompt is not None:
        context.append({"role": "system", "content": system_prompt})
    context.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
    )
    return {
        "context": context,
        "completions": [
            {"rank": 0, "completion": [{"role": "assistant", "content": chosen}]},
            {
                "rank": 1,
                "completion": [{"role": "assistant", "content": rejected}],
            },
        ],
    }


def take_unused(
    candidates: deque[int] | None, used: set[int]
) -> int | None:
    if candidates is None:
        return None
    while candidates and candidates[0] in used:
        candidates.popleft()
    return candidates.popleft() if candidates else None


def main() -> None:
    args = parse_args()
    if args.validation_every < 2:
        raise ValueError("--validation-every must be at least 2")
    system_prompt = None
    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text().strip()
        if not system_prompt:
            raise ValueError("--system-prompt-file is empty")

    rows = [json.loads(line) for line in args.data.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    if len(rows) != len(predictions):
        raise ValueError("data and prediction lengths differ")

    failures: list[int] = []
    correct: list[int] = []
    expected: list[int] = []
    for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        count = expected_count(row)
        expected.append(count)
        predicted = prediction.get("boxed_predicted_count")
        if predicted is None:
            raise ValueError(f"prediction {index} has no boxed answer")
        (correct if predicted == count else failures).append(index)

    # Anchors for an undercount reject count+1; anchors for an overcount reject
    # count-1.  Bucket them from most-specific to fallback matching levels.
    exact: dict[tuple[str, int, str], deque[int]] = defaultdict(deque)
    by_count: dict[tuple[int, str], deque[int]] = defaultdict(deque)
    by_direction: dict[str, deque[int]] = defaultdict(deque)
    for index in correct:
        count = expected[index]
        color = rows[index]["target_color"]
        for direction in ("up", "down"):
            if direction == "down" and count == 0:
                continue
            exact[(color, count, direction)].append(index)
            by_count[(count, direction)].append(index)
            by_direction[direction].append(index)

    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    used_anchors: set[int] = set()
    match_counts = {"exact": 0, "count": 0, "direction": 0}
    failure_directions = {"under": 0, "over": 0}

    for unit_index, failure_index in enumerate(failures):
        row = rows[failure_index]
        prediction = predictions[failure_index]
        count = expected[failure_index]
        predicted = int(prediction["boxed_predicted_count"])
        failure_direction = "under" if predicted < count else "over"
        failure_directions[failure_direction] += 1

        # Counter the real pair's preferred numeric move: an undercount failure
        # prefers moving up, so its anchor prefers the correct lower answer.
        anchor_reject_direction = "up" if failure_direction == "under" else "down"
        keys = (
            ("exact", exact.get((row["target_color"], count, anchor_reject_direction))),
            ("count", by_count.get((count, anchor_reject_direction))),
            ("direction", by_direction.get(anchor_reject_direction)),
        )
        anchor_index = None
        for match_kind, candidates in keys:
            anchor_index = take_unused(candidates, used_anchors)
            if anchor_index is not None:
                match_counts[match_kind] += 1
                break
        if anchor_index is None:
            raise RuntimeError(f"no unused counter-anchor for failure {failure_index}")
        used_anchors.add(anchor_index)

        actual_response = prediction["response"]
        failure_pair = preference(
            row,
            replace_box(actual_response, count),
            actual_response,
            system_prompt,
        )

        anchor_row = rows[anchor_index]
        anchor_response = predictions[anchor_index]["response"]
        anchor_count = expected[anchor_index]
        rejected_count = (
            anchor_count + 1
            if anchor_reject_direction == "up"
            else anchor_count - 1
        )
        anchor_pair = preference(
            anchor_row,
            anchor_response,
            replace_box(anchor_response, rejected_count),
            system_prompt,
        )

        split = (
            "validation"
            if unit_index % args.validation_every == 0
            else "train"
        )
        outputs[split].extend((failure_pair, anchor_pair))

    for split, output in (
        ("train", args.train_output),
        ("validation", args.validation_output),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as destination:
            for row in outputs[split]:
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(
        json.dumps(
            {
                "correct_pool": len(correct),
                "failure_directions": failure_directions,
                "failures": len(failures),
                "match_quality": match_counts,
                "train": len(outputs["train"]),
                "unique_anchors": len(used_anchors),
                "validation": len(outputs["validation"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
