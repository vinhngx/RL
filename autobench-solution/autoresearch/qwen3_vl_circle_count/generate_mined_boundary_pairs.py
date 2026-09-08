# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build balanced adjacent-count pairs around on-policy failures and anchors."""

import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("circle_count_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rerender(generator, example: dict) -> None:
    circles = example["circles"]
    content = example["responses_create_params"]["input"][1]["content"]
    content[0]["image_url"] = generator._generate_image(
        circles, img_size=1000, radius=circles[0]["radius"]
    )


def make_pair(generator, source: dict, *, remove_target: bool, pair_id: int, kind: str, rng: random.Random):
    target = source["target_color"]
    original_count = sum(circle["color"] == target for circle in source["circles"])
    if remove_target:
        candidates = [i for i, circle in enumerate(source["circles"]) if circle["color"] == target]
    else:
        candidates = [i for i, circle in enumerate(source["circles"]) if circle["color"] != target]
    if not candidates:
        remove_target = not remove_target
        candidates = [
            i
            for i, circle in enumerate(source["circles"])
            if (circle["color"] == target) == remove_target
        ]
    if not candidates:
        return None

    changed = copy.deepcopy(source)
    index = rng.choice(candidates)
    if remove_target:
        distractors = [circle["color"] for circle in source["circles"] if circle["color"] != target]
        if not distractors:
            distractors = [color for color in generator.COLORS if color != target]
        changed["circles"][index]["color"] = rng.choice(distractors)
        low, high = changed, copy.deepcopy(source)
        low_count, high_count = original_count - 1, original_count
    else:
        changed["circles"][index]["color"] = target
        low, high = copy.deepcopy(source), changed
        low_count, high_count = original_count, original_count + 1
    rerender(generator, changed)
    for variant, row, expected in (("low", low, low_count), ("high", high, high_count)):
        row.update(
            pair_id=pair_id,
            boundary_kind=kind,
            counterfactual_variant=variant,
            expected_count=expected,
        )
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=401)
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.data.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    if len(examples) != len(predictions):
        raise ValueError("example/prediction row mismatch")
    failures, anchors = [], []
    for index, (example, prediction) in enumerate(zip(examples, predictions, strict=True)):
        if prediction["index"] != index:
            raise ValueError(f"prediction index mismatch at {index}")
        expected = sum(c["color"] == example["target_color"] for c in example["circles"])
        predicted = prediction.get("predicted_count")
        if predicted is None:
            continue
        if prediction["correct"]:
            anchors.append((example, expected, predicted))
        else:
            failures.append((example, expected, predicted))
    if not failures or not anchors:
        raise ValueError("both failures and correct anchors are required")

    generator = load_generator(args.generator)
    rng = random.Random(args.seed)
    rng.shuffle(failures)
    rng.shuffle(anchors)
    rows = []
    for i in range(args.failure_pairs):
        failure, expected, predicted = failures[i % len(failures)]
        # Put the original on the side toward which the model erred. An
        # undercount becomes (N-1, N); an overcount becomes (N, N+1).
        failure_pair = make_pair(
            generator,
            failure,
            remove_target=predicted < expected,
            pair_id=2 * i,
            kind="failure",
            rng=rng,
        )
        anchor, _, _ = anchors[i % len(anchors)]
        anchor_pair = make_pair(
            generator,
            anchor,
            remove_target=bool(i % 2),
            pair_id=2 * i + 1,
            kind="anchor",
            rng=rng,
        )
        if failure_pair is None or anchor_pair is None:
            raise ValueError("source row cannot support requested counterfactual")
        rows.extend((*failure_pair, *anchor_pair))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"failures={len(failures)} anchors={len(anchors)} pairs={len(rows)//2}")


if __name__ == "__main__":
    main()
