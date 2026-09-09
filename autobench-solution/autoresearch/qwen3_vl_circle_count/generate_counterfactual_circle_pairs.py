# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate image pairs differing by exactly one target-colour circle."""

import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path
from types import ModuleType


AGENT_REF = {
    "type": "responses_api_agents",
    "name": "circle_count_simple_agent",
}


def load_generator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("circle_count_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Circle Count generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_variant(
    generator: ModuleType,
    source: dict,
    *,
    colors: list[str],
    target_color: str,
    pair_id: int,
    variant: str,
    expected_count: int,
) -> dict:
    example = copy.deepcopy(source)
    circles = example["circles"]
    for circle, color in zip(circles, colors, strict=True):
        circle["color"] = color
    content = example["responses_create_params"]["input"][1]["content"]
    content[0]["image_url"] = generator._generate_image(
        circles,
        img_size=1000,
        radius=circles[0]["radius"],
    )
    content[1]["text"] = f"How many {target_color} circles are in the image?"
    example.update(
        target_color=target_color,
        pair_id=pair_id,
        counterfactual_variant=variant,
        expected_count=expected_count,
        agent_ref=AGENT_REF,
    )
    return example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--min-target-count", type=int, default=4)
    parser.add_argument("--max-target-count", type=int, default=10)
    parser.add_argument("--min-total-circles", type=int)
    parser.add_argument("--max-total-circles", type=int, default=20)
    parser.add_argument(
        "--added-location",
        choices=("random", "edge"),
        default="random",
        help="Choose the recolored circle randomly or nearest an image edge.",
    )
    parser.add_argument("--seed-offset", type=int, default=4_600_000)
    args = parser.parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    if args.min_target_count < 0:
        raise ValueError("--min-target-count must be non-negative")
    if args.max_target_count < args.min_target_count:
        raise ValueError("--max-target-count must be >= --min-target-count")
    if args.max_total_circles <= args.max_target_count:
        raise ValueError("--max-total-circles must exceed --max-target-count")
    if (
        args.min_total_circles is not None
        and args.min_total_circles > args.max_total_circles
    ):
        raise ValueError("--min-total-circles must be <= --max-total-circles")

    generator = load_generator(args.generator)
    palette = tuple(generator.COLORS)
    rows = []
    for pair_id in range(args.pairs):
        seed = args.seed_offset + pair_id
        rng = random.Random(seed)
        base_count = rng.randint(args.min_target_count, args.max_target_count)
        min_total = max(base_count + 2, args.min_total_circles or 12)
        total = rng.randint(min_total, args.max_total_circles)
        source = generator.make_example(
            seed,
            img_size_range=(1000, 1000),
            num_circles_range=(total, total),
            num_colors_range=(2, 4),
        )
        target_color = palette[pair_id % len(palette)]
        distractors = [color for color in palette if color != target_color]
        distractor_palette = rng.sample(distractors, rng.randint(1, 3))
        if args.added_location == "edge":
            added_index = min(
                range(total),
                key=lambda index: min(
                    source["circles"][index]["x"] - source["circles"][index]["radius"],
                    source["circles"][index]["y"] - source["circles"][index]["radius"],
                    1000
                    - source["circles"][index]["x"]
                    - source["circles"][index]["radius"],
                    1000
                    - source["circles"][index]["y"]
                    - source["circles"][index]["radius"],
                ),
            )
            target_indices = set(
                rng.sample([i for i in range(total) if i != added_index], base_count)
            )
        else:
            target_indices = set(rng.sample(range(total), base_count))
            added_index = rng.choice(
                [i for i in range(total) if i not in target_indices]
            )
        base_colors = [
            target_color if i in target_indices else rng.choice(distractor_palette)
            for i in range(total)
        ]
        added_colors = base_colors.copy()
        added_colors[added_index] = target_color

        base = render_variant(
            generator,
            source,
            colors=base_colors,
            target_color=target_color,
            pair_id=pair_id,
            variant="base",
            expected_count=base_count,
        )
        plus_one = render_variant(
            generator,
            source,
            colors=added_colors,
            target_color=target_color,
            pair_id=pair_id,
            variant="plus_one",
            expected_count=base_count + 1,
        )
        center = [
            source["circles"][added_index]["x"] / 1000,
            source["circles"][added_index]["y"] / 1000,
        ]
        for row in (base, plus_one):
            row["counterfactual_center"] = center
            row["counterfactual_location"] = args.added_location
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
