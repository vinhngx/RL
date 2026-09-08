# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate paired quadrant-concentrated and quadrant-distributed tasks."""

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


def quadrant(circle: dict) -> int:
    return 2 * (circle["y"] >= 500) + (circle["x"] >= 500)


def select_concentrated(circles: list[dict], count: int, seed: int) -> set[int]:
    rng = random.Random(seed)
    preferred = rng.randrange(4)
    center_x = 250 if preferred % 2 == 0 else 750
    center_y = 250 if preferred < 2 else 750
    indices = sorted(
        range(len(circles)),
        key=lambda index: (
            quadrant(circles[index]) != preferred,
            (circles[index]["x"] - center_x) ** 2
            + (circles[index]["y"] - center_y) ** 2,
        ),
    )
    return set(indices[:count])


def select_distributed(circles: list[dict], count: int, seed: int) -> set[int]:
    rng = random.Random(seed)
    buckets = [[] for _ in range(4)]
    for index, circle in enumerate(circles):
        buckets[quadrant(circle)].append(index)
    for bucket in buckets:
        rng.shuffle(bucket)
    order = list(range(4))
    rng.shuffle(order)
    selected = []
    while len(selected) < count:
        added = False
        for bucket_index in order:
            if buckets[bucket_index]:
                selected.append(buckets[bucket_index].pop())
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise RuntimeError("Not enough circles to select target count")
    return set(selected)


def recolor_example(
    generator: ModuleType,
    source: dict,
    *,
    target_count: int,
    target_color: str,
    spatial_mode: str,
    seed: int,
) -> dict:
    example = copy.deepcopy(source)
    circles = example["circles"]
    if spatial_mode == "concentrated":
        target_indices = select_concentrated(circles, target_count, seed)
    elif spatial_mode == "distributed":
        target_indices = select_distributed(circles, target_count, seed)
    else:
        raise ValueError(f"Unknown spatial mode: {spatial_mode}")
    rng = random.Random(seed)
    distractor_colors = [color for color in generator.COLORS if color != target_color]
    distractor_palette = rng.sample(distractor_colors, rng.randint(1, 3))
    for index, circle in enumerate(circles):
        circle["color"] = (
            target_color if index in target_indices else rng.choice(distractor_palette)
        )

    user_content = example["responses_create_params"]["input"][1]["content"]
    user_content[0]["image_url"] = generator._generate_image(
        circles,
        img_size=1000,
        radius=circles[0]["radius"],
    )
    user_content[1]["text"] = f"How many {target_color} circles are in the image?"
    example["target_color"] = target_color
    example["spatial_mode"] = spatial_mode
    example["agent_ref"] = AGENT_REF
    return example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-target-count", type=int, default=5)
    parser.add_argument("--max-target-count", type=int, default=9)
    parser.add_argument("--examples-per-count", type=int, default=200)
    parser.add_argument("--seed-offset", type=int, default=4_400_000)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_target_count < 1:
        raise ValueError("--min-target-count must be positive")
    if args.max_target_count < args.min_target_count:
        raise ValueError("--max-target-count must be >= --min-target-count")
    if args.examples_per_count <= 0:
        raise ValueError("--examples-per-count must be positive")

    generator = load_generator(args.generator)
    colors = tuple(generator.COLORS)
    rows = []
    example_index = 0
    for target_count in range(args.min_target_count, args.max_target_count + 1):
        for repetition in range(args.examples_per_count):
            seed = args.seed_offset + example_index
            source = generator.make_example(
                seed,
                num_circles_range=(max(14, target_count), 20),
                num_colors_range=(2, 4),
            )
            target_color = colors[repetition % len(colors)]
            for mode_offset, spatial_mode in enumerate(
                ("concentrated", "distributed")
            ):
                rows.append(
                    recolor_example(
                        generator,
                        source,
                        target_count=target_count,
                        target_color=target_color,
                        spatial_mode=spatial_mode,
                        seed=seed * 2 + mode_offset,
                    )
                )
            example_index += 1

    random.Random(args.shuffle_seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
