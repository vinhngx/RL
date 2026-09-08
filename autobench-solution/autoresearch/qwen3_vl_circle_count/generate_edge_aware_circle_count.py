# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate matched edge-heavy and center-heavy Circle Count examples."""

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
    indices = sorted(
        range(len(circles)),
        key=lambda index: min(
            circles[index]["x"],
            circles[index]["y"],
            1000 - circles[index]["x"],
            1000 - circles[index]["y"],
        ),
        reverse=spatial_mode == "center",
    )
    target_indices = set(indices[:target_count])
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
    parser.add_argument("--seed-offset", type=int, default=3_800_000)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_target_count < 0:
        raise ValueError("--min-target-count must be non-negative")
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
            for mode_offset, spatial_mode in enumerate(("edge", "center")):
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
