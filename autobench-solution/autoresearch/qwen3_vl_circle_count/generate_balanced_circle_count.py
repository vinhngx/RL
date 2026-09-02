# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build deterministic Circle Count examples balanced by exact target count."""

import argparse
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


def make_balanced_example(
    generator: ModuleType,
    *,
    seed: int,
    target_count: int,
    target_color: str,
) -> dict:
    rng = random.Random(seed)
    example = generator.make_example(
        seed,
        num_circles_range=(max(5, target_count), 20),
        num_colors_range=(2, 4),
    )

    circles = example["circles"]
    indices = list(range(len(circles)))
    rng.shuffle(indices)
    target_indices = set(indices[:target_count])
    distractor_colors = [color for color in generator.COLORS if color != target_color]
    distractor_palette = rng.sample(distractor_colors, rng.randint(1, 3))
    for index, circle in enumerate(circles):
        circle["color"] = (
            target_color if index in target_indices else rng.choice(distractor_palette)
        )

    image_url = generator._generate_image(
        circles,
        img_size=1000,
        radius=circles[0]["radius"],
    )
    user_content = example["responses_create_params"]["input"][1]["content"]
    user_content[0]["image_url"] = image_url
    user_content[1]["text"] = f"How many {target_color} circles are in the image?"
    example["target_color"] = target_color
    example["agent_ref"] = AGENT_REF
    return example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-count", type=int, default=32)
    parser.add_argument("--max-target-count", type=int, default=14)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    args = parser.parse_args()

    generator = load_generator(args.generator)
    colors = tuple(generator.COLORS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for target_count in range(args.max_target_count + 1):
            for repetition in range(args.examples_per_count):
                seed = (
                    args.seed_offset
                    + target_count * args.examples_per_count
                    + repetition
                )
                example = make_balanced_example(
                    generator,
                    seed=seed,
                    target_count=target_count,
                    target_color=colors[repetition % len(colors)],
                )
                output.write(json.dumps(example, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
