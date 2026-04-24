#!/usr/bin/env python3
"""Build a clean + hard-purple Circle Click training split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path


def _load_circle_generator(repo_root: Path):
    module_path = repo_root / "3rdparty/Gym-workspace/Gym/resources_servers/circle_click/generate_data.py"
    spec = importlib.util.spec_from_file_location("circle_click_generate_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_tooling_fields(example: dict, generator) -> dict:
    example["responses_create_params"]["tool_choice"] = {"type": "function", "name": "click"}
    example["agent_ref"] = {"type": "responses_api_agents", "name": "circle_click_simple_agent"}
    return example


def _non_overlapping_position(
    rng: random.Random,
    generator,
    radius: int,
    existing: list[dict],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    image_size: int = 1000,
) -> dict:
    margin = radius + 10
    lo_x = max(margin, x_range[0])
    hi_x = min(image_size - margin, x_range[1])
    lo_y = max(margin, y_range[0])
    hi_y = min(image_size - margin, y_range[1])
    for _ in range(1000):
        x = rng.randint(lo_x, hi_x)
        y = rng.randint(lo_y, hi_y)
        if all(((x - c["x"]) ** 2 + (y - c["y"]) ** 2) ** 0.5 > 2 * radius + 15 for c in existing):
            return {"x": x, "y": y}
    # Fall back to the normal placer if a tight region is full.
    for p in generator._place_circles(1, image_size, radius, rng):
        if all(((p["x"] - c["x"]) ** 2 + (p["y"] - c["y"]) ** 2) ** 0.5 > 2 * radius + 15 for c in existing):
            return p
    raise RuntimeError("Could not place hard-purple circle")


def _make_hard_purple(seed: int, generator) -> dict:
    rng = random.Random(seed)
    image_size = 1000
    radius = rng.randint(70, 120)
    hard_regions = [
        ((radius + 10, 260), (680, 900)),   # bottom-left, close to persistent miss 126/322
        ((300, 470), (720, 900)),           # lower-middle, close to miss 251
        ((480, 700), (360, 560)),           # center-right, close to miss 246
        ((650, 850), (180, 360)),           # upper-right, close to step70's extra miss
    ]
    target_x, target_y = rng.choice(hard_regions)

    circles = []
    target_pos = _non_overlapping_position(rng, generator, radius, circles, target_x, target_y, image_size)
    circles.append({"x": target_pos["x"], "y": target_pos["y"], "radius": radius, "color": "purple"})

    distractor_colors = rng.sample(["blue", "red", "pink", "cyan", "orange"], rng.randint(2, 3))
    for color in distractor_colors:
        pos = _non_overlapping_position(rng, generator, radius, circles, (80, 920), (80, 920), image_size)
        circles.append({"x": pos["x"], "y": pos["y"], "radius": radius, "color": color})

    rng.shuffle(circles)
    image_url = generator._generate_image(circles, image_size, radius)
    others = [c["color"] for c in circles if c["color"] != "purple"]
    others_str = ", ".join(f"a {col} circle" for col in others)
    user_text = (
        f"This is a {image_size}x{image_size} pixel image. "
        f"It shows {others_str}, and a purple circle. "
        "Click on the purple circle."
    )

    example = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": generator.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url, "detail": "auto"},
                        {"type": "input_text", "text": user_text},
                    ],
                },
            ],
            "tools": [generator.CLICK_TOOL],
        },
        "circles": circles,
        "target_color": "purple",
    }
    return _copy_tooling_fields(example, generator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-train", type=Path, required=True)
    parser.add_argument("--clean-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-count", type=int, default=3000)
    parser.add_argument("--hard-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=240426)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    generator = _load_circle_generator(repo_root)
    rng = random.Random(args.seed)

    clean_rows = [json.loads(line) for line in args.clean_train.read_text().splitlines()]
    if args.clean_count > len(clean_rows):
        raise ValueError(f"Requested {args.clean_count} clean rows from {len(clean_rows)} available rows")
    train_rows = rng.sample(clean_rows, args.clean_count)
    train_rows.extend(_make_hard_purple(args.seed + i, generator) for i in range(args.hard_count))
    rng.shuffle(train_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    with train_path.open("w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    shutil.copyfile(args.clean_validation, args.output_dir / "validation.jsonl")

    print(f"Wrote {len(train_rows)} rows to {train_path}")
    print(f"  clean={args.clean_count} hard_purple={args.hard_count}")
    print(f"Copied validation to {args.output_dir / 'validation.jsonl'}")


if __name__ == "__main__":
    main()
