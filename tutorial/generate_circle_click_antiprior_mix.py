#!/usr/bin/env python3
"""Build a clean + targeted anti-prior Circle Click training split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path


SYSTEM_HINT = (
    " The target circle may be close to an image edge or corner. "
    "Click the actual visible center of the requested colored circle, "
    "and do not default to the image center or a common prior location."
)


def _load_circle_generator(repo_root: Path):
    module_path = repo_root / "3rdparty/Gym-workspace/Gym/resources_servers/circle_click/generate_data.py"
    spec = importlib.util.spec_from_file_location("circle_click_generate_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _set_tooling(example: dict) -> dict:
    params = example["responses_create_params"]
    params["tool_choice"] = {"type": "function", "name": "click"}
    system = params["input"][0]
    if system.get("role") == "system" and SYSTEM_HINT not in system["content"]:
        system["content"] += SYSTEM_HINT
    example["agent_ref"] = {"type": "responses_api_agents", "name": "circle_click_simple_agent"}
    return example


def _non_overlapping_position(
    rng: random.Random,
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
    raise RuntimeError("Could not place non-overlapping circle")


def _add_circle(
    rng: random.Random,
    circles: list[dict],
    color: str,
    radius: int,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
) -> None:
    pos = _non_overlapping_position(rng, radius, circles, x_range, y_range)
    circles.append({"x": pos["x"], "y": pos["y"], "radius": radius, "color": color})


def _make_case(seed: int, generator, kind: str) -> dict:
    rng = random.Random(seed)
    image_size = 1000
    radius = rng.randint(76, 106)
    circles: list[dict] = []

    if kind == "purple_bottom_left":
        target_color = "purple"
        _add_circle(rng, circles, "purple", radius, (110, 170), (730, 830))
        _add_circle(rng, circles, rng.choice(["pink", "cyan", "orange"]), radius, (410, 500), (820, 920))
        distractors = ["red", "yellow"]
    elif kind == "purple_mid":
        target_color = "purple"
        _add_circle(rng, circles, "purple", radius, (550, 640), (420, 520))
        _add_circle(rng, circles, rng.choice(["pink", "cyan", "orange"]), radius, (510, 610), (740, 840))
        distractors = ["red", "blue"]
    elif kind == "yellow_top":
        target_color = "yellow"
        _add_circle(rng, circles, "yellow", radius, (460, 540), (150, 230))
        _add_circle(rng, circles, rng.choice(["green", "red", "pink"]), radius, (450, 550), (450, 550))
        distractors = ["blue", "cyan"]
    else:
        raise ValueError(f"unknown kind: {kind}")

    for color in distractors:
        _add_circle(rng, circles, color, radius, (120, 880), (120, 880))

    rng.shuffle(circles)
    image_url = generator._generate_image(circles, image_size, radius)
    others = [c["color"] for c in circles if c["color"] != target_color]
    others_str = ", ".join(f"a {col} circle" for col in others)
    user_text = (
        f"This is a {image_size}x{image_size} pixel image. "
        f"It shows {others_str}, and a {target_color} circle. "
        f"Click on the {target_color} circle."
    )

    return _set_tooling(
        {
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
            "target_color": target_color,
            "antiprior_kind": kind,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-train", type=Path, required=True)
    parser.add_argument("--clean-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-count", type=int, default=3800)
    parser.add_argument("--hard-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=240447)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    generator = _load_circle_generator(repo_root)
    rng = random.Random(args.seed)

    clean_rows = [json.loads(line) for line in args.clean_train.read_text().splitlines()]
    if args.clean_count > len(clean_rows):
        raise ValueError(f"Requested {args.clean_count} clean rows from {len(clean_rows)} available rows")

    kinds = ["purple_bottom_left", "purple_mid", "yellow_top"]
    hard_rows = [_make_case(args.seed + i, generator, kinds[i % len(kinds)]) for i in range(args.hard_count)]
    train_rows = rng.sample(clean_rows, args.clean_count) + hard_rows
    rng.shuffle(train_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    shutil.copyfile(args.clean_validation, args.output_dir / "validation.jsonl")

    print(f"Wrote {len(train_rows)} rows to {train_path}")
    print(f"  clean={args.clean_count} hard_antiprior={args.hard_count}")
    print(f"Copied validation to {args.output_dir / 'validation.jsonl'}")


if __name__ == "__main__":
    main()
