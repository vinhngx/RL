# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any


G_REPO_ROOT = Path(__file__).resolve().parents[2]
G_STAR_COUNT_DIR = (
    G_REPO_ROOT
    / "3rdparty"
    / "Gym-workspace"
    / "Gym"
    / "resources_servers"
    / "star_count"
)
sys.path.insert(0, str(G_STAR_COUNT_DIR))

from generate_data import make_example  # noqa: E402


def _extract_content(example: dict[str, Any]) -> tuple[str, str]:
    user_content = example["responses_create_params"]["input"][1]["content"]
    image_url = next(
        item["image_url"] for item in user_content if item["type"] == "input_image"
    )
    text = next(item["text"] for item in user_content if item["type"] == "input_text")
    return image_url, text


def _write_png(image_url: str, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(image_url.removeprefix("data:image/png;base64,"))
    image_path.write_bytes(image_bytes)


def _make_sft_row(example: dict[str, Any], image_path: Path) -> dict[str, Any]:
    _, user_text = _extract_content(example)
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a visual counting assistant. Return only the requested JSON object.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(example["expected_counts"], sort_keys=True),
                    }
                ],
            },
        ]
    }


def _write_split(
    *,
    output_path: Path,
    image_dir: Path,
    count: int,
    seed_offset: int,
    star_radius_range: tuple[int, int],
    count_per_color_range: tuple[int, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        for index in range(count):
            example = make_example(
                seed_offset + index,
                star_radius_range=star_radius_range,
                count_per_color_range=count_per_color_range,
            )
            image_url, _ = _extract_content(example)
            image_path = image_dir / f"{output_path.stem}_{index:06d}.png"
            _write_png(image_url, image_path)
            row = _make_sft_row(example, image_path)
            output_file.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare star_count SFT train/validation JSONL files."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-n", type=int, default=512)
    parser.add_argument("--val-n", type=int, default=128)
    parser.add_argument("--eval-n", type=int, default=64)
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--radius-min", type=int, default=16)
    parser.add_argument("--radius-max", type=int, default=28)
    parser.add_argument("--count-min", type=int, default=1)
    parser.add_argument("--count-max", type=int, default=3)
    args = parser.parse_args()

    image_dir = args.out_dir / "images"
    star_radius_range = (args.radius_min, args.radius_max)
    count_per_color_range = (args.count_min, args.count_max)
    _write_split(
        output_path=args.out_dir / "train.jsonl",
        image_dir=image_dir,
        count=args.train_n,
        seed_offset=args.seed_offset,
        star_radius_range=star_radius_range,
        count_per_color_range=count_per_color_range,
    )
    _write_split(
        output_path=args.out_dir / "val.jsonl",
        image_dir=image_dir,
        count=args.val_n,
        seed_offset=args.seed_offset + args.train_n,
        star_radius_range=star_radius_range,
        count_per_color_range=count_per_color_range,
    )
    _write_split(
        output_path=args.out_dir / "eval.jsonl",
        image_dir=image_dir,
        count=args.eval_n,
        seed_offset=args.seed_offset + args.train_n + args.val_n,
        star_radius_range=star_radius_range,
        count_per_color_range=count_per_color_range,
    )
    print(f"Wrote star_count SFT data to {args.out_dir}")


if __name__ == "__main__":
    main()
