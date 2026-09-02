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

"""Suppress non-target colors in Circle Count JSONL images without using labels."""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageChops


COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (133, 153, 0),
    "yellow": (181, 137, 0),
    "purple": (108, 113, 196),
    "orange": (203, 75, 22),
    "cyan": (42, 161, 152),
    "pink": (211, 54, 130),
}


def filter_image(image_url: str, target_color: str) -> str:
    prefix = "data:image/png;base64,"
    if not image_url.startswith(prefix):
        raise ValueError("Expected a base64 PNG data URL")

    image = Image.open(io.BytesIO(base64.b64decode(image_url.removeprefix(prefix))))
    image = image.convert("RGB")
    target = Image.new("RGB", image.size, COLORS[target_color])
    difference = ImageChops.difference(image, target).convert("L")
    exact_match_mask = difference.point(lambda value: 255 if value == 0 else 0)
    filtered = Image.new("RGB", image.size, "white")
    filtered.paste(image, mask=exact_match_mask)

    buffer = io.BytesIO()
    filtered.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return prefix + encoded


def filter_example(example: dict) -> dict:
    target_color = example["target_color"]
    if target_color not in COLORS:
        raise ValueError(f"Unknown target color: {target_color}")

    content = example["responses_create_params"]["input"][1]["content"]
    image_item = next(item for item in content if item["type"] == "input_image")
    image_item["image_url"] = filter_image(image_item["image_url"], target_color)
    example["image_preprocessing"] = "exact_target_color_isolation_v1"
    return example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as source, args.output.open("w") as destination:
        for line in source:
            example = filter_example(json.loads(line))
            destination.write(json.dumps(example, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
