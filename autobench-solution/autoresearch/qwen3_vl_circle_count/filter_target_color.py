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

from PIL import Image, ImageChops, ImageDraw


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


def count_components(mask: Image.Image) -> int:
    """Count disconnected exact-color regions in a binary Pillow mask."""
    working = mask.copy()
    count = 0
    while (bounds := working.getbbox()) is not None:
        cropped = working.crop(bounds)
        offset = cropped.tobytes().find(b"\xff")
        if offset < 0:
            raise RuntimeError("Non-empty mask had no foreground pixel")
        x = bounds[0] + offset % cropped.width
        y = bounds[1] + offset // cropped.width
        ImageDraw.floodfill(working, (x, y), 0)
        count += 1
    return count


def draw_grid(count: int, color: tuple[int, int, int], size: tuple[int, int]) -> Image.Image:
    """Render a regular visual counting grid without writing the numeral."""
    image = Image.new("RGB", size, "white")
    if count == 0:
        return image

    columns = min(5, count)
    rows = (count + columns - 1) // columns
    y_gap = size[1] / (rows + 1)
    radius = min(55, int(min(size[0] / 6, y_gap) * 0.3))
    draw = ImageDraw.Draw(image)
    for row in range(rows):
        row_start = row * columns
        row_count = min(columns, count - row_start)
        x_gap = size[0] / (row_count + 1)
        for column in range(row_count):
            x = round((column + 1) * x_gap)
            y = round((row + 1) * y_gap)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def filter_image(image_url: str, target_color: str, layout: str) -> str:
    prefix = "data:image/png;base64,"
    if not image_url.startswith(prefix):
        raise ValueError("Expected a base64 PNG data URL")

    image = Image.open(io.BytesIO(base64.b64decode(image_url.removeprefix(prefix))))
    image = image.convert("RGB")
    target = Image.new("RGB", image.size, COLORS[target_color])
    difference = ImageChops.difference(image, target).convert("L")
    exact_match_mask = difference.point(lambda value: 255 if value == 0 else 0)
    if layout == "grid":
        filtered = draw_grid(
            count_components(exact_match_mask), COLORS[target_color], image.size
        )
    else:
        filtered = Image.new("RGB", image.size, "white")
        filtered.paste(image, mask=exact_match_mask)

    buffer = io.BytesIO()
    filtered.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return prefix + encoded


def filter_example(example: dict, layout: str) -> dict:
    target_color = example["target_color"]
    if target_color not in COLORS:
        raise ValueError(f"Unknown target color: {target_color}")

    content = example["responses_create_params"]["input"][1]["content"]
    image_item = next(item for item in content if item["type"] == "input_image")
    image_item["image_url"] = filter_image(
        image_item["image_url"], target_color, layout
    )
    example["image_preprocessing"] = f"exact_target_color_{layout}_v1"
    return example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("isolate", "grid"), default="isolate")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as source, args.output.open("w") as destination:
        for line in source:
            example = filter_example(json.loads(line), args.layout)
            destination.write(json.dumps(example, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
