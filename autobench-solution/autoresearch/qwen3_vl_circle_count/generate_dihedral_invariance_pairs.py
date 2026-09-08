# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pair each Circle Count image with a lossless dihedral transform."""

import argparse
import base64
import copy
import io
import json
from pathlib import Path

from PIL import Image


TRANSFORMS = ("hflip", "vflip", "rot180")


def transform_image(data_url: str, transform: str) -> str:
    header, payload = data_url.split(",", 1)
    image = Image.open(io.BytesIO(base64.b64decode(payload)))
    operation = {
        "hflip": Image.Transpose.FLIP_LEFT_RIGHT,
        "vflip": Image.Transpose.FLIP_TOP_BOTTOM,
        "rot180": Image.Transpose.ROTATE_180,
    }[transform]
    output = io.BytesIO()
    image.transpose(operation).save(output, format="PNG", optimize=False)
    return f"{header},{base64.b64encode(output.getvalue()).decode()}"


def transform_circle(circle: dict, transform: str, size: int = 1000) -> dict:
    circle = copy.deepcopy(circle)
    if transform in ("hflip", "rot180"):
        circle["x"] = size - 1 - circle["x"]
    if transform in ("vflip", "rot180"):
        circle["y"] = size - 1 - circle["y"]
    return circle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2048)
    args = parser.parse_args()

    source = [json.loads(line) for line in args.input.open()]
    if not 0 < args.pairs <= len(source):
        raise ValueError("--pairs must be positive and no larger than the input")

    rows = []
    for pair_id, original in enumerate(source[: args.pairs]):
        transform = TRANSFORMS[pair_id % len(TRANSFORMS)]
        a = copy.deepcopy(original)
        b = copy.deepcopy(original)
        a.update(pair_id=pair_id, orbit_variant="original", orbit_transform=transform)
        b.update(pair_id=pair_id, orbit_variant="transformed", orbit_transform=transform)
        content = b["responses_create_params"]["input"][1]["content"]
        content[0]["image_url"] = transform_image(content[0]["image_url"], transform)
        b["circles"] = [transform_circle(circle, transform) for circle in b["circles"]]
        rows.extend((a, b))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
