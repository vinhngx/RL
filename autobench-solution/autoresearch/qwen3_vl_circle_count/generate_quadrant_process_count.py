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

"""Generate full-image Circle Count rows with verifiable quadrant subtotals."""

import argparse
import json
from pathlib import Path

from generate_balanced_circle_count import load_generator, make_balanced_example


def add_process_supervision(example: dict, *, image_size: int = 1000) -> dict:
    """Attach quadrant ground truth and a structured full-image question."""
    midpoint = image_size / 2
    target_color = example["target_color"]
    counts = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
    for circle in example["circles"]:
        if circle["color"] != target_color:
            continue
        vertical = "t" if circle["y"] < midpoint else "b"
        horizontal = "l" if circle["x"] < midpoint else "r"
        counts[vertical + horizontal] += 1

    counts["total"] = sum(counts.values())
    user_content = example["responses_create_params"]["input"][1]["content"]
    user_content[1]["text"] = (
        f"How many {target_color} circles are in the image? First count the "
        "circles whose centers lie in each image quadrant. Reply with five "
        "plain integers and no angle brackets, using this schema: "
        r"TL=2 TR=1 BL=0 BR=3 \boxed{6}. Replace every example number with "
        "your counts."
    )
    example["process_ground_truth"] = json.dumps(counts, separators=(",", ":"))
    return example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-count", type=int, default=256)
    parser.add_argument("--min-target-count", type=int, default=7)
    parser.add_argument("--max-target-count", type=int, default=14)
    parser.add_argument("--seed-offset", type=int, default=1_400_000)
    args = parser.parse_args()

    generator = load_generator(args.generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        example_index = 0
        for target_count in range(args.min_target_count, args.max_target_count + 1):
            for repetition in range(args.examples_per_count):
                example = make_balanced_example(
                    generator,
                    seed=args.seed_offset + example_index,
                    target_count=target_count,
                    target_color=tuple(generator.COLORS)[
                        repetition % len(generator.COLORS)
                    ],
                    min_total_circles=14,
                    max_total_circles=20,
                )
                add_process_supervision(example)
                output.write(json.dumps(example, separators=(",", ":")) + "\n")
                example_index += 1


if __name__ == "__main__":
    main()
