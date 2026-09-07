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

# Generate SFT data for circle-count: gym images + templated CoT assistant with \boxed{N}.
import random
import sys
from pathlib import Path

sys.path.insert(0, "/opt/nemo-rl/3rdparty/Gym-workspace/Gym/resources_servers/circle_count")
from generate_data import make_example  # noqa: E402

TEMPLATES = [
    "Let me look at the image carefully and count the {color} circles. "
    "Scanning from the top left to the bottom right, I find them one by one. "
    "The total count is \\boxed{{{n}}}.",
    "I need to count the {color} circles in this image. "
    "Carefully going through each region of the image, "
    "I can identify the {color} circles. There are \\boxed{{{n}}} {color} circles.",
    "Counting the {color} circles: I'll scan the image systematically. "
    "After checking the whole image, I count exactly \\boxed{{{n}}} {color} circles.",
    "To count the {color} circles, I scan row by row. In the image I can spot "
    "each {color} circle clearly. Final answer: \\boxed{{{n}}}.",
    "Looking at this image, I count the {color} circles. Going through them "
    "carefully, the count of {color} circles is \\boxed{{{n}}}.",
    "First, I locate all the {color} circles. Then I tabulate them: 1, 2, 3... "
    "The final count of {color} circles is \\boxed{{{n}}}.",
    "I'll hover over the image and mark each {color} circle mentally. "
    "After a full pass, I have counted \\boxed{{{n}}}.",
    "Careful count: there are \\boxed{{{n}}} {color} circles in the image.",
    "Starting at the top of the image, I track each {color} circle downward. "
    "Tally complete: \\boxed{{{n}}}.",
    "The image contains several colored circles. Isolating the {color} ones "
    "and counting them gives \\boxed{{{n}}}.",
    "I divide the image into four quadrants and count {color} circles in each. "
    "The quadrant counts sum to \\boxed{{{n}}}.",
    "Let me be systematic: for each {color} circle I see, I increase my counter "
    "by one. The counter reads \\boxed{{{n}}} at the end.",
]

TASK = "circle-count-sft"


def main():
    n, out, seed_offset = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40000
    nmin = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    nmax = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    import json

    rng = random.Random(777)
    rows = []
    for i in range(n):
        ex = make_example(seed_offset + i, num_circles_range=(nmin, nmax))
        expected = sum(1 for c in ex["circles"] if c["color"] == ex["target_color"])
        params = ex["responses_create_params"]
        system = next(m["content"] for m in params["input"] if m["role"] == "system")
        user_content = []
        for content in next(m for m in params["input"] if m["role"] == "user")["content"]:
            if content["type"] == "input_image":
                user_content.append({"type": "image", "image": content["image_url"]})
            else:
                user_content.append({"type": "text", "text": content["text"]})
        assistant = rng.choice(TEMPLATES).format(color=ex["target_color"], n=expected)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant},
                ],
                "task_name": TASK,
            }
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {n} SFT examples to {out}")


if __name__ == "__main__":
    main()
