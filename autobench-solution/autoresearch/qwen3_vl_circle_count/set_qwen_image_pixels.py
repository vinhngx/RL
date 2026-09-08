# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Set a fixed native Qwen image-processing area on a local checkpoint."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-pixels", type=int, required=True)
    args = parser.parse_args()
    if args.image_pixels <= 0:
        raise ValueError("--image-pixels must be positive")

    config_path = args.checkpoint / "processor_config.json"
    config = json.loads(config_path.read_text())
    image_processor = config.get("image_processor")
    if not isinstance(image_processor, dict):
        raise ValueError(f"Missing image_processor in {config_path}")
    image_processor["size"] = {
        "longest_edge": args.image_pixels,
        "shortest_edge": args.image_pixels,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
