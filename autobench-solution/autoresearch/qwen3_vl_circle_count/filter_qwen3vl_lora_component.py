#!/usr/bin/env python3
"""Split a Qwen3-VL PEFT adapter into language-only or vision-only weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=("language", "vision"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    keep_vision = args.component == "vision"
    tensors = load_file(args.input / WEIGHTS_NAME, device="cpu")
    tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if (".visual." in name) == keep_vision
    }
    if not tensors:
        raise ValueError(f"no {args.component} tensors found")

    with (args.input / CONFIG_NAME).open() as source:
        config = json.load(source)
    config["target_modules"] = [
        name
        for name in config["target_modules"]
        if (".visual." in name) == keep_vision
    ]
    with (args.output / CONFIG_NAME).open("w") as destination:
        json.dump(config, destination, indent=2, sort_keys=True)
        destination.write("\n")
    for source in args.input.iterdir():
        if source.is_file() and source.name not in (WEIGHTS_NAME, CONFIG_NAME):
            shutil.copy2(source, args.output / source.name)
    with safe_open(args.input / WEIGHTS_NAME, framework="pt", device="cpu") as source:
        metadata = source.metadata()
    save_file(tensors, args.output / WEIGHTS_NAME, metadata=metadata)
    print(
        json.dumps(
            {
                "component": args.component,
                "target_modules": len(config["target_modules"]),
                "tensors": len(tensors),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
