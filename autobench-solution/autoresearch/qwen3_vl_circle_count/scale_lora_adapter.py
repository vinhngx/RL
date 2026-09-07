# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Scale the effective delta of a PEFT LoRA adapter."""

import argparse
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


WEIGHTS_NAME = "adapter_model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_weights = args.input / WEIGHTS_NAME
    if not input_weights.is_file():
        raise FileNotFoundError(f"missing PEFT weights: {input_weights}")
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    for source in args.input.iterdir():
        if source.name != WEIGHTS_NAME and source.is_file():
            shutil.copy2(source, args.output / source.name)

    tensors = load_file(input_weights, device="cpu")
    scaled_count = 0
    for name, tensor in tensors.items():
        if "lora_B" in name:
            tensors[name] = tensor * args.scale
            scaled_count += 1
    if scaled_count == 0:
        raise ValueError(f"no LoRA-B tensors found in {input_weights}")

    with safe_open(input_weights, framework="pt", device="cpu") as source:
        metadata = source.metadata()
    save_file(tensors, args.output / WEIGHTS_NAME, metadata=metadata)
    print(f"scaled {scaled_count} LoRA-B tensors by {args.scale:g}")


if __name__ == "__main__":
    main()
