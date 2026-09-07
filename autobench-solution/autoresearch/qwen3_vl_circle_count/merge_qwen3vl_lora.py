#!/usr/bin/env python3
"""Merge a Qwen3-VL PEFT adapter and save a standalone HF checkpoint."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"missing PEFT weights under {args.adapter}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output}")

    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    args.output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
