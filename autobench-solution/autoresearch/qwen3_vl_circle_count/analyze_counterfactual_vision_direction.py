# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Probe whether +1 image changes align with the decoder count direction."""

import argparse
import base64
import io
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=256)
    return parser.parse_args()


def unpack(row: dict) -> tuple[Image.Image, str, int]:
    content = row["responses_create_params"]["input"][1]["content"]
    url = next(item["image_url"] for item in content if item["type"] == "input_image")
    question = next(item["text"] for item in content if item["type"] == "input_text")
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert(
        "RGB"
    )
    expected = sum(
        circle["color"] == row["target_color"] for circle in row["circles"]
    )
    return image, question, expected


def divergent_ids(tokenizer, lower: int, upper: int) -> tuple[int, int]:
    lower_ids = tokenizer.encode(str(lower), add_special_tokens=False)
    upper_ids = tokenizer.encode(str(upper), add_special_tokens=False)
    for lower_id, upper_id in zip(lower_ids, upper_ids, strict=False):
        if lower_id != upper_id:
            return lower_id, upper_id
    raise ValueError(f"Counts do not have a divergent token: {lower}, {upper}")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> None:
    args = parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    rows = [json.loads(line) for line in args.data.open()]
    if len(rows) < 2 * args.pairs:
        raise ValueError(f"Requested {args.pairs} pairs from only {len(rows)} rows")
    prompt = args.prompt_file.read_text().strip()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_visual(_module, _inputs, output) -> None:
        captured["pooler_output"] = output.pooler_output

    hook = model.model.visual.register_forward_hook(capture_visual)
    results = []
    merge_size = processor.image_processor.merge_size
    with torch.inference_mode():
        for pair_index in range(args.pairs):
            lower_row, upper_row = rows[2 * pair_index : 2 * pair_index + 2]
            if lower_row.get("pair_id") != upper_row.get("pair_id"):
                raise ValueError(f"Rows for pair {pair_index} are not contiguous")
            lower = unpack(lower_row)
            upper = unpack(upper_row)
            if upper[2] != lower[2] + 1:
                raise ValueError(f"Pair {pair_index} is not a +1 count pair")
            images = [lower[0], upper[0]]
            texts = []
            for image, question, _expected in (lower, upper):
                messages = [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
                texts.append(
                    processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            inputs = processor(
                text=texts, images=images, padding=True, return_tensors="pt"
            ).to("cuda")
            model(**inputs)
            vision = captured.pop("pooler_output")
            sizes = (
                inputs["image_grid_thw"].prod(-1) // (merge_size * merge_size)
            ).tolist()
            pooled = torch.stack(
                [chunk.float().mean(0) for chunk in vision.split(sizes)]
            )
            lower_id, upper_id = divergent_ids(
                processor.tokenizer, lower[2], upper[2]
            )
            direction = (
                model.lm_head.weight[upper_id] - model.lm_head.weight[lower_id]
            ).float()
            delta = pooled[1] - pooled[0]
            cosine = F.cosine_similarity(delta, direction, dim=0).item()
            results.append(
                {
                    "pair_id": lower_row["pair_id"],
                    "lower_count": lower[2],
                    "upper_count": upper[2],
                    "lower_token_id": lower_id,
                    "upper_token_id": upper_id,
                    "cosine": cosine,
                    "positive": cosine > 0,
                    "delta_norm": delta.norm().item(),
                }
            )
            if (pair_index + 1) % 16 == 0:
                print(f"scored={pair_index + 1}/{args.pairs}", flush=True)
    hook.remove()

    cosines = [row["cosine"] for row in results]
    summary = {
        "pairs": len(results),
        "positive": sum(row["positive"] for row in results),
        "positive_rate": sum(row["positive"] for row in results) / len(results),
        "cosine": {
            "mean": sum(cosines) / len(cosines),
            "p10": percentile(cosines, 0.1),
            "median": percentile(cosines, 0.5),
            "p90": percentile(cosines, 0.9),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for row in results:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

