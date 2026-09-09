# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Probe frozen color-word alignment at known circle visual-token locations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from analyze_counterfactual_vision_direction import unpack


COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.data.open()]
    if args.pairs <= 0 or len(rows) < 2 * args.pairs:
        raise ValueError("invalid requested pair count")
    prompt = args.prompt_file.read_text().strip()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()
    embedding = model.get_input_embeddings().weight
    color_token_ids = {}
    color_vectors = []
    for color in COLORS:
        token_ids = processor.tokenizer.encode(" " + color, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"color {color} has no token IDs")
        color_token_ids[color] = token_ids
        color_vectors.append(embedding[token_ids].float().mean(0))
    color_vectors = F.normalize(torch.stack(color_vectors), dim=-1)
    captured: dict[str, torch.Tensor] = {}

    def capture_visual(_module, _inputs, output) -> None:
        captured["tokens"] = output.pooler_output

    hook = model.model.visual.register_forward_hook(capture_visual)
    merge_size = processor.image_processor.merge_size
    records = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    with torch.inference_mode():
        for pair_index in range(args.pairs):
            pair_rows = rows[2 * pair_index : 2 * pair_index + 2]
            images = []
            texts = []
            for row in pair_rows:
                image, question, _expected = unpack(row)
                images.append(image)
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
            shapes = [
                (int(grid[1]) // merge_size, int(grid[2]) // merge_size)
                for grid in inputs["image_grid_thw"]
            ]
            token_chunks = captured.pop("tokens").float().split(
                [height * width for height, width in shapes]
            )
            for row, tokens, (height, width) in zip(
                pair_rows, token_chunks, shapes, strict=True
            ):
                normalized_tokens = F.normalize(tokens, dim=-1)
                for circle in row["circles"]:
                    grid_x = min(int(float(circle["x"]) / 1000 * width), width - 1)
                    grid_y = min(int(float(circle["y"]) / 1000 * height), height - 1)
                    similarities = normalized_tokens[grid_y * width + grid_x] @ color_vectors.T
                    predicted = COLORS[int(similarities.argmax())]
                    expected = circle["color"]
                    confusion[expected][predicted] += 1
                    records.append(
                        {
                            "pair_id": row["pair_id"],
                            "variant": row["counterfactual_variant"],
                            "target_color": row["target_color"],
                            "circle_color": expected,
                            "predicted_color": predicted,
                            "correct": predicted == expected,
                            "target_circle": expected == row["target_color"],
                            "margin": (
                                similarities[COLORS.index(expected)]
                                - similarities.topk(2).values[
                                    1
                                    if int(similarities.argmax())
                                    == COLORS.index(expected)
                                    else 0
                                ]
                            ).item(),
                        }
                    )
            if (pair_index + 1) % 16 == 0:
                print(f"scored={pair_index + 1}/{args.pairs}", flush=True)
    hook.remove()
    target_records = [row for row in records if row["target_circle"]]
    distractor_records = [row for row in records if not row["target_circle"]]
    summary = {
        "pairs": args.pairs,
        "circles": len(records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "target_accuracy": sum(row["correct"] for row in target_records)
        / len(target_records),
        "distractor_accuracy": sum(row["correct"] for row in distractor_records)
        / len(distractor_records),
        "color_token_ids": color_token_ids,
        "by_color": {
            color: {
                "correct": confusion[color][color],
                "total": sum(confusion[color].values()),
                "predictions": dict(confusion[color]),
            }
            for color in COLORS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for row in records:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
