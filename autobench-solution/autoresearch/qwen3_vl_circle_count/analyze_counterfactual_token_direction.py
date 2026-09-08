# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Probe local visual-token alignment on same-geometry +1 image pairs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from analyze_counterfactual_vision_direction import (
    divergent_ids,
    percentile,
    unpack,
)


TOP_K = (1, 2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=256)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for key in ["mean"] + [f"top{k}" for k in TOP_K]:
        values = [row[f"{key}_cosine"] for row in rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "median": percentile(values, 0.5),
            "positive_rate": sum(value > 0 for value in values) / len(values),
        }
    return summary


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
    captured: dict[str, torch.Tensor] = {}

    def capture_visual(_module, _inputs, output) -> None:
        captured["tokens"] = output.pooler_output

    hook = model.model.visual.register_forward_hook(capture_visual)
    merge_size = processor.image_processor.merge_size
    results = []
    with torch.inference_mode():
        for pair_index in range(args.pairs):
            lower_row, upper_row = rows[2 * pair_index : 2 * pair_index + 2]
            if lower_row["pair_id"] != upper_row["pair_id"]:
                raise ValueError(f"noncontiguous pair at {pair_index}")
            lower, upper = unpack(lower_row), unpack(upper_row)
            if upper[2] != lower[2] + 1:
                raise ValueError(f"pair {pair_index} is not adjacent")
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
            sizes = (
                inputs["image_grid_thw"].prod(-1) // (merge_size * merge_size)
            ).tolist()
            chunks = captured.pop("tokens").float().split(sizes)
            if chunks[0].shape != chunks[1].shape:
                raise ValueError(f"visual grids differ for pair {pair_index}")
            token_delta = chunks[1] - chunks[0]
            lower_id, upper_id = divergent_ids(
                processor.tokenizer, lower[2], upper[2]
            )
            direction = (
                model.lm_head.weight[upper_id] - model.lm_head.weight[lower_id]
            ).float()
            norms = token_delta.norm(dim=-1)
            result = {
                "pair_id": lower_row["pair_id"],
                "lower_count": lower[2],
                "mean_cosine": F.cosine_similarity(
                    token_delta.mean(0), direction, dim=0
                ).item(),
            }
            for k in TOP_K:
                actual_k = min(k, token_delta.shape[0])
                indices = norms.topk(actual_k).indices
                aggregate = token_delta[indices].mean(0)
                result[f"top{k}_cosine"] = F.cosine_similarity(
                    aggregate, direction, dim=0
                ).item()
            results.append(result)
            if (pair_index + 1) % 16 == 0:
                print(f"scored={pair_index + 1}/{args.pairs}", flush=True)
    hook.remove()

    by_boundary: dict[int, list[dict]] = defaultdict(list)
    for row in results:
        by_boundary[row["lower_count"]].append(row)
    summary = {
        "pairs": len(results),
        "overall": summarize(results),
        "by_boundary": {
            f"{lower}->{lower + 1}": summarize(boundary_rows)
            for lower, boundary_rows in sorted(by_boundary.items())
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
