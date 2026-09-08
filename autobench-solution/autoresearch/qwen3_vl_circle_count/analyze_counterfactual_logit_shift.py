# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Measure whether a +1 image moves the deployed boxed-answer logit margin."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from analyze_answer_margins import first_divergent_tokens, percentile
from analyze_counterfactual_vision_direction import unpack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--batch-pairs", type=int, default=2)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    shifts = [row["counterfactual_logit_shift"] for row in rows]
    return {
        "pairs": len(rows),
        "mean": sum(shifts) / len(shifts),
        "median": percentile(shifts, 0.5),
        "p10": percentile(shifts, 0.1),
        "p90": percentile(shifts, 0.9),
        "positive": sum(shift > 0 for shift in shifts),
        "positive_rate": sum(shift > 0 for shift in shifts) / len(shifts),
    }


def main() -> None:
    args = parse_args()
    if args.pairs <= 0 or args.batch_pairs <= 0:
        raise ValueError("pair counts must be positive")
    rows = [json.loads(line) for line in args.data.open()]
    if len(rows) < 2 * args.pairs:
        raise ValueError("source does not contain enough pairs")
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

    jobs = []
    for pair_index in range(args.pairs):
        lower_row, upper_row = rows[2 * pair_index : 2 * pair_index + 2]
        if lower_row["pair_id"] != upper_row["pair_id"]:
            raise ValueError(f"noncontiguous pair at {pair_index}")
        lower, upper = unpack(lower_row), unpack(upper_row)
        if upper[2] != lower[2] + 1:
            raise ValueError(f"pair {pair_index} is not adjacent")
        _, lower_id, upper_id = first_divergent_tokens(
            processor.tokenizer, lower[2], upper[2]
        )
        pair_job = {
            "pair_id": lower_row["pair_id"],
            "lower_count": lower[2],
            "lower_token_id": lower_id,
            "upper_token_id": upper_id,
            "images": [lower[0], upper[0]],
            "texts": [],
        }
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
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            pair_job["texts"].append(rendered + r"\boxed{")
        jobs.append(pair_job)

    results = []
    with torch.inference_mode():
        for start in range(0, len(jobs), args.batch_pairs):
            batch = jobs[start : start + args.batch_pairs]
            inputs = processor(
                text=[text for job in batch for text in job["texts"]],
                images=[image for job in batch for image in job["images"]],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            logits = model(**inputs).logits[:, -1, :].float()
            for local_index, job in enumerate(batch):
                lower_logits, upper_logits = logits[2 * local_index : 2 * local_index + 2]
                lower_id = job["lower_token_id"]
                upper_id = job["upper_token_id"]
                base_margin = (lower_logits[upper_id] - lower_logits[lower_id]).item()
                upper_margin = (upper_logits[upper_id] - upper_logits[lower_id]).item()
                results.append(
                    {
                        "pair_id": job["pair_id"],
                        "lower_count": job["lower_count"],
                        "base_upper_vs_lower_logit": base_margin,
                        "upper_upper_vs_lower_logit": upper_margin,
                        "counterfactual_logit_shift": upper_margin - base_margin,
                    }
                )
            print(f"scored={min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)

    by_boundary: dict[int, list[dict]] = defaultdict(list)
    for row in results:
        by_boundary[row["lower_count"]].append(row)
    summary = {
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
