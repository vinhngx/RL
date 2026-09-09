# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize same-example checkpoint evaluations and paired disagreements."""

import argparse
import json
import math
from pathlib import Path


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    """Two-sided exact binomial p-value for discordant paired outcomes."""
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    predictions: dict[str, list[dict]] = {}
    for summary_path in sorted(args.results_root.glob("*.summary.json")):
        label = summary_path.name.removesuffix(".summary.json")
        summary = json.loads(summary_path.read_text())
        prediction_path = args.results_root / f"{label}.jsonl"
        with prediction_path.open() as source:
            model_predictions = [json.loads(line) for line in source]
        if len(model_predictions) != summary["total"]:
            raise ValueError(f"prediction count mismatch for {label}")
        predictions[label] = model_predictions
        rows.append(
            {
                "label": label,
                "correct": summary["correct"],
                "total": summary["total"],
                "accuracy": summary["accuracy"],
                "boxed_parseable": summary["boxed_parseable"],
                "wilson_95": summary["wilson_95"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    if not rows:
        raise ValueError("no evaluation summaries found")
    totals = {row["total"] for row in rows}
    if len(totals) != 1:
        raise ValueError("checkpoints were not evaluated on equal-sized sets")

    rows.sort(key=lambda row: (-row["accuracy"], row["label"]))
    winner = rows[0]["label"]
    winner_outcomes = [item["correct"] for item in predictions[winner]]
    paired = []
    for row in rows[1:]:
        outcomes = [item["correct"] for item in predictions[row["label"]]]
        winner_only = sum(a and not b for a, b in zip(winner_outcomes, outcomes, strict=True))
        other_only = sum(b and not a for a, b in zip(winner_outcomes, outcomes, strict=True))
        paired.append(
            {
                "winner": winner,
                "other": row["label"],
                "winner_only_correct": winner_only,
                "other_only_correct": other_only,
                "exact_mcnemar_p": exact_mcnemar_p(winner_only, other_only),
            }
        )

    total = rows[0]["total"]
    all_outcomes = [
        [item["correct"] for item in predictions[row["label"]]] for row in rows
    ]
    oracle_correct = sum(any(outcomes[index] for outcomes in all_outcomes) for index in range(total))
    unanimous_errors = [
        index
        for index in range(total)
        if not any(outcomes[index] for outcomes in all_outcomes)
    ]
    result = {
        "ranking": rows,
        "paired_vs_winner": paired,
        "oracle_any_checkpoint_correct": oracle_correct,
        "unanimous_error_count": len(unanimous_errors),
        "unanimous_error_indices": unanimous_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# Fresh 1K checkpoint comparison",
        "",
        "| Rank | Checkpoint | Correct | Accuracy | Wilson 95% | Boxed |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lower, upper = row["wilson_95"]
        markdown.append(
            f"| {rank} | `{row['label']}` | {row['correct']}/{row['total']} | "
            f"{row['accuracy']:.1%} | {lower:.1%}--{upper:.1%} | "
            f"{row['boxed_parseable']}/{row['total']} |"
        )
    markdown.extend(
        [
            "",
            f"Best checkpoint: `{winner}`.",
            f"Oracle union: {oracle_correct}/{total}; unanimous errors: {len(unanimous_errors)}.",
            "",
            "## Paired comparisons against the winner",
            "",
            "| Other checkpoint | Winner-only correct | Other-only correct | Exact McNemar p |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison in paired:
        markdown.append(
            f"| `{comparison['other']}` | {comparison['winner_only_correct']} | "
            f"{comparison['other_only_correct']} | {comparison['exact_mcnemar_p']:.4g} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main()
