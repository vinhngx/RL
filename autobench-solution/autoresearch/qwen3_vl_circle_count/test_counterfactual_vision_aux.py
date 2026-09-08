#!/usr/bin/env python3
"""Focused numerical checks for the counterfactual vision auxiliary loss."""

from __future__ import annotations

import ast
from pathlib import Path

import torch


def load_loss_function():
    source_path = Path("/opt/nemo-rl/nemo_rl/models/automodel/train.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "counterfactual_vision_direction_loss"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["counterfactual_vision_direction_loss"]


counterfactual_vision_direction_loss = load_loss_function()


def compute(upper_sign: float) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    pooled = torch.zeros(4, 3, requires_grad=True)
    with torch.no_grad():
        pooled[2:, 0] = upper_sign
        if upper_sign < 0:
            pooled[2:, 1] = 0.5
    input_ids = torch.zeros(4, 4, dtype=torch.long)
    token_mask = torch.zeros_like(input_ids)
    # Qwen single-digit IDs are 15..24 for text digits 0..9.
    input_ids[0, 2] = 19  # 4
    input_ids[2, 2] = 20  # 5
    token_mask[0, 2] = 1
    token_mask[2, 2] = 1
    weights = torch.zeros(25, 3)
    weights[20, 0] = 1
    loss, metrics = counterfactual_vision_direction_loss(
        pooled=pooled,
        input_ids=input_ids,
        token_mask=token_mask,
        sample_mask=torch.ones(4),
        count_direction_weights=weights,
        global_valid_seqs=torch.tensor(4.0),
        margin=0.02,
        temperature=0.01,
    )
    return loss, metrics, pooled


def main() -> None:
    aligned_loss, aligned_metrics, _ = compute(1.0)
    opposed_loss, opposed_metrics, opposed_pooled = compute(-1.0)
    assert aligned_metrics["counterfactual_vision_positive"] == 1.0
    assert opposed_metrics["counterfactual_vision_positive"] == 0.0
    assert aligned_metrics["counterfactual_vision_cosine"] > 0.99
    assert opposed_metrics["counterfactual_vision_cosine"] < 0
    assert aligned_loss < opposed_loss
    opposed_loss.backward()
    assert opposed_pooled.grad is not None
    assert torch.isfinite(opposed_pooled.grad).all()
    assert opposed_pooled.grad.abs().sum() > 0
    print("counterfactual vision auxiliary loss checks passed")


if __name__ == "__main__":
    main()
