#!/usr/bin/env python3
"""Focused numerical checks for direct circle-color supervision."""

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
        and node.name == "circle_color_alignment_loss"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["circle_color_alignment_loss"]


circle_color_alignment_loss = load_loss_function()


def compute(aligned: bool):
    tokens = torch.zeros(4, 3, requires_grad=True)
    with torch.no_grad():
        tokens[0, 1 if aligned else 2] = 1
    loss, metrics = circle_color_alignment_loss(
        visual_tokens=[tokens],
        grid_shapes=[(2, 2)],
        annotations=torch.tensor([[[0.25, 0.25, 1.0], [0.0, 0.0, -1.0]]]),
        sample_mask=torch.ones(1),
        color_vectors=torch.eye(3),
        temperature=0.1,
    )
    return loss, metrics, tokens


def main() -> None:
    aligned_loss, aligned_metrics, _ = compute(True)
    wrong_loss, wrong_metrics, wrong_tokens = compute(False)
    assert aligned_metrics["circle_color_accuracy"] == 1.0
    assert wrong_metrics["circle_color_accuracy"] == 0.0
    assert aligned_metrics["circle_color_count"] == 1.0
    assert aligned_loss < wrong_loss
    wrong_loss.backward()
    assert wrong_tokens.grad is not None
    assert torch.isfinite(wrong_tokens.grad).all()
    assert wrong_tokens.grad.abs().sum() > 0
    print("circle color auxiliary checks passed")


if __name__ == "__main__":
    main()
