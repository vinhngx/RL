#!/usr/bin/env python3
"""Focused numerical checks for the local visual-token auxiliary loss."""

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
        and node.name == "counterfactual_localization_loss"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["counterfactual_localization_loss"]


counterfactual_localization_loss = load_loss_function()


def compute(change_index: int):
    tokens = [torch.zeros(4, 3) for _ in range(4)]
    upper = torch.zeros(4, 3, requires_grad=True)
    with torch.no_grad():
        upper[change_index, 0] = 1
    tokens[2] = upper
    tokens[3] = upper
    loss, metrics = counterfactual_localization_loss(
        visual_tokens=tokens,
        grid_shapes=[(2, 2)] * 4,
        centers=torch.tensor([[0.25, 0.25]] * 4),
        sample_mask=torch.ones(4),
        global_valid_seqs=torch.tensor(4.0),
        temperature=0.1,
        sigma=0.25,
    )
    return loss, metrics, upper


def main() -> None:
    aligned_loss, aligned_metrics, _ = compute(0)
    wrong_loss, wrong_metrics, wrong_upper = compute(3)
    assert aligned_metrics["counterfactual_local_top1"] == 1.0
    assert wrong_metrics["counterfactual_local_top1"] == 0.0
    assert aligned_loss < wrong_loss
    wrong_loss.backward()
    assert wrong_upper.grad is not None
    assert torch.isfinite(wrong_upper.grad).all()
    assert wrong_upper.grad.abs().sum() > 0
    print("counterfactual local-token auxiliary checks passed")


if __name__ == "__main__":
    main()
