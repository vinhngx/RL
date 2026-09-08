#!/usr/bin/env python3
"""Focused numerical checks for differing-token SimPO."""

from __future__ import annotations

import torch

from nemo_rl.algorithms.loss.loss_functions import DPOLossFn


def test_difference_only_ignores_shared_completion_tokens() -> None:
    cfg = {
        "reference_policy_kl_penalty": 2.0,
        "preference_loss_weight": 1.0,
        "sft_loss_weight": 0.0,
        "preference_average_log_probs": False,
        "sft_average_log_probs": False,
        "preference_algorithm": "simpo",
        "simpo_gamma": 1.0,
        "preference_difference_only": True,
    }
    loss_fn = DPOLossFn(cfg)
    # Sequences differ only at target-token position 3 (next-token-logprob 2).
    input_ids = torch.tensor([[1, 2, 3, 7, 4], [1, 2, 3, 6, 4]])
    token_mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
    data = {
        "input_ids": input_ids,
        "reference_policy_logprobs": torch.zeros(2, 5),
        "token_mask": token_mask,
        "sample_mask": torch.ones(2),
    }
    base = torch.tensor([[9.0, 9.0, -0.1, -20.0], [9.0, 9.0, -2.1, 20.0]])
    loss, metrics = loss_fn(base, data, torch.tensor(2.0), None)
    changed_shared = base.clone()
    changed_shared[:, 3] = torch.tensor([100.0, -100.0])
    shared_loss, shared_metrics = loss_fn(
        changed_shared, data, torch.tensor(2.0), None
    )
    assert torch.allclose(loss, shared_loss)
    assert metrics["accuracy"] == shared_metrics["accuracy"] == 1.0


def test_difference_only_rejects_identical_valid_pair() -> None:
    cfg = {
        "reference_policy_kl_penalty": 2.0,
        "preference_loss_weight": 1.0,
        "sft_loss_weight": 0.0,
        "preference_average_log_probs": False,
        "sft_average_log_probs": False,
        "preference_algorithm": "simpo",
        "simpo_gamma": 1.0,
        "preference_difference_only": True,
    }
    loss_fn = DPOLossFn(cfg)
    data = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
        "reference_policy_logprobs": torch.zeros(2, 3),
        "token_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
        "sample_mask": torch.ones(2),
    }
    try:
        loss_fn(torch.zeros(2, 2), data, torch.tensor(2.0), None)
    except ValueError as error:
        assert "without a differing response token" in str(error)
    else:
        raise AssertionError("identical valid preference pair was accepted")
