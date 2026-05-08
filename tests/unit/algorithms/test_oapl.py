import math

import torch

from nemo_rl.algorithms.advantage_estimator import OAPLAdvantageEstimator
from nemo_rl.algorithms.loss.loss_functions import OAPLLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_oapl_advantage_uses_logmeanexp_value_per_prompt():
    prompts = torch.tensor(
        [
            [1, 2],
            [1, 2],
            [1, 2],
            [3, 4],
            [3, 4],
            [3, 4],
        ]
    )
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    mask = torch.ones(6, 3)
    estimator = OAPLAdvantageEstimator(
        estimator_config={"name": "oapl"},
        loss_config={"vstar_beta": 1.0},
    )

    advantages = estimator.compute_advantage(prompts, rewards, mask)

    first_vstar = math.log((math.exp(0.0) + math.exp(1.0) + math.exp(0.0)) / 3.0)
    second_vstar = math.log((math.exp(1.0) + math.exp(1.0) + math.exp(0.0)) / 3.0)
    expected_sequence_advantages = torch.tensor(
        [
            0.0 - first_vstar,
            1.0 - first_vstar,
            0.0 - first_vstar,
            1.0 - second_vstar,
            1.0 - second_vstar,
            0.0 - second_vstar,
        ]
    )

    assert torch.allclose(
        advantages,
        expected_sequence_advantages.unsqueeze(-1).expand_as(mask),
    )


def test_oapl_loss_regresses_sequence_log_ratio_to_advantage():
    loss_fn = OAPLLossFn(
        {
            "name": "oapl",
            "vstar_beta": 1.0,
            "policy_beta": 0.5,
            "length_normalize_log_ratio": False,
            "sync_interval": 2,
        }
    )
    curr_logprobs = torch.tensor(
        [[0.2, -0.1], [0.4, 0.6]],
        requires_grad=True,
    )
    generation_logprobs = torch.tensor(
        [[0.0, 0.1, -0.3], [0.0, 0.1, 0.2]],
    )
    token_mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    sample_mask = torch.tensor([1.0, 1.0])
    advantages = torch.tensor([[0.0, 1.0, 1.0], [0.0, -0.5, -0.5]])
    data = BatchedDataDict(
        {
            "generation_logprobs": generation_logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": advantages,
        }
    )

    loss, metrics = loss_fn(
        curr_logprobs,
        data,
        global_valid_seqs=sample_mask.sum(),
        global_valid_toks=token_mask[:, 1:].sum(),
    )

    sequence_log_ratio = ((curr_logprobs - generation_logprobs[:, 1:]) * 1.0).sum(
        dim=-1
    )
    prediction = 0.5 * sequence_log_ratio
    expected = torch.square(prediction - torch.tensor([1.0, -0.5])).mean()

    assert torch.allclose(loss, expected)
    assert metrics["num_valid_samples"] == 2
    assert metrics["oapl_mse"] == loss.item()
    loss.backward()
    assert curr_logprobs.grad is not None


def test_oapl_loss_can_length_normalize_sequence_log_ratio():
    loss_fn = OAPLLossFn(
        {
            "name": "oapl",
            "vstar_beta": 1.0,
            "policy_beta": 1.0,
            "length_normalize_log_ratio": True,
            "sync_interval": 1,
        }
    )
    curr_logprobs = torch.tensor([[0.6, 0.2]])
    generation_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
    token_mask = torch.tensor([[0.0, 1.0, 1.0]])
    sample_mask = torch.tensor([1.0])
    advantages = torch.zeros(1, 3)
    data = BatchedDataDict(
        {
            "generation_logprobs": generation_logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": advantages,
        }
    )

    loss, _ = loss_fn(
        curr_logprobs,
        data,
        global_valid_seqs=sample_mask.sum(),
        global_valid_toks=token_mask[:, 1:].sum(),
    )

    assert torch.allclose(loss, torch.tensor(0.4**2))
