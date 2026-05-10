import math

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import OAPLAdvantageEstimator
from nemo_rl.algorithms.grpo import _validate_oapl_config
from nemo_rl.algorithms.loss.loss_functions import OAPLLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _oapl_loss_config(policy_beta=1.0, length_normalize_log_ratio=False):
    return {
        "name": "oapl",
        "vstar_beta": 1.0,
        "policy_beta": policy_beta,
        "length_normalize_log_ratio": length_normalize_log_ratio,
        "sync_interval": 2,
    }


def _setup_oapl_loss_data(
    advantages: torch.Tensor,
    generation_logprobs: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> BatchedDataDict:
    batch_size, _ = advantages.shape
    if generation_logprobs is None:
        generation_logprobs = torch.zeros_like(advantages)
    if token_mask is None:
        token_mask = torch.ones_like(advantages)
        token_mask[:, 0] = 0
    if sample_mask is None:
        sample_mask = torch.ones(batch_size, dtype=advantages.dtype)

    return BatchedDataDict(
        {
            "generation_logprobs": generation_logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": advantages,
        }
    )


def _oapl_valid_counts(data: BatchedDataDict) -> tuple[torch.Tensor, torch.Tensor]:
    sample_mask = data["sample_mask"]
    valid_tokens = data["token_mask"][:, 1:] * sample_mask.unsqueeze(-1)
    return sample_mask.sum(), valid_tokens.sum()


def _oapl_master_config():
    return {
        "grpo": {
            "adv_estimator": {"name": "oapl"},
            "use_dynamic_sampling": False,
            "skip_reference_policy_logprobs_calculation": True,
            "seq_logprob_error_threshold": None,
            "val_period": 2,
        },
        "loss_fn": {
            "name": "oapl",
            "reference_policy_kl_penalty": 0.0,
            "use_importance_sampling_correction": False,
            "vstar_beta": 1.0,
            "policy_beta": 1.0e-3,
            "length_normalize_log_ratio": False,
            "sync_interval": 2,
        },
    }


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
    loss_fn = OAPLLossFn(_oapl_loss_config(policy_beta=0.5))
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
    assert metrics["oapl/residual_mean"] == pytest.approx(
        (prediction - torch.tensor([1.0, -0.5])).mean().item()
    )
    assert "token_mult_prob_error" in metrics
    assert "gen_kl_error" in metrics
    assert "policy_kl_error" in metrics
    assert "js_divergence_error" in metrics
    assert "approx_entropy" in metrics
    loss.backward()
    assert curr_logprobs.grad is not None


def test_oapl_loss_can_length_normalize_sequence_log_ratio():
    cfg = _oapl_loss_config(length_normalize_log_ratio=True)
    cfg["sync_interval"] = 1
    loss_fn = OAPLLossFn(cfg)
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


def test_oapl_loss_ignores_padding_prompt_tokens_and_invalid_samples():
    loss_fn = OAPLLossFn(_oapl_loss_config())
    advantages = torch.zeros(2, 5)
    advantages[0, 1:] = 0.25
    advantages[1, 1:] = 100.0
    token_mask = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    sample_mask = torch.tensor([1.0, 0.0])
    data = _setup_oapl_loss_data(
        advantages,
        token_mask=token_mask,
        sample_mask=sample_mask,
    )
    next_token_logprobs = torch.zeros(2, 4, requires_grad=True)
    global_valid_seqs, global_valid_toks = _oapl_valid_counts(data)

    loss, _ = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.25**2))
    assert torch.all(next_token_logprobs.grad[0][token_mask[0, 1:] == 0] == 0)
    assert torch.all(next_token_logprobs.grad[1] == 0)


def test_oapl_full_batch_loss_equals_summed_microbatch_loss():
    loss_fn = OAPLLossFn(_oapl_loss_config(policy_beta=0.75))
    advantages = torch.zeros(4, 4)
    targets = torch.tensor([0.1, -0.2, 0.3, -0.4])
    advantages[:, 1:] = targets.unsqueeze(-1)
    data = _setup_oapl_loss_data(advantages)
    next_token_logprobs = torch.zeros(4, 3)
    global_valid_seqs, global_valid_toks = _oapl_valid_counts(data)

    full_loss, _ = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )

    chunk0 = data.chunk(0, 2)
    chunk1 = data.chunk(1, 2)
    loss0, _ = loss_fn(
        next_token_logprobs=next_token_logprobs[:2],
        data=chunk0,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )
    loss1, _ = loss_fn(
        next_token_logprobs=next_token_logprobs[2:],
        data=chunk1,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )

    torch.testing.assert_close(full_loss, loss0 + loss1)


def test_oapl_config_validation_requires_matching_loss_and_estimator():
    config = _oapl_master_config()
    _validate_oapl_config(config)

    config["grpo"]["adv_estimator"] = {"name": "grpo"}
    with pytest.raises(ValueError, match="requires both"):
        _validate_oapl_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("grpo", "use_dynamic_sampling"), True, "dynamic_sampling"),
        (("loss_fn", "reference_policy_kl_penalty"), 0.1, "reference_policy"),
        (("loss_fn", "use_importance_sampling_correction"), True, "importance"),
        (
            ("grpo", "skip_reference_policy_logprobs_calculation"),
            False,
            "skip_reference",
        ),
        (("grpo", "seq_logprob_error_threshold"), 1.0, "seq_logprob"),
        (("grpo", "val_period"), 3, "sync boundaries"),
    ],
)
def test_oapl_config_validation_rejects_incompatible_settings(path, value, message):
    config = _oapl_master_config()
    section, key = path
    config[section][key] = value

    with pytest.raises(ValueError, match=message):
        _validate_oapl_config(config)
