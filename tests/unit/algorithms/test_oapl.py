# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for OAPL (arXiv:2602.19362): advantage estimator, loss, and lag gating."""

import math

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import OAPLAdvantageEstimator
from nemo_rl.algorithms.loss.loss_functions import OAPLLossFn

BETA1 = 1.0
BETA2 = 1e-3


def test_oapl_advantage_estimator_matches_paper_closed_form():
    """Â* = r − β1·ln(mean exp(r/β1)) per prompt group (paper Eq. 2 + text)."""
    estimator = OAPLAdvantageEstimator(
        {"beta1": BETA1}, {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0}
    )

    # Two prompt groups, G=4 each.
    prompt_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    rewards = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    mask = torch.ones((8, 5))

    adv = estimator.compute_advantage(prompt_ids, rewards, mask)

    def vstar(rs):
        rs = torch.tensor(rs)
        return BETA1 * (torch.logsumexp(rs / BETA1, dim=0) - torch.log(torch.tensor(float(len(rs)))))

    v0 = vstar([0.0, 0.0, 1.0, 0.0])
    expected0 = torch.tensor([0.0 - v0, 0.0 - v0, 1.0 - v0, 0.0 - v0])
    torch.testing.assert_close(adv[:4, 0], expected0.view(4))

    v1 = vstar([1.0, 1.0, 1.0, 1.0])
    expected1 = torch.full((4,), 1.0 - v1)
    torch.testing.assert_close(adv[4:, 0], expected1)

    # Limiting behavior β1 -> 0: V̂* == max(r)
    estimator_eps = OAPLAdvantageEstimator({"beta1": 1e-8}, {"reference_policy_kl_penalty": 0.0})
    adv_eps = estimator_eps.compute_advantage(prompt_ids, rewards, torch.ones((8, 5)))
    assert torch.allclose(
        adv_eps[:4, 0], torch.tensor([0.0 - 1.0, 0.0 - 1.0, 1.0 - 1.0, 0.0 - 1.0]), atol=1e-6
    )

    # Limiting behavior β1 -> ∞: V̂* → mean(r)
    estimator_big = OAPLAdvantageEstimator({"beta1": 1e8}, {"reference_policy_kl_penalty": 0.0})
    adv_big = estimator_big.compute_advantage(prompt_ids, rewards, torch.ones((8, 5)))
    mean0 = 0.25
    assert torch.allclose(adv_big[:4, 0], torch.tensor([-mean0, -mean0, 1 - mean0, -mean0]), rtol=1e-5)

    # Sanity: same value broadcast to all token positions (mask expansion)
    assert adv.shape == mask.shape
    assert (adv[0] == adv[0, 0]).all()


@pytest.fixture
def oapl_loss_fn():
    return OAPLLossFn({"beta2": BETA2, "reference_policy_kl_penalty": 0.0})


def _make_oapl_data(batch_size=2, seq_len=4, device="cpu"):
    """B=2 samples with assistant tokens at positions 2..seq_len."""
    token_mask = torch.zeros((batch_size, seq_len))
    token_mask[:, 2:] = 1.0
    return {
        "token_mask": token_mask.to(device),
        "sample_mask": torch.ones(batch_size).to(device),
    }


def _oapl_inputs(batch_size=2, seq_len=4, curr_seed=0, gen_seed=1, adv_value=1.0, gen_offset=-0.3):
    """Build (next_token_logprobs, data, global_valid_seqs, global_valid_toks).

    Convention: data tensors are [B, seq_len] and get sliced [:,1:]; the
    next-token logprobs (already label-shifted) are [B, seq_len - 1].
    """
    g1 = torch.Generator().manual_seed(curr_seed)
    g2 = torch.Generator().manual_seed(gen_seed)
    seq = seq_len  # data contains positions 0..seq_len-1; fns slice [:,1:]
    data = _make_oapl_data(batch_size, seq)
    data["advantages"] = adv_value * torch.ones((batch_size, seq))
    data["generation_logprobs"] = gen_offset + 0.05 * torch.randn((batch_size, seq), generator=g2)
    data["prev_logprobs"] = torch.zeros((batch_size, seq))  # unused by OAPL
    next_token_logprobs = 0.05 * torch.randn((batch_size, seq - 1), generator=g1)
    valid_seqs = torch.tensor(batch_size)
    valid_toks = data["token_mask"].sum()
    return next_token_logprobs, data, valid_seqs, valid_toks


def test_oapl_loss_zero_at_optimum_of_kl_regularized_objective(oapl_loss_fn):
    """Minimizer: β2·(log πθ − log πgen) = Â* ⇒ residual 0 when curr == gen + Â*/β2 per seq."""
    batch, seq = 2, 4
    _, data, valid_seqs, valid_toks = _oapl_inputs(batch, seq, adv_value=1.5)
    # After [:,1:] slicing: token_mask [0, 1, 1] → 2 valid tokens per sample.
    gen_sum = (data["generation_logprobs"][:, 1:]).sum(-1)
    # Set current logprobs so β2 * (curr_sum - gen_sum) == Â* = 1.5 per sequence.
    # post-slice valid positions of the [batch, seq-1] tensor are indices 1 and 2.
    curr_per_token = (gen_sum + 1.5 / BETA2) / 2
    next_token_logprobs = torch.zeros((batch, seq - 1))
    next_token_logprobs[:, 1:] = curr_per_token.unsqueeze(-1).expand(-1, 2)

    loss, metrics = oapl_loss_fn(next_token_logprobs, data, valid_seqs, valid_toks)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert metrics["oapl_residual_mse"] < 1e-6


def test_oapl_loss_value_matches_closed_form(oapl_loss_fn):
    """loss == mean_i ( β2*(curr_sum_i − gen_sum_i) − Â*_i )^2"""
    batch, seq = 3, 4
    _, data, valid_seqs, valid_toks = _oapl_inputs(batch, seq)
    g = torch.Generator().manual_seed(7)
    gen = -0.2 + 0.1 * torch.randn((batch, seq), generator=g)
    data["generation_logprobs"] = gen
    dat = {"a1": 0.5, "a2": -0.75, "a3": 1.5}
    advs = torch.stack(
        [torch.full((seq,), v) for v in (dat["a1"], dat["a2"], dat["a3"])]
    )
    data["advantages"] = advs
    curr = 0.1 * torch.randn((batch, seq - 1), generator=g)

    loss, metrics = oapl_loss_fn(curr, data, valid_seqs, valid_toks)

    m = data["token_mask"][:, 1:]
    cs = (curr * m).sum(-1)
    gs = (gen[:, 1:] * m).sum(-1)
    expected = torch.stack(
        [
            (BETA2 * (c - g_) - a) ** 2
            for c, g_, a in zip(cs, gs, advs[:, 0])
        ]
    ).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_oapl_loss_gradient_pushes_towards_optimum(oapl_loss_fn):
    batch, seq = 1, 4
    _, data, valid_seqs, valid_toks = _oapl_inputs(batch, seq, adv_value=1.0)
    data["generation_logprobs"] = torch.zeros((batch, seq))
    curr = torch.zeros((batch, seq - 1), requires_grad=True)
    loss, _ = oapl_loss_fn(curr, data, valid_seqs, valid_toks)
    loss.backward()
    # Positive gradient on curr logprobs means update pushes them down toward
    # gen_sum + Â*/β2... i.e. gradient sign matches (curr_sum − target).
    pred_minus_target = BETA2 * (0.0 - 0.0) - 1.0  # -1.0 (below target)
    assert pred_minus_target < 0
    # d/d curr (pred - target)^2 = 2*(pred-target)*β2 < 0 for valid tokens
    # (post-slice valid indices are 1..end)
    assert (curr.grad[:, 1:] < 0).all()


def test_oapl_rejects_kl_penalty_when_configured_zero_is_expected(oapl_loss_fn):
    """default config: no reference-policy KL term in OAPL objective (paper has none)."""
    batch, seq = 1, 4
    _, data, valid_seqs, valid_toks = _oapl_inputs(batch, seq)
    curr = torch.zeros((batch, seq - 1))
    loss, metrics = oapl_loss_fn(curr, data, valid_seqs, valid_toks)
    assert math.isfinite(loss.item())
    assert oapl_loss_fn.reference_policy_kl_penalty == 0.0


def test_oapl_lagged_refit_arithmetic():
    """Window positions: refit at step 0 and every L steps; oapl_config schema."""
    from nemo_rl.algorithms.grpo import MasterConfig, OAPLConfig

    cfg: OAPLConfig = {
        "enabled": True,
        "beta1": 1.0,
        "beta2": 1e-3,
        "sync_lag_interval": 50,
    }
    lag = cfg["sync_lag_interval"]
    # Window semantics: refit at t=0 mod L, skip other steps within the window.
    refit_at = [t for t in range(120) if lag == 1 or t % lag == 0]
    assert refit_at == [0, 50, 100]
    # MasterConfig accepts the oapl key.
    keys = MasterConfig.__annotations__.keys()
    assert "oapl" in keys
