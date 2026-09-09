import torch

from nemo_rl.algorithms.advantage_estimator import RLZVPAdvantageEstimator


def test_rl_zvp_mixed_and_zero_variance_groups():
    estimator = RLZVPAdvantageEstimator(
        {
            "normalize_rewards": True,
            "use_leave_one_out_baseline": True,
            "alpha": 0.1,
            "entropy_top_k": 4,
        },
        {},
    )
    prompt_ids = torch.tensor([[1], [1], [1], [1], [2], [2], [3], [3]])
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    mask = torch.tensor([[1.0, 1.0]] * 8)
    entropy = torch.tensor(
        [
            [0.2, 0.6],
            [0.3, 0.5],
            [0.4, 0.7],
            [0.2, 0.9],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.5, 0.7],
        ]
    )

    advantage = estimator.compute_advantage(
        prompt_ids, rewards, mask, token_entropies=entropy
    )

    assert torch.all(advantage[0] < 0) and torch.all(advantage[1] > 0)
    torch.testing.assert_close(advantage[4], 0.1 * entropy[4])
    torch.testing.assert_close(advantage[5], 0.1 * entropy[5])
    torch.testing.assert_close(advantage[6], torch.tensor([-0.06, 0.0]))
    torch.testing.assert_close(advantage[7], torch.tensor([-0.02, 0.0]))


if __name__ == "__main__":
    test_rl_zvp_mixed_and_zero_variance_groups()
    print("RL-ZVP advantage test passed")
