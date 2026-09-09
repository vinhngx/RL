import torch

from nemo_rl.algorithms.advantage_estimator import RLZVPAdvantageEstimator
from nemo_rl.algorithms.grpo import _dataset_prompt_group_ids
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.rewards import boxed_numeric_proximity_reward


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


def test_vlm_group_identity_keeps_distinct_images_separate():
    batch = BatchedDataDict({"idx": [index for index in range(16) for _ in range(8)]})
    group_ids = _dataset_prompt_group_ids(batch, num_generations=8)
    assert group_ids.shape == (128, 1)
    unique, counts = torch.unique(group_ids, return_counts=True)
    assert unique.numel() == 16
    assert torch.all(counts == 8)


def test_dynamic_sampling_uses_full_group_variance():
    """A 7/1 group must retain all eight rollouts during DAPO filtering."""
    prompt_ids = torch.ones((8, 1), dtype=torch.long)
    rewards = torch.tensor([1.0] * 7 + [0.0])
    mask = torch.ones_like(rewards)
    _, full_std = calculate_baseline_and_std_per_prompt(
        prompt_ids, rewards, mask, leave_one_out_baseline=False
    )
    _, loo_std = calculate_baseline_and_std_per_prompt(
        prompt_ids, rewards, mask, leave_one_out_baseline=True
    )
    assert torch.count_nonzero(full_std).item() == 8
    assert torch.count_nonzero(loo_std).item() == 7


def test_boxed_numeric_proximity_is_distance_sensitive():
    assert boxed_numeric_proximity_reward("7", r"\\boxed{7}", decay=0.5) == (
        1.0,
        True,
    )
    assert boxed_numeric_proximity_reward("7", r"\\boxed{6}", decay=0.5) == (
        0.5,
        False,
    )
    assert boxed_numeric_proximity_reward("7", r"\\boxed{9}", decay=0.5) == (
        0.25,
        False,
    )
    assert boxed_numeric_proximity_reward("7", "answer: 7", decay=0.5) == (
        0.0,
        False,
    )


if __name__ == "__main__":
    test_rl_zvp_mixed_and_zero_variance_groups()
    test_vlm_group_identity_keeps_distinct_images_separate()
    test_dynamic_sampling_uses_full_group_variance()
    test_boxed_numeric_proximity_is_distance_sensitive()
    print("RL-ZVP advantage test passed")
