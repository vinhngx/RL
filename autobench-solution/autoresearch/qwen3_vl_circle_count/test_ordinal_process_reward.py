#!/usr/bin/env python3
"""Focused checks for ordinal Circle Count data and reward behavior."""

from __future__ import annotations

import json

from generate_ordinal_process_count import make_ordinal_example
from nemo_rl.environments.rewards import ordinal_count_reward


def test_generator_builds_monotone_threshold_targets() -> None:
    row = {
        "target_color": "red",
        "circles": [
            *({"color": "red"} for _ in range(7)),
            *({"color": "blue"} for _ in range(3)),
        ],
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,x"},
                        {"type": "input_text", "text": "old"},
                    ],
                },
            ]
        },
    }
    output = make_ordinal_example(row)
    expected = json.loads(output["process_ground_truth"])
    assert expected == {
        "kind": "ordinal_count",
        "ge5": 1,
        "ge6": 1,
        "ge7": 1,
        "ge8": 0,
        "ge9": 0,
        "ge10": 0,
        "total": 7,
    }
    assert "GE5=" in output["responses_create_params"]["input"][1]["content"][1]["text"]


def test_reward_is_dense_and_requires_box_for_exactness() -> None:
    truth = json.dumps(
        {
            "kind": "ordinal_count",
            "ge5": 1,
            "ge6": 1,
            "ge7": 1,
            "ge8": 0,
            "ge9": 0,
            "ge10": 0,
            "total": 7,
        }
    )
    assert ordinal_count_reward(
        truth, r"GE5=1 GE6=1 GE7=1 GE8=0 GE9=0 GE10=0 \boxed{7}"
    ) == (1.0, True)
    partial_score, partial_exact = ordinal_count_reward(
        truth, r"GE5=1 GE6=1 GE7=0 GE8=0 GE9=0 GE10=0 \boxed{6}"
    )
    assert partial_score == 5 / 12
    assert partial_exact is False
    assert ordinal_count_reward(truth, "7") == (0.0, False)


def test_reward_falls_back_for_canonical_validation_rows() -> None:
    assert ordinal_count_reward("7", r"The answer is \boxed{7}.") == (1.0, True)
    assert ordinal_count_reward("7", r"The answer is \boxed{6}.") == (0.0, False)
