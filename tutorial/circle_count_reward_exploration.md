# Circle Count Reward Exploration

## Goal

The current Circle Count dense reward gives partial credit by absolute count error:

```text
reward = exp(-abs(predicted_count - expected_count) / temperature)
```

That helped expose near-miss signal, but it treats every off-by-one equally. An answer of `8` for target `9` receives the same reward as `0` for target `1`, even though the former is much closer as a fraction of the target.

## New Variant

I added a configurable dense reward mode:

- `absolute_exp`: existing behavior, still the default.
- `relative_exp`: percent-off-target behavior.

The relative mode is:

```text
relative_error = abs(predicted_count - expected_count) / max(expected_count, 1)
reward = exp(-relative_error / temperature)
```

The `max(expected_count, 1)` denominator avoids division by zero. For zero-count targets, predicting one circle is treated as a full one-count error.

## Reward Shape At Temperature 1.0

| Expected | Prediction | Absolute Reward | Relative Reward |
| ---: | ---: | ---: | ---: |
| 1 | 0 or 2 | 0.368 | 0.368 |
| 2 | 1 or 3 | 0.368 | 0.607 |
| 5 | 4 or 6 | 0.368 | 0.819 |
| 9 | 8 | 0.368 | 0.895 |

Full table: `tutorial/circle_count_reward_modes_table.csv`

## Candidate Run

Config prepared:

```text
tutorial/circle_count_qwen3_vl_2b_from_step35_relative_dense_temp1_seed14141_lr1e9.yaml
```

This starts from the same config family as the tied-best dense run and switches only the reward mode to `relative_exp`, with validation every step and printed validation samples enabled.

## Hypothesis

The relative reward may help if the current plateau is caused by high-count near misses being punished too harshly. It may hurt if the model learns to over-count on high-count examples because off-by-one and off-by-two errors become too forgiving.

The useful comparison is therefore exact validation accuracy, not dense reward. A successful run must beat the current best exact score of **389/512 = 75.98%**.

## Verification Notes

Static syntax check passed:

```text
python3 -m py_compile 3rdparty/Gym-workspace/Gym/resources_servers/circle_count/app.py 3rdparty/Gym-workspace/Gym/resources_servers/circle_count/tests/test_app.py
```

Full pytest did not run in the sandbox because `pytest` is not installed in the base shell, and `uv run pytest` attempted to resolve build metadata from PyPI while network access is restricted.

