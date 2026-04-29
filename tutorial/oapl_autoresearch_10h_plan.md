# OAPL Auto-Research Plan: 10h Math Campaign

Date: 2026-04-29

## Goal

Find a practical path to high OAPL math accuracy in a 10-hour single-GPU campaign. The primary metric is validation accuracy on AIME-style math prompts, with final reporting on AIME-2024, AIME-2025, and MATH-500 once a checkpoint clears the smoke gate.

## Current Facts

- Hardware: one L40S-class GPU with 46 GB VRAM.
- Runtime supports `qwen2` and `qwen3`, but not `qwen3_5`.
- `Qwen/Qwen3.5-0.8B-Base` is blocked by `model_type: qwen3_5` in the pinned Transformers/vLLM stack.
- `Qwen/Qwen3-0.6B` OAPL smoke trained and checkpointed, but AIME reward stayed all-zero.
- `Qwen/Qwen2.5-Math-1.5B-Instruct` is supported and scored `3/30 = 0.1000` pass@1 on AIME-2024 with temperature 0 and 2048 tokens.

The key decision is to spend the 10h budget on a model with nonzero reward signal instead of trying to force sparse-RL learning from a tiny base model.

## Campaign Setup

Shared branch prefix from the auto-research skill:

```text
autoresearch/2026-04-29-oapl-math
```

Ledger:

```text
reports/oapl_autoresearch_20260429.tsv
```

Baseline result already recorded:

```text
Qwen/Qwen2.5-Math-1.5B-Instruct, AIME-2024 pass@1 = 0.1000
```

Primary training config:

```text
examples/configs/oapl_qwen25_math_1p5b_dapo_10h.yaml
```

Budget:

- Campaign deadline: 10 hours from launch.
- Per-experiment cap: 2 hours unless the run is clearly improving.
- Stop early only for a clear blocker, disk exhaustion risk, or a checkpoint that materially exceeds baseline and needs final eval.

## Why This Recipe

The OAPL estimator needs prompt groups with reward diversity. A base model that gets `0/30` on AIME produced no learning signal in smoke. The 1.5B math instruct model is still small enough for one GPU, but strong enough to generate occasional correct samples. With `num_prompts_per_step=128` and `num_generations_per_prompt=8`, each OAPL step uses GBS 1024 while keeping the group size aligned to the optimal-value estimator.

The first recipe uses:

- `beta_v=0.5` to sharpen grouped value targets on binary rewards.
- `beta_loss=1.0e-3` to keep sequence-level log-ratio regression numerically gentle.
- `sync_interval_steps=10` so the lagged generation policy gives OAPL a real off-policy target while still refreshing often enough for a short campaign.
- `lr=5.0e-7` because this starts from an instruct model and OAPL has no fixed reference-policy KL.
- `max_total_sequence_length=2048` to preserve throughput; a 4096-token variant is staged if truncation appears to be limiting.

## Experiment Order

1. Baseline eval: done, `0.1000` AIME-2024 pass@1.
2. Health run: launch `oapl_qwen25_math_1p5b_dapo_10h.yaml` and inspect the first validation at step 10.
3. Continue base recipe if train reward, OAPL residuals, and validation are finite and nonzero.
4. If validation is flat but train rewards are nonzero, test `oapl_qwen25_math_1p5b_dapo_10h_lr1e6.yaml`.
5. If group reward variance is weak, test `oapl_qwen25_math_1p5b_dapo_10h_g64x16.yaml`.
6. If many generations truncate or final answers are cut off, test `oapl_qwen25_math_1p5b_dapo_10h_len4096.yaml`.
7. Keep the best checkpoint by `val:accuracy`, then run final eval with `examples/run_eval.py` on AIME-2024, AIME-2025, and MATH-500.

## Launch Command

Use isolated caches so root disk does not fill:

```bash
env -u RAY_ADDRESS \
  HF_HOME=/ephemeral/oapl_autoresearch_hf_cache \
  TRANSFORMERS_CACHE=/ephemeral/oapl_autoresearch_hf_cache/hub \
  HF_DATASETS_CACHE=/ephemeral/oapl_autoresearch_hf_cache/datasets \
  XDG_CACHE_HOME=/ephemeral/oapl_autoresearch_cache \
  UV_CACHE_DIR=/ephemeral/oapl_autoresearch_uv_cache \
  RAY_TMPDIR=/ephemeral/oapl_autoresearch_raytmp \
  RAY_TEMP_DIR=/ephemeral/oapl_autoresearch_raytmp \
  TMPDIR=/ephemeral/oapl_autoresearch_raytmp \
  TORCHINDUCTOR_CACHE_DIR=/ephemeral/oapl_autoresearch_raytmp/torchinductor \
  uv run examples/run_grpo.py \
    --config examples/configs/oapl_qwen25_math_1p5b_dapo_10h.yaml \
    > /ephemeral/nemo-rl/oapl_qwen25_math_1p5b_dapo_10h.log 2>&1
```

## Acceptance Criteria

Sound prototype:

- No NaNs in loss or gradients.
- Nonzero valid tokens.
- Nonzero train reward on at least some steps.
- OAPL metrics logged, including sequence log-ratio and policy lag.
- Checkpoint saved before timeout.

Promising accuracy:

- AIME-2024 validation beats `0.1000` baseline during the 10h campaign.
- Prefer checkpoints that improve without major response-length blowup.
- Confirm with final `run_eval.py`, not only training validation.

High-confidence conclusion:

- Same eval settings for base and trained checkpoint.
- Report pass@1 on AIME-2024, AIME-2025, and MATH-500.
- If possible, add pass@k with `eval.num_tests_per_prompt=16`.

## Main Risks

- Disk: `/ephemeral` was already tight after smoke runs. Keep only one top checkpoint and avoid consolidated saves during search.
- Sparse reward: if the instruct model still gets too few positives under temperature 1 rollouts, shift to `64x16` groups before changing optimizer settings.
- Drift: OAPL has no reference-policy KL by design. If validation drops while train reward rises, lower LR or shorten sync interval.
- Throughput: 4096-token runs may spend too much of the 10h budget on generation. Only use them if truncation is visible in logs.
