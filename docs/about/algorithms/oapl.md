# OAPL

Optimal Advantage-based Policy Learning (OAPL) is an off-policy RL algorithm for training from sampled completions. In NeMo RL, OAPL reuses the GRPO training loop, data pipeline, reward computation, rollout generation, and distributed policy workers, but swaps in an OAPL-specific advantage estimator and sequence-level regression loss.

Use OAPL when you want to train against rewards from a generation policy that is synchronized with the learner periodically instead of requiring every optimization step to be fully on-policy.

## How OAPL Works

For each prompt, OAPL samples multiple responses from the current generation policy. Those responses are scored by the configured environment or reward function, just as in GRPO.

OAPL then computes an optimal value estimate per prompt group:

```text
V*(x) = beta_v * log mean_i exp(r_i / beta_v)
```

where `x` is the prompt, `r_i` is the reward for response `i`, and `beta_v` controls the softness of the value estimate.

Each sampled response receives a sequence-level target advantage:

```text
A*(x, y_i) = r_i - V*(x)
```

The policy is trained to make its log probability ratio against the generation policy match this target:

```text
loss = (beta_loss * log(pi_theta(y|x) / pi_gen(y|x)) - A*(x, y))^2
```

The log ratio is reduced across response tokens at sequence level. OAPL v1 requires `log_ratio_reduction: "sum"`.

## Key Differences from GRPO

- OAPL uses the optimal-advantage target `A*(x, y)` instead of GRPO's normalized group-relative advantage.
- OAPL uses a squared sequence-level regression loss instead of clipped policy-gradient loss.
- OAPL compares the training policy to the generation policy that produced the samples, so the generation log probabilities are required.
- OAPL periodically refits/synchronizes the generation policy from the training policy according to `grpo.oapl.sync_interval_steps`.
- OAPL does not use a reference-policy KL penalty or importance-sampling correction in the current implementation.

## OAPL Single Node

A minimal single-node run can use the OAPL math config:

```sh
uv run python examples/run_grpo.py \
  --config examples/configs/oapl_math_1B.yaml
```

You can override the model, checkpoint path, and logger options in the same way as GRPO:

```sh
uv run python examples/run_grpo.py \
  --config examples/configs/oapl_math_1B.yaml \
  policy.model_name="Qwen/Qwen2.5-1.5B" \
  checkpointing.checkpoint_dir="results/oapl_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="oapl-math"
```

For a quick smoke test, use a small model config if present in your checkout:

```sh
uv run python examples/run_grpo.py \
  --config examples/configs/oapl_math_0p5b_smoke.yaml
```

## OAPL Multi-node

OAPL uses the same launch pattern as GRPO. For example:

```sh
# Run from the root of the NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_grpo.py \
  --config examples/configs/oapl_math_1B.yaml \
  cluster.num_nodes=2 \
  checkpointing.checkpoint_dir='results/oapl_2nodes' \
  logger.wandb_enabled=True \
  logger.wandb.name='oapl-multinode'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

> [!NOTE]
> For GB200 systems with 4 GPUs per node, use `--gres=gpu:4` instead.

## Configuration

Enable OAPL by setting `grpo.algorithm: "oapl"` and providing the OAPL block under `grpo.oapl`:

```yaml
grpo:
  algorithm: "oapl"
  skip_reference_policy_logprobs_calculation: true
  seq_logprob_error_threshold: null
  use_dynamic_sampling: false
  val_period: 50
  oapl:
    beta_v: 1.0
    beta_loss: 1.0e-3
    sync_interval_steps: 50
    log_ratio_reduction: "sum"

loss_fn:
  reference_policy_kl_penalty: 0.0
  use_importance_sampling_correction: false
  truncated_importance_sampling_ratio: null
  truncated_importance_sampling_ratio_min: null
```

### OAPL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grpo.algorithm` | `"grpo"` | Set to `"oapl"` to select the OAPL advantage estimator and loss. |
| `grpo.oapl.beta_v` | `1.0` | Temperature for the prompt-level soft value estimate `V*(x)`. Smaller values make the value estimate closer to the best reward in the group; larger values make it closer to an average-reward baseline. Must be positive. |
| `grpo.oapl.beta_loss` | `1.0e-3` | Scale applied to the sequence log probability ratio in the regression loss. Must be positive. |
| `grpo.oapl.sync_interval_steps` | `50` | Number of learner steps between generation-policy refits. Larger values reduce refit overhead but increase policy lag. Must be positive. |
| `grpo.oapl.log_ratio_reduction` | `"sum"` | How token log ratios are reduced to a sequence log ratio. OAPL v1 requires `"sum"`. |

### Required Compatibility Settings

The current OAPL implementation validates these settings at startup:

| Setting | Required value | Why |
|---------|----------------|-----|
| `grpo.use_dynamic_sampling` | `false` | OAPL v1 expects fixed prompt groups for estimating `V*(x)`. |
| `grpo.skip_reference_policy_logprobs_calculation` | `true` | OAPL does not use reference-policy log probabilities. |
| `grpo.seq_logprob_error_threshold` | `null` | Sequence logprob error masking is not supported for OAPL v1. |
| `loss_fn.reference_policy_kl_penalty` | `0.0` | OAPL does not use a reference-policy KL penalty. |
| `loss_fn.use_importance_sampling_correction` | `false` | OAPL's loss is already defined against the generation-policy log ratio. |
| `grpo.val_period` | `0` or a multiple of `grpo.oapl.sync_interval_steps` | Validation is restricted to generation-policy sync boundaries. |

## Metrics

OAPL logs the standard train-vs-generation diagnostics used by the GRPO training path, plus OAPL-specific metrics:

| Metric | Meaning |
|--------|---------|
| `oapl/residual_mean` | Mean residual of `beta_loss * log_ratio - A*` over valid samples. |
| `oapl/residual_std` | Standard deviation of the OAPL residual. |
| `oapl/seq_log_ratio_mean` | Mean sequence log probability ratio between the training policy and generation policy. |
| `oapl/target_mean` | Mean OAPL target advantage `A*`. |
| `oapl/policy_lag_steps` | Number of learner steps since the generation policy was last synchronized. |
| `token_mult_prob_error` | Token-level multiplicative probability mismatch between training and generation policies. |
| `gen_kl_error` | Approximate KL from generation policy to training policy. |
| `policy_kl_error` | Approximate KL from training policy to generation policy. |
| `js_divergence_error` | Jensen-Shannon style divergence diagnostic between the two policies. |

## Practical Guidance

Start with `beta_v: 1.0`, `beta_loss: 1.0e-3`, and `sync_interval_steps: 50`. If `oapl/policy_lag_steps`, KL diagnostics, or probability-error metrics grow quickly, reduce `sync_interval_steps` so generation is refit more often. If updates are too small or too large, tune `beta_loss` together with the learning rate because both affect the strength of the log-ratio regression update.

For more details on shared rollout, data, backend, and cluster options, see the [GRPO documentation](grpo.md).
