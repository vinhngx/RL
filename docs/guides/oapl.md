# OAPL: Off-Policy RL with Lagged Inference Policy

Reference: **"LLMs Can Learn to Reason Via Off-Policy RL"** (Ritter et al., arXiv:2602.19362).

NeMo RL's GRPO trainer derives on-policy behavior from importance-sampled ratios of
the trainer and inference-engine log-probabilities. **OAPL** (Optimal Advantage-based
Policy Optimization with Lagged Inference policy) instead embraces off-policyness:
it regresses the policy's log-probability shift onto the *optimal advantage* of the
KL-regularized RL problem, with no importance sampling.

For a prompt `x` with a group of `G` rollouts `{y_i}`, rewards `{r_i}`:

```
V̂*(x) = β1 · ln( mean_i exp(r_i / β1) )          (paper Eq. 2)
Â*(x, y_i) = r_i − V̂*(x)
L      =  mean_i ( β2 · ( log π(y_i|x) − log π_vllm(y_i|x) ) − Â*(x, y_i) )^2   (paper Eq. 3)
```

Unlike GRPO/DAPO, there is **no importance ratio and no clipping**. The minimizer
of the loss is the same regardless of how stale the sampling policy π_vllm is,
so OAPL can tolerate a large policy lag with substantially fewer refreshes of the
colocated inference engine. Practically, NeMo RL freezes the inference engine for
`sync_lag_interval` training steps (paper's `L`) between refits.

## Configuration

```yaml
oapl:
  enabled: true
  beta1: 1.0             # smoothing of V̂* (paper Eq. 2); paper: 1
  beta2: 1e-3            # log-ratio regression scale (paper Eq. 3); paper: 1e-3
  sync_lag_interval: 50  # steps between inference-engine refits; paper: 50

grpo:
  adv_estimator:
    name: "oapl"         # uses the OAPL optimal-advantage estimator
```

Notes:

- `oapl.enabled: true` swaps the loss to `OAPLLossFn` (see
  `nemo_rl/algorithms/loss/loss_functions.py`); `grpo.adv_estimator.name: oapl` is
  required so the `beta1` value is visible to the estimator.
- Paper best math setup: `beta1=1`, `beta2=1e-3`, `L=50` (Qwen3-4B-Thinking,
  Deepscaler train set). AdamW lr 1e-6, grad-norm clip 1e-3.
- Use a longer `max_total_sequence_length` (16k-32k) if mimicking the paper's
  thinking-style math runs.
- OAPL skips the reference-policy KL term of GRPO; set
  `grpo.skip_reference_policy_logprobs_calculation: true` to save logprob passes.
- Lag currently applies to the synchronous training path. The async engine
  (GrpoAsyncConfig) has its own staleness semantics and is left for follow-up.

See `examples/configs/grpo_oapl_qwen3_1p7b_dapo17k.yaml` for a runnable recipe.
