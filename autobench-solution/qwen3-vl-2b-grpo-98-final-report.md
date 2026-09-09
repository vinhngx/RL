# Qwen3-VL-2B Circle-Count Optimization: Final Report

Date: 2026-09-09
Status: stopped at user request; no training or evaluation process remains

## Executive result

The campaign improved the reproduced SFT checkpoint from **184/200 (92.0%)**
to a best verified **190/200 (95.0%)** strict boxed accuracy on the frozen
200-example development set. All 200 leader responses contained a parseable
`\boxed{N}` answer. The requested **>98%** target (at least 197/200) was not
reached, so the sealed blind final set was never opened.

This remains a single-model VLM solution: a raw task image and text question
are passed directly to `Qwen/Qwen3-VL-2B-Instruct`. There is no external object
detector, image component counter, crop ensemble, vote, or answer aggregator at
inference time.

| Checkpoint | Strict correct | Accuracy | Box compliance | Decision |
|---|---:|---:|---:|---|
| Reproduced merged SFT step 150 | 184/200 | 92.0% | 200/200 | baseline |
| Early exact dynamic GRPO leader | 186/200 | 93.0% | 200/200 | superseded |
| Direction-matched DPO leader | 187/200 | 93.5% | 200/200 | superseded |
| Natural-DPO scaled leader | 188/200 | 94.0% | 200/200 | superseded |
| Prompt-matched edge-aware GRPO | 189/200 | 94.5% | 200/200 | superseded |
| **Counterfactual-pair GRPO step 2** | **190/200** | **95.0%** | **200/200** | **retain** |

The 95% estimate has a Wilson 95% interval of **91.04% to 97.26%**. It is a
six-example absolute gain, or **+3.0 percentage points**, over the SFT baseline.

## Evaluation contract

- Data: `/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260904-qwen3-vl-2b-gym6k-reference/eval-gym-profile-200/profile-seed10000.jsonl`
- Evaluator: `autobench-solution/autoresearch/qwen3_vl_circle_count/evaluate_hf_lora.py`
- Prompt: `autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_gym_boxed.txt`
- Generation: greedy (`temperature=0`), seed 42, batch size 4, maximum 128 new tokens, prompt role `system`
- Metric: final boxed integer exactly equals the generated task ground truth
- Summary: `/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/evals/step-2-canonical.summary.json`

## Retained recipe

The retained checkpoint is a merged step-2 vision adapter at:

`/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/counterfactual-pair-grpo/merged-step2`

Its final improvement came from exact GRPO on 4,096 synthetic
counterfactual pairs (8,192 images). Each pair preserves geometry and changes
exactly one distractor circle to the requested color, producing target counts
`N` and `N+1`. This retained run used the ordinary exact boxed reward; the
paired construction supplied a tightly controlled adjacent-count curriculum.

Key settings:

- NeMo-RL v0.6.0, DTensor backend, BF16, one L40S GPU
- Qwen3-VL-2B base with a previously promoted 94.5% campaign parent
- LoRA rank 8, alpha 16, vision-tower linear layers only
- AdamW, constant learning rate `2.5e-7`, no weight decay
- 16 prompts per update, 8 generations per prompt, global batch 128
- training microbatch 1 with token-dynamic batching (`2048` train tokens)
- rollout temperature 1.0, maximum 64 new tokens
- exact boxed reward, GRPO leave-one-out normalized advantages
- KL coefficient 0.1, token-level clipped objective, six planned steps
- step 2 selected by the canonical deterministic evaluation

The implementation and launch entry points are:

- `grpo_counterfactual_pairs_from_945_v06.yaml`
- `generate_counterfactual_circle_pairs.py`
- `run_counterfactual_pair_grpo_v06_docker.sh`
- `run_counterfactual_pair_eval_docker.sh`

All are under `autobench-solution/autoresearch/qwen3_vl_circle_count/`.

To rerun the canonical checkpoint evaluation from `/home/ubuntu/RL`:

```bash
STEPS=2 bash autobench-solution/autoresearch/qwen3_vl_circle_count/run_counterfactual_pair_eval_docker.sh
```

## Residual errors

The leader's ten errors are development indices:

`3, 8, 10, 37, 63, 93, 105, 109, 139, 195`

Every error is off by exactly one: seven undercounts and three overcounts.
There is no remaining answer-format failure. Accuracy is perfect through
expected count 4, while most residual risk occurs at counts 5 through 9; for
example, count 8 is 12/15 and count 9 is 4/6. Yellow is 29/29; the weaker color
slices are still small and do not support a reliable color-specific rule.

A teacher-forced diagnostic found that all ten misses locally preferred the
wrong adjacent numeric answer. On 256 disjoint same-geometry `N`/`N+1` pairs,
the added circle shifted the adjacent answer margin in the correct direction
for 255/256 examples. This suggests the final ceiling is principally narrow
decoder/decision-boundary calibration, not missing boxed formatting or a total
lack of visual sensitivity.

## Important framework findings

Two NeMo-RL v0.6 VLM-specific issues were found and regression-tested late in
the campaign:

1. GRPO grouped prompts by text-token identity, which conflated different
   images having the same color question. The correction uses repeated dataset
   example IDs at both baseline and final-advantage sites and was verified in a
   real run as 16 image groups x 8 rollouts.
2. Dynamic-sampling selection combined with leave-one-out standard deviation
   could retain a partial 7-of-8 group. Selection now uses full-group variance,
   while the actual advantage estimator retains leave-one-out behavior.

Corrected follow-up runs preserved 200/200 boxed answers but scored 188/200 and
187/200, respectively, so neither replaced the leader. These corrections
should nevertheless be retained for future VLM GRPO work.

Across the campaign, longer training, larger rank, hard-negative-only replay,
structured quadrant/location rewards, higher image resolution, DPO/SimPO
calibration, vision-only localization auxiliaries, adapter interpolation, and
zero-variance entropy shaping all failed to exceed 190/200. The common failure
mode was fixing zero or a few residual mistakes while breaking an equal or
larger number of previously correct borderline examples.

## Final stopped run

The last proposed experiment used a distance-sensitive boxed reward
(`exact=1`, off-by-one `0.5`, off-by-two `0.25`) with both grouping fixes. It
was committed as `179676ce4` and logged to W&B run `ecwnowgm`. At the user's
request it was stopped during initial validation, before validation completed,
before any optimizer step, and before any checkpoint was saved. It therefore
has **no accuracy result** and produced no candidate model state.

After stopping, both Docker process listing and `nvidia-smi` showed no active
campaign process. Large assets remain on `/data`; `/ephemeral` resolves to
`/data/ephemeral`.

## Reproducibility and limitations

- Work branch at report time:
  `autoresearch/2026-09-09-qwen3-vl-2b-grpo-98/vlm-proximity-dynamic-from-950`
- Counterfactual experiment implementation commit: `d599a0386`
- Corrected prompt/reward continuation commits are present in the retained
  `counterfactual-pair-grpo` branch history.
- Final stopped-experiment commit: `179676ce4`
- W&B retained counterfactual run: `kx8mpog0`
- Container: `nvcr.io/nvidia/nemo-rl:v0.6.0`
- Runtime worktree:
  `/data/ephemeral/nemo-rl/ubuntu/reference-rl-v060-corrected`

The 200-example set was used repeatedly for model selection, so 95% is a
development result rather than an unbiased final estimate. The blind final was
intentionally kept sealed because the predeclared 197/200 promotion threshold
was never met. No secret values are included in this report.

## Post-report fresh final evaluation

After this campaign report was closed, the user requested a new 1,000-example
final comparison across all promoted milestones. On frozen seeds
30,000,000--30,000,999, counterfactual-pair GRPO and its edge-GRPO parent tied
at **950/1000 (95.0%)**, with 1000/1000 boxed answers. Full methodology,
checkpoint ranking, paired statistics, and artifact paths are in
`autobench-solution/qwen3-vl-2b-fresh-final-1k-report.md`. This final set is now
spent and must not be used for further tuning.
