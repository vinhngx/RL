# NeMo RL Accuracy Search

This repo is for running NeMo RL experiments inside a container that is already set up and ready to run.

The agent should focus on improving accuracy, not just making the first config work. If the container is healthy, start experimenting immediately and try multiple strategies until the best result is found.

## Goal

Reach the highest validation or task accuracy possible for the chosen recipe, while keeping the run stable and reproducible.

Accuracy may come from:
- better hyperparameters
- a better backend or parallelism strategy
- a different precision mode
- a more suitable sequence length or batch layout
- a different recipe or dataset mix
- small targeted code fixes that improve correctness or stability

## Operating Rules

1. Treat the container as ready to use.
2. Prefer existing configs and recipes in `examples/configs/` and `examples/configs/recipes/`.
3. Do not stop after the first successful run. Try alternative strategies and compare results.
4. Keep changes small and reversible unless a larger refactor is clearly justified by a gain in accuracy.
5. If a run crashes, fix the issue if it is straightforward. If the idea is fundamentally poor, discard it and move on.
6. Preserve unrelated user work.

## Git Persistence

Treat git as the experiment ledger.

- Before starting an experiment, capture the current state with a git commit so there is always a rollback point.
- Keep one commit per experiment or hypothesis; do not mix unrelated ideas into the same commit.
- If an experiment needs a follow-up fix, create a new commit for the fix rather than rewriting the previous experiment.
- Keep successful experiments committed even if later experiments build on top of them.

## What To Inspect First

Before changing anything, read the relevant recipe and its parents:
- the target config under `examples/configs/recipes/`
- the base config it inherits from
- any backend-specific code paths that the recipe depends on
- the relevant docs in `docs/` if the strategy touches a specialized feature

For model and training work, also check:
- `nemo_rl/models/`
- `nemo_rl/algorithms/`
- `examples/run_grpo.py`

## Strategy Search

When a container is ready, an agent should try different strategies rather than committing to the first working setup.

Good search axes include:
- `fp16`, `bf16`, or other precision settings
- Megatron vs DTensor vs automodel paths
- tensor, pipeline, and context parallel sizes
- batch size and gradient accumulation
- sequence length and packing settings
- vLLM generation settings, including attention backend compatibility
- optimizer and scheduler settings
- reward scaling, reward shaping, and clipping settings
- dataset choice or validation split changes
- small correctness fixes in code paths that affect generation or evaluation

When comparing strategies, prefer changes that:
- improve accuracy directly
- keep the run stable
- preserve or reduce complexity
- avoid fragile, hardware-specific hacks unless they are necessary

## Experiment Loop

For each experiment:

1. Pick one concrete hypothesis.
2. Edit the smallest set of files needed.
3. Run the experiment in the container.
4. Inspect the exact metric the recipe reports.
5. Record the result.
6. Keep the change if it improves accuracy or enables a better next experiment.
7. Revert it if it is worse and not useful for future tests.

If several promising ideas exist, prioritize the ones most likely to affect accuracy quickly:
- backend compatibility fixes
- precision or stability fixes
- batch and sequence layout changes
- optimizer tuning
- recipe-specific settings

## Measuring Results

Use the metric reported by the run as the source of truth.

Typical commands:

```bash
grep "^val_accuracy:\|^accuracy:\|^val_bpb:\|^peak_vram_mb:" run.log
```

If the exact metric name differs for a recipe, use the one that recipe prints at the end.

## Logging

Keep a simple experiment log, either in a TSV or in your own notes, with:
- commit hash
- recipe or config used
- main change tried
- metric value
- memory or stability notes
- keep/discard/crash status

Do not overwrite the user’s files unless the task explicitly asks for it.

## Practical Guidance

- Prefer using `uv run` with the repo’s existing entrypoints.
- Keep commands reproducible and copy-pasteable.
- For container runs, use the container’s filesystem paths, not host-only paths.
- If a backend or GPU compatibility issue appears, fix the actual compatibility layer rather than working around it in only one recipe.
- If a config is already close to correct, try a second and third variant before stopping.

## Stop Condition

Stop only when:
- you have a clearly better-performing strategy to report, or
- further changes are unlikely to improve accuracy without a larger redesign, or
- the user asks you to stop.

The intended behavior is persistent, iterative experimentation until the best practical result is reached.
