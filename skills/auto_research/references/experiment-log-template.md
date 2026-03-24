# Experiment Log Template

Use this as the model for an untracked TSV such as `reports/auto_research_results.tsv`.

```tsv
index	branch	commit	recipe	metric	memory_gb	elapsed_min	status	description
1	autoresearch/2026-03-24-dapo-qwen2p5/baseline	abc1234	examples/configs/recipes/llm/dapo-qwen2.5-0.5b-b512-p512-g16-fp16.yaml	0.000000	0.0	12.4	crash	baseline failed before training
2	autoresearch/2026-03-24-dapo-qwen2p5/prompt-compact-schema	def5678	examples/configs/recipes/llm/dapo-qwen2.5-0.5b-b512-p512-g16-fp16.yaml	0.742100	43.9	58.7	keep	baseline with current prompt template
3	autoresearch/2026-03-24-dapo-qwen2p5/rollout-batch-up	fedcba9	examples/configs/recipes/llm/dapo-qwen2.5-0.5b-b512-p512-g16-fp16.yaml	0.751200	44.1	59.8	discard	raise rollout batch size without prompt changes
```

Suggested interpretation:
- `index` is the attempted experiment count; use it for rules like `do 50 experiments`
- `metric` should be the primary task or validation metric reported by the recipe, such as `val_accuracy` or `accuracy`
- `elapsed_min` is the wall-clock duration of the run; sum it or compare it against the remaining budget when the user gives time limits
- `memory_gb` is an auxiliary resource signal, not the target metric
- use `0.000000` and `0.0` for crash rows if no valid metric was produced
- keep the description short and hypothesis-focused
- `branch` should use the shared experiment prefix so all hypotheses stay grouped

Status values:
- `keep`
- `discard`
- `crash`
