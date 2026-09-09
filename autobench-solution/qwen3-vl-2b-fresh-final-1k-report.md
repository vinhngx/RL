# Qwen3-VL-2B Fresh Final 1K Checkpoint Comparison

Date: 2026-09-09

## Result

Six checkpoints that had successively held or represented a campaign-best
development score were evaluated on the same newly generated 1,000-example
natural Circle Count set. The two latest GRPO checkpoints **tie at 950/1000
(95.0%)**. Every checkpoint produced a parseable boxed answer on all 1,000
examples.

| Rank | Checkpoint | Prior dev | Fresh correct | Fresh accuracy | Wilson 95% |
|---:|---|---:|---:|---:|---:|
| 1 | Counterfactual-pair GRPO step 2 | 190/200 | 950/1000 | 95.0% | 93.5%--96.2% |
| 1 | Prompt-matched edge GRPO | 189/200 | 950/1000 | 95.0% | 93.5%--96.2% |
| 3 | Natural scaled DPO | 188/200 | 948/1000 | 94.8% | 93.2%--96.0% |
| 4 | Strong answer DPO | 187/200 | 946/1000 | 94.6% | 93.0%--95.8% |
| 5 | Reproduced SFT step 150 | 184/200 | 925/1000 | 92.5% | 90.7%--94.0% |
| 6 | Early dynamic exact GRPO | 186/200 | 920/1000 | 92.0% | 90.2%--93.5% |

The retained 95% models improve over SFT by 25/1000, or 2.5 percentage
points. In the paired comparison, the counterfactual model is uniquely correct
on four examples and the edge model is uniquely correct on four others
(two-sided exact McNemar p=1.0). The later checkpoint therefore has no
measurable advantage over its 189/200 parent on this final set.

Against SFT, counterfactual GRPO is uniquely correct on 36 examples while SFT
is uniquely correct on 11 (exact paired p=0.000346). The improvement over SFT
is supported by this sample even though the originally requested >98% target
is not achieved.

## Dataset and inference contract

- Generator: the NeMo-Gym `circle_count/generate_data.py` used by the campaign
- Seeds: 30,000,000 through 30,000,999 inclusive
- Examples: 1,000 natural-distribution raw images
- Dataset SHA-256: `f96c7fc23f40befad563289762acce7bd0cd84c1d913e9da730bd17aaa88eff3`
- Dataset path: `/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-final-1k-best-checkpoints/data/profile-seed30000000-n1000.jsonl`
- Prompt: canonical system-role boxed prompt
- Decoding: greedy, temperature 0, seed 42, batch size 4, maximum 128 new tokens
- Inference: one Qwen3-VL-2B checkpoint directly consumes the raw image and
  question; no detector, preprocessing counter, ensemble, or outer aggregation

The dataset was generated and its manifest/checksum fixed before any checkpoint
was evaluated. It is now exposed final evidence and must not be used for future
training, checkpoint selection, prompt tuning, or threshold tuning.

## Paired behavior

All 50 errors made by each tied leader are exactly off by one. Counterfactual
GRPO has 28 undercounts and 22 overcounts; edge GRPO has 26 undercounts and 24
overcounts. Their unique-correct indices are:

- Counterfactual only: `63, 151, 733, 814`
- Edge only: `39, 190, 523, 990`

Across all six checkpoints, an oracle that selected any correct prediction
would score 966/1000, leaving 34 unanimous failures. This is diagnostic only:
the campaign's single-model inference constraint makes such an oracle or
ensemble ineligible.

The largest gains over SFT are at high counts. For expected counts 10--13, SFT
gets 19/45 while each tied leader gets 38/45. Counts 0 and 1 are perfect for
all checkpoints. Counts 14--15 contain only three examples and are too sparse
for a stable slice conclusion.

## Artifacts and reproduction

All generated assets are under:

`/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260907-qwen3-vl-2b-grpo-98/fresh-final-1k-best-checkpoints`

Important files:

- `manifest.json`: seed range, generator, and frozen dataset checksum
- `evals/*.jsonl`: per-example responses and parsed predictions
- `evals/*.summary.json`: per-checkpoint aggregate and slice metrics
- `comparison.json`: ranking, paired tests, oracle union, unanimous errors
- `comparison.md`: compact generated comparison table
- `logs/*.log`: complete evaluator logs

Reproduce from `/home/ubuntu/RL` with:

```bash
bash autobench-solution/autoresearch/qwen3_vl_circle_count/run_fresh_final_1k_best_checkpoints_docker.sh
```

The runner is idempotent: it verifies the fixed dataset manifest and skips
completed checkpoint outputs. Evaluation used
`nvcr.io/nvidia/nemo-rl:v0.6.0` on one NVIDIA L40S. No GPU process remained
after completion.
