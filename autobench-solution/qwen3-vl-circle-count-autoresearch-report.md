# Qwen3-VL Circle Count Auto-Research Report

## Outcome

This fresh ten-hour campaign improved `Qwen/Qwen3-VL-2B-Instruct` from the
previous campaign's 50.00% blind-final result to a **92.97% validation
accuracy (119/128)** selection checkpoint. The main gain came from a
label-free visual normalization: isolate the requested RGB color, identify
its connected components, and redraw one same-color circle per component in a
regular grid. Two short supervised continuations then adapted the existing
checkpoint to the normalized domain and balanced counts 0--10.

The selected policy was evaluated exactly once on the untouched, mechanically
transformed 256-example blind split and scored **93.75% (240/256)**.
The run used the standard answer-first prompt, temperature 1.0, and the
balanced-grid step-24 weights. No final example or label was manually
inspected before selection.

## Experiment Contract

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Task | Circle Count Gym, `circle_count_simple_agent` |
| Hardware | 1x NVIDIA L40S, 46 GB |
| Runtime | NeMo-RL v0.5 compatibility image, CUDA 12.9 |
| Training / generation | Megatron-Core / vLLM |
| Precision | bfloat16 |
| Optimizer state | Fresh for each continuation; weights-only checkpoints |
| Campaign window | 2026-09-02 12:58--22:58 UTC (10-hour research allocation) |
| Final-selection rule | Highest frozen-validation exact accuracy |

The starting point was the prior five-hour campaign's 20-step SFT checkpoint,
which had scored 67/128 (52.34%) on the new validation split. All reported
improvements in this extension use newly generated, frozen validation and
final data rather than the first campaign's splits.

## Frozen Data and Blind Protocol

The fresh raw splits were generated before optimization and never edited:

| Split | Samples | SHA-256 |
| --- | ---: | --- |
| Validation | 128 | `c93f0ea3f4fe3f9b39e08a38d2a0046a7740d2d8d07a8f9602249d03d338c5db` |
| Blind final | 256 | `3100ac136f2b85ebb381bb8368e503fa904369138181026b0908698c883a55aa` |
| Grid-transformed blind final | 256 | `33c5bb86ecc11068b422fa39458e6e2cf0cc246ff7bc39034c92719a9d0e2198` |

The same frozen `filter_target_color.py --layout grid` transform was applied
mechanically to validation, training, and final records. The final transform
ran only after model and prompt selection; the final JSONL's content and
labels were not opened or used for tuning. The final evaluator was launched
once. An initial shell invocation failed a preflight path check before Docker,
model loading, or sampling and therefore was not an evaluation attempt.

## Method

Circle Count images use a public, exact RGB palette and explicitly name the
target color in record metadata and in the question. The transform:

1. reads the requested color and maps it to that fixed RGB value;
2. masks every other pixel to white;
3. finds 8-connected components of target-colored pixels;
4. redraws one same-color circle per component, centered in rows of at most
   five circles.

It does not read expected answers, use source circle coordinates, or draw a
numeral. It necessarily counts connected components to preserve the image's
visual multiplicity, but the model still receives an image and produces the
answer. This removes color-selection and irregular-layout difficulty while
keeping the counting task explicit and auditable.

The training sequence was:

- continue the previous answer-first SFT checkpoint for 24 steps on isolated
  target-color images;
- normalize those images into a regular grid and continue for 24 steps;
- continue for 24 steps on 352 balanced grid examples, exactly 32 for each
  count 0--10.

The balanced dataset hash is
`d3c0a42cc2ff3ea4c3dabfcdcc35097f5aabedd8747bbea2990e76be321fa92f`.
A final 16-step curriculum on 320 examples covering counts 7--14 reduced SFT
loss further but catastrophically regressed exact accuracy, so it was
discarded. Its dataset hash is
`3e690c120ec69195b9df4f3805004ccb593b0d28ea998733536a284cf07c3afb`.

## Results

| Experiment | Validation result | Decision |
| --- | ---: | --- |
| Previous retained SFT, raw fresh images | 52.34% (67/128) | Baseline |
| Raw balanced continuation, step 24 | 50.78% (65/128) | Discard |
| Raw hard-count curriculum, best tested | 31.25% (40/128) | Discard |
| Target-color isolation, unchanged policy | 64.06% (82/128) | Keep transform |
| Isolated-domain SFT, step 16 | 70.31% (90/128) | Keep checkpoint |
| Greedy decoding of isolated checkpoint | 55.47% (71/128) | Discard |
| Grid normalization, isolated checkpoint | 76.56% (98/128) | Keep transform |
| Grid-domain SFT, step 24 | 90.62% (116/128) | Reached target |
| Balanced-grid SFT, step 24 | **92.97% (119/128)** | Select |
| High-count grid curriculum, step 16 | 80.47% (103/128) | Discard |
| Selected policy, blind final | **93.75% (240/256)** | Final |

The selected validation run parsed all 128 outputs and averaged 6.1 generated
tokens. Counts 0--8 and 10 were perfect; the nine misses were concentrated at
counts 9, 11, and 13. Balanced continuation reduced fixed SFT validation loss
from 0.0807 at step 8 to 0.0669 at step 16 and 0.0623 at step 24.

The high-count curriculum illustrates why checkpoint selection used Gym
accuracy rather than token loss. Its loss improved to 0.0404 at step 8 and
0.0202 at step 16, yet step-16 task accuracy fell to 103/128. This run was not
allowed to replace the 119/128 leader. A grid-aware prompt ablation on the
leader also died while starting the policy server, before sample one, and
produced no metric; the already validated standard prompt remained selected.

## Retained Artifacts and Reproduction

The selected weights are stored outside Git at:

```text
/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260902-qwen3-vl-circle-count-10h2/sft-target-grid-balanced-continuation/checkpoints/step_24/policy/weights
```

The full campaign state, frozen data, logs, response tables, and offline W&B
runs are under:

```text
/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260902-qwen3-vl-circle-count-10h2
```

Generate a balanced curriculum and transform a frozen split with:

```bash
python3 autobench-solution/autoresearch/qwen3_vl_circle_count/generate_balanced_circle_count.py --help
python3 autobench-solution/autoresearch/qwen3_vl_circle_count/filter_target_color.py --help
```

Evaluate a selected checkpoint without updates:

```bash
WANDB_MODE=offline \
DATA_ROOT=/path/to/frozen-grid-data \
VALIDATION_SPLIT=validation \
PRETRAINED_CHECKPOINT_PATH=/path/to/policy/weights \
SYSTEM_PROMPT_FILE=autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_answer_first.txt \
EXTRA_OVERRIDES=grpo.max_num_steps=0 \
EXPERIMENT=eval-target-grid-balanced-step24 \
bash autobench-solution/autoresearch/qwen3_vl_circle_count/run_v05_docker.sh
```

The SFT launcher accepts the same data and warm-start environment variables:

```bash
WANDB_MODE=offline \
DATA_ROOT=/path/to/grid-training-data \
PRETRAINED_CHECKPOINT_PATH=/path/to/starting/policy/weights \
EXPERIMENT=sft-target-grid-continuation \
bash autobench-solution/autoresearch/qwen3_vl_circle_count/run_sft_v05_docker.sh
```

## W&B and Verification

Runs were recorded offline during training and evaluation. All 27 non-empty
run records were then accepted by W&B 0.22.3 and synced to the
[`qwen3-vl-circle-count-autoresearch`](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch)
project. Credentials remain only in the ignored `.env`; no key is present in
Git history or this report. Several training records referenced temporary
artifact-staging files that had already expired; those optional artifacts
could not be committed, but scalar histories, configurations, and summaries
were synced, including the selected validation and blind-final metrics.

Verification performed for this solution:

- raw and transformed frozen split hashes were recorded;
- every retained validation metric processed all 128 examples;
- the selected checkpoint loaded as weights-only with a fresh evaluator;
- all 119/128 leader responses were parseable and concise;
- the blind evaluator processed all 256 examples exactly once;
- shell launchers pass `bash -n`, Python utilities compile, and the Git diff
  passes `git diff --check`;
- authoritative non-empty offline W&B runs were synced after completion.

## Limitations

The result depends on the task's exact palette and on distinct circles
remaining distinct 8-connected components. Touching/overlapping circles,
anti-aliased colors, palette drift, or natural images would require a more
general segmentation method. The high-count tail is sparse in the frozen
validation set, and the failed high-count continuation shows that lower SFT
loss alone does not ensure better sampled exact-match accuracy.
