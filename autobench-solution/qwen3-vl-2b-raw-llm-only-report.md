# Qwen3-VL-2B Raw-Image, LLM-Only Auto-Research Report

## Outcome

This campaign optimized the original `Qwen/Qwen3-VL-2B-Instruct` under a
strict inference contract: one unmodified task image and its question go
directly into one model call, and the model emits the answer. There is no
segmentation, component detection, masking, cropping, tiling, image rewriting,
OCR, multi-call voting, or answer aggregation.

The selected checkpoint scored **61.33% (157/256)** on the frozen raw-image
validation split, improving the fresh retained-2B baseline of **56.25%
(144/256)** by 5.08 percentage points in the selection pass. Three post-lock
replications put the four-pass mean at **58.98%**, exposing meaningful sampling
variance. The checkpoint was produced by keeping the language decoder frozen
while adapting the vision encoder and projector, first on the natural task
distribution and then briefly on a raw-image curriculum weighted toward counts
6--14. The winning checkpoint was step 12 of the latter continuation.

The selected policy was evaluated exactly once on the untouched 512-example
blind split and scored **58.01% (297/512)**. All examples were processed, and
responses averaged 6.2 tokens.

## Inference Contract

| Property | Selected solution |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Model size | Original 2B model; no 4B model or ensemble |
| Image input | Original generated image, byte-for-byte from the task record |
| Model calls | One |
| External vision processing | None |
| Output processing | Gym's standard boxed-answer parsing and exact scoring |
| Prompt | Fixed answer-first system prompt used during training |
| Sampling | Temperature 1.0 |

Earlier work in this repository reported a 93.75% result obtained with RGB
masking, connected-component detection, and grid redrawing. That result is
outside this stricter contract and is not a result of the solution documented
here.

## Experiment Contract

| Item | Value |
| --- | --- |
| Hardware | 1x NVIDIA L40S, 46 GB |
| Runtime | NeMo-RL v0.5 compatibility image, CUDA 12.9 |
| Training / inference | Megatron-Core / vLLM |
| Precision | bfloat16 |
| Starting weights | Prior 20-step Qwen3-VL-2B SFT weights |
| Optimizer state | Fresh per continuation; weights-only checkpoints |
| Campaign window | 2026-09-02 22:45--2026-09-03 08:45 UTC |
| Selection rule | Highest exact accuracy on one frozen raw validation split |

All large model, dataset, checkpoint, response, and W&B assets were written
under `/data`; `/ephemeral` points to `/data/ephemeral`.

## Frozen Data and Selection Protocol

The validation and final splits were generated and hashed before optimization:

| Split | Samples | SHA-256 |
| --- | ---: | --- |
| Validation | 256 | `c8fc49dd94bb5fc856d0a6ae979bb4c3c67bffdb2a8ed91eded3b835c1c89562` |
| Blind final | 512 | `c9ab0777ea93d4029f2e7bdf283539e21ac3c90552b270450a097dc67f32e4e0` |
| High-count training | 408 | `8c33c7efb1ce563e717cfe655885e87b4a616775d15a0048086fd19d75286bf4` |

The 256-example validation split was reused for controlled comparisons. The
512-example final split remained untouched and uninspected during training,
prompt selection, decoding selection, and checkpoint selection. It was
reserved for one final evaluation after those choices were locked.

An exact-image audit found no overlap between any selected training set and
either held-out split, nor between validation and blind final. The 480-example
balanced training set and 408-example high-count curriculum did share eight
images; this reuse is confined to training and is reported here for
reproducibility.

## Research Findings

The useful intervention was visual-side specialization without changing the
inference graph. Freezing the language model preserved the short, reliably
parseable boxed-answer behavior, while training both the vision encoder and
projection improved raw-image counting. A small natural-distribution
continuation corrected the initial visual-side model; a high-count continuation
then reduced undercounting and supplied the final one-example gain.

Projector-only training was insufficient, showing that the vision encoder
itself needed to move. Very low learning-rate natural reconsolidation regressed,
and prompt attempts that requested more explicit visual reasoning destabilized
the terse answer format. Lower-temperature decoding also regressed. SFT token
loss was therefore not used as the selection metric; every retained candidate
was chosen by exact Gym accuracy.

## Results

| Experiment | Raw validation accuracy | Decision |
| --- | ---: | --- |
| Retained 2B checkpoint | 56.25% (144/256) | Baseline |
| Spatial-band reasoning prompt | 12.89% (33/256) | Discard |
| Vision encoder + projector SFT, step 24 | 58.59% (150/256) | Continue |
| Projector-only SFT, best step 16 | 58.20% (149/256) | Discard |
| Natural visual-side calibration, step 8 | 60.94% (156/256) | Continue |
| Explicit target-only prompt | 33.59% (86/256) | Discard |
| Temperature 0.7 | 53.52% (137/256) | Discard |
| High-count visual-side SFT, step 12 | **61.33% (157/256)** | Select |
| Natural reconsolidation, best step 6 | 58.20% (149/256) | Discard |
| Full SFT batch 128, step 1 | 59.77% (153/256) | Discard |
| Full SFT batch 128, step 2 | 57.81% (148/256) | Discard |
| Selected policy, blind final | **58.01% (297/512)** | Final |

On the paired 256-example validation set, the selected checkpoint fixed 29
baseline errors while regressing 16 baseline-correct examples; 128 examples
were correct under both and 83 were wrong under both. The corresponding exact
two-sided McNemar p-value is 0.0725, so the measured 13-example gain is useful
selection evidence but not a high-confidence claim of a general improvement.

After model and final-result lock, three additional validation passes using the
same checkpoint, answer-first prompt, and temperature but seeds 43--45 scored
59.38% (152/256), 57.03% (146/256), and 58.20% (149/256). Including the
61.33% selection pass at seed 42, the four-pass mean is 58.98%, with a
1.83-percentage-point sample standard deviation and 57.03--61.33% range. These
replications characterize stochastic decoding variance; they did not reopen
checkpoint selection or touch blind final again.

A post-lock same-seed control reran the retained baseline at seed 43: it scored
52.73% (135/256), versus 59.38% (152/256) for the selected checkpoint. On that
pass the selected checkpoint fixed 32 baseline errors and regressed 15
baseline-correct examples (exact two-sided McNemar p=0.0186). Thus the selected
model beat the baseline by 5.08 and 6.64 percentage points in the two matched
seed-42 and seed-43 comparisons, even though absolute scores varied.

The selected validation checkpoint had signed count error +0.176 and mean
absolute error 0.520. Its per-color exact counts were blue 23/45, cyan 23/35,
green 17/29, orange 17/29, pink 16/21, purple 17/25, red 16/29, and yellow
28/43. On blind final, the expected-count mean was 4.748 and the prediction
mean was 4.865, for a smaller +0.117 signed bias. All 512 calls terminated
naturally with no truncation; 510 responses used only six or seven generated
tokens, and all 512 answers parsed successfully. The recorded request for every
example contained exactly one input image and one question. A byte-string
multiset comparison confirmed that all 512 recorded request images exactly
matched the image data URLs in the locked final source records.

A post-lock audit of the blind responses found mean absolute count error
0.5625, with 126 overcounts, 89 undercounts, and 297 exact predictions. Exact
accuracy by expected count was 75.0% at 0, 87.5% at 1, 86.8% at 2, 71.4% at
3, 60.0% at 4, 52.5% at 5, 50.0% at 6, 33.3% at 7, 22.6% at 8, and 21.4%
at 9. Counts 10--14 had only 36 examples in total, so their individual rates
are too noisy to interpret. By target color, exact accuracy ranged from 43.9%
for red to 65.7% for orange. Together with the uniformly short, naturally
terminated outputs, this points to dense visual counting—not response parsing
or verbosity—as the principal remaining bottleneck.

### Requested full-SFT batch-128/200-step trial

The requested recipe was instantiated exactly as a full-model 2B update with
global batch 128, microbatch 1, 200 optimizer steps, 100 two-step epochs, LR
1e-7, and cosine decay. It warm-started from the selected step-12 weights.
Validation and weights-only saving were scheduled every step to retain useful
evidence within the fixed campaign window. The updates ran without OOM, with
observed device memory reaching about 20.4 GB. The two policy updates took
1,865.78 and 1,880.29 seconds; validation and checkpointing brought total step
times to 2,119.17 and 2,126.17 seconds. At their mean measured rate, 200 steps
require about 104.1 training hours, or 117.9 hours with every-step validation
and saving, on one L40S. The fixed campaign therefore treats this as a measured
feasibility trial rather than claiming the 200-step schedule completed. Two
updates were completed. Training
loss fell from 0.1101 to 0.0881 and SFT validation loss from 0.1242 to 0.1235,
but exact Gym accuracy fell from the 61.33% starting checkpoint to 59.77% at
step 1 and 57.81% at step 2. Full SFT was therefore rejected.

## Selected Recipe

1. Start from the retained Qwen3-VL-2B SFT checkpoint.
2. Freeze the language decoder; update the vision encoder and projector for 24
   steps on balanced, unmodified raw images at LR 8e-7.
3. Continue from step 24 for 8 selected steps on independent natural raw images
   at LR 2e-7.
4. Continue from that checkpoint for 12 selected steps on raw examples weighted
   toward target counts 6--14 at LR 1e-7.
5. Keep the original answer-first prompt and temperature 1.0; select by frozen
   exact accuracy, not by SFT loss.

The final inference path remains simply:

```text
raw task image + question -> Qwen3-VL-2B -> boxed count
```

## Artifacts and Reproduction

The selected weights are stored outside Git at:

```text
/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260902-qwen3-vl-circle-count-raw-10h/sft-2b-highcount-visual/checkpoints/step_12/policy/weights
```

The campaign data, logs, checkpoints, responses, and local W&B records are at:

```text
/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260902-qwen3-vl-circle-count-raw-10h
```

The full-SFT configuration is
`autoresearch/qwen3_vl_circle_count/sft_2b_full_b128_200_v05.yaml`. The retained
visual-side configurations and launchers are in the same directory. Credentials
remain only in the ignored `.env` and are not included in Git history.

## W&B Runs

- [Selected high-count visual-side training](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/vy3qkzhf)
- [Selected checkpoint validation](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/us0lpbfr)
- [Post-lock validation seed 43](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/5rt00nuo)
- [Post-lock validation seed 44](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/gnrrfbsb)
- [Post-lock validation seed 45](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/4ewgkok4)
- [Post-lock baseline seed 43](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/npuqervf)
- [Full-SFT batch-128/200-step trial](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/3m4qbr6f)
- [Full-SFT step-1 evaluation](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/g18fov8v)
- [Full-SFT step-2 evaluation](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/k9cwa4c8)
- [Locked blind-final evaluation](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/6r0gh9sq)

## Limitations

Accuracy is 61.33% on frozen validation and 58.01% on blind final, and residual
errors are spread across colors rather than isolated to one palette entry. The
result is specific to this synthetic circle-count distribution and single-sample
temperature-1.0 decoding. The campaign evaluates stochastic decoding on one
fixed validation pass per candidate, so small one-example differences warrant
caution; post-lock replications quantify this variance for the selected model.
Wilson 95% intervals are 55.24--67.08% for the selected validation pass and
53.69--62.21% for blind final. The strict raw-only contract deliberately
forgoes the much larger gain available from deterministic task-specific image
normalization.
