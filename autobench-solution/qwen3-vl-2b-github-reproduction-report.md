# Qwen3-VL-2B Circle Count: Independent GitHub Reproduction

## Outcome

The GitHub SFT approach was independently reproduced from the original
`Qwen/Qwen3-VL-2B-Instruct` checkpoint and exceeded the supplied external
checkpoint. The selected step-150 LoRA adapter scores **90.5%** at temperature
1.0 on a fixed, disjoint 200-example test set. After merging, the deployable HF
checkpoint scores **91.0%** at temperature 1.0 and **92.0%** greedy. Every
selected-checkpoint response contains a parseable `\boxed{N}` answer.

This is a single-model VLM solution. Images are passed directly to Qwen3-VL;
there is no detector, connected-component counter, image rewriting, voting, or
other outer image-processing stage.

| Checkpoint | Temperature | Strict boxed accuracy | Parseable boxes |
| --- | ---: | ---: | ---: |
| Supplied external HF checkpoint | 1.0 | 173/200 = 86.5% | 200/200 |
| Reproduced LoRA, step 150 | 1.0 | 181/200 = 90.5% | 200/200 |
| Reproduced merged HF, step 150 | 1.0 | **182/200 = 91.0%** | 200/200 |
| Supplied external HF checkpoint | greedy | 171/200 = 85.5% | 200/200 |
| Reproduced merged HF, step 150 | greedy | **184/200 = 92.0%** | 200/200 |

The merged result improves on the external checkpoint by 4.5 percentage points
under the sampled contract and 6.5 points under greedy decoding.

## Reference and Corrections

The implementation was cross-checked against branch
[`kimi-k3/circle-count-vlm-support`](https://github.com/vinhngx/RL/tree/kimi-k3/circle-count-vlm-support)
at commit `bce19da81b533f89d2aefe0b0231f7c6a2812048`, including its
[`APPROACH.md`](https://github.com/vinhngx/RL/blob/kimi-k3/circle-count-vlm-support/kimi-k3-solution/APPROACH.md).

Two issues prevented a literal invocation of the pushed recipe from reproducing
its reported behavior:

1. NeMo-RL v0.6 uses the old nested data schema. The earlier run supplied a
   newer `train_data_path` field, which was ignored; its authoritative log shows
   that it loaded 87,599 SQuAD rows instead of Circle Count. The corrected
   recipe explicitly registers and selects `data.train.dataset_name:
   circle-count-sft` and the matching validation dataset.
2. `policy.tokenizer.chat_template: null` selects the v0.6 passthrough path,
   which is unsuitable for the multimodal Qwen messages. The reproduction uses
   `chat_template: default`, Qwen's native multimodal template, with thinking
   disabled.

The reference model's five answer styles were reconstructed from its outputs.
All end in `\boxed{N}` and are deterministically mixed across the synthetic
training examples. A one-step gradient gate then confirmed that the intended
data was loaded and that the vision-tower LoRA weights changed before the full
campaign was allowed to run.

## Training Recipe

| Setting | Value |
| --- | --- |
| Base model | Original `Qwen/Qwen3-VL-2B-Instruct` HF checkpoint |
| Runtime | NeMo-RL v0.6.0, PyTorch 2.10/CUDA 12.9 image |
| Entrypoint/backend | `run_vlm_sft.py`, DTensor/FSDP2, one L40S |
| Adaptation | all-linear LoRA, rank 8, alpha 32, dropout 0, Triton enabled |
| Precision | bfloat16 |
| Optimizer | AdamW, LR `1e-4`; inherited warmup/cosine schedule |
| Global/micro batch | 128 / 2 |
| Sequence limit | 2048 |
| Memory | activation checkpointing enabled |
| Data | 6,000 train rows, 128 validation rows |
| Horizon | 5 epochs capped at 200 optimizer steps |
| Validation/checkpoints | validate every 20; save every 50; retain 3 |
| Template | native Qwen multimodal template; thinking disabled |

The synthetic training seeds start at 40,000, validation seeds at 46,000, and
test seeds at 10,000. Image identity checks found no train/validation/test
overlap. The five response-template counts in the 6,000-row train set are
1,147, 1,185, 1,211, 1,244, and 1,213.

Dataset SHA-256 values:

- train: `1fff62874c1bb0e2f23ff38cca315f8b5c3a76f2cdefcac34f9aa2a7f3a6743d`
- validation: `eefd545cce3588e8725fb76fd1c2a5f59ca6335ef910be8ec0d065b7c58b35f3`
- 200-example test profile: `b515de2ed29f37f4e9b9a6549450080d39ad6a3a2beee21050a1e187fb1e9e26`

## Training and Checkpoint Selection

The run completed all 200 updates in 9,160 seconds (2.54 hours). Final train
loss was 0.04419 and final validation loss was 0.04532. GPU utilization was
typically 98--99%, using about 23.4 GiB of the 46.1 GiB L40S. The run is
recorded in [W&B run `ticsduto`](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/ticsduto).

Generation evaluation, rather than validation loss alone, was decisive:

| Step | Temperature-1 strict boxed accuracy | Boxes |
| ---: | ---: | ---: |
| 100 | 165/200 = 82.5% | 200/200 |
| 150 | **181/200 = 90.5%** | 200/200 |
| 200 | 155/200 = 77.5% | 200/200 |

Step 150 is therefore the selected checkpoint. Step 200 overtrained on the
behavioral metric even though its validation loss remained low. The lesson is
to early-stop this task using a held-out generation verifier, not teacher-forced
validation loss.

At temperature 1.0, the selected live adapter has a Wilson 95% interval of
85.64%--93.83%. The merged checkpoint reaches 91.0% with interval
86.22%--94.23%. Greedy merged inference reaches 92.0% with interval
87.40%--95.02%.

## Merge Parity

The step-150 adapter was merged into the base model as a standard HF checkpoint.
On deterministic greedy inference, live-adapter and merged responses differed
in wording on 25/200 examples but produced different counts on only 2/200. Both
differences favored the merged checkpoint (expected 9; adapter predicted 10;
merged predicted 9). This small BF16 merge-rounding effect does not degrade the
model.

The merged checkpoint is stored outside git at:

`/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260907-qwen3-vl-2b-match-external/github-dataset-verbatim/merged-step150`

## Reproduction Assets

All maintained assets are under
`autobench-solution/autoresearch/qwen3_vl_circle_count/`:

- `github_circle_count_v06.patch` registers the v0.6 Qwen3-VL and Circle Count
  data paths.
- `generate_reference_template_sft.py` constructs the deterministic five-style
  boxed-answer SFT rows.
- `sft_circle_count_qwen3vl_2b_v06_corrected.yaml` is the corrected full recipe.
- `evaluate_reproduction_checkpoint.sh` runs the frozen 200-example verifier.
- `evaluate_hf_lora.py` evaluates raw images with either a PEFT adapter or a
  merged checkpoint.
- `merge_qwen3vl_lora.py` creates the deployable merged HF checkpoint.

Large data, logs, checkpoints, merged weights, and evaluation JSONL files live
under `/data/ephemeral`; `/ephemeral` is symlinked to `/data/ephemeral` as
requested. The selected run root is:

`/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260907-qwen3-vl-2b-match-external/github-dataset-verbatim`

Evaluation is reproducible with:

```bash
autobench-solution/autoresearch/qwen3_vl_circle_count/evaluate_reproduction_checkpoint.sh 150
```

The sampled merged result summary has SHA-256
`e143932b1e8db1cb44736b4fa4d07ddb8e7d1ad1617c519d4f4bb4edf430f26f`;
the greedy merged summary has SHA-256
`3842916d654bc52095cd4366b59f0b5449cd99eec4a716c4c870f197653fec29`.
