# Qwen3-VL Circle Count Auto-Research Report

## Outcome

This five-hour campaign evaluated and trained `Qwen/Qwen3-VL-2B-Instruct`
for the Circle Count Gym task on one NVIDIA L40S. The best zero-update result
was the answer-first prompt at **46.88% validation accuracy (30/64)**, compared
with the out-of-the-box baseline of **31.25% (20/64)**. GRPO was unstable at
the tested learning rates. A leakage-free supervised run was therefore used
to teach the exact answer format and visual counting task.

The retained 20-step SFT run completed with **0.1350 validation loss** and a
successfully finalized 4.0 GB weights-only checkpoint. It scored **59.38%
validation accuracy (38/64)** and **50.00% on the untouched final split
(64/128)**. This exceeded the original 31.25% baseline but fell short of the
90% working target.

## Experiment Contract

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Task | Circle Count Gym, `circle_count_simple_agent` |
| Working target | 90% verifier accuracy |
| Hardware | 1x NVIDIA L40S, 46 GB |
| Host driver | 565.57.01 |
| Runtime | NeMo-RL v0.5, CUDA 12.9 compatibility image |
| Training backend | Megatron-Core |
| Generation backend | vLLM |
| Precision | bfloat16 |
| Campaign window | 2026-09-02 05:52:40–10:52:40 UTC |

The host's CUDA 13.2 NeMo-RL image required a newer NVIDIA driver. The tested
`nemo-rl:qwen3vl-smoke-cu129` image was retained because it had already passed
an end-to-end Qwen3-VL GRPO optimizer step on this machine.

## Frozen Data

The campaign generated deterministic JSONL splits before optimization and did
not change them between hypotheses:

| Split | Samples | Seed range | SHA-256 prefix |
| --- | ---: | --- | --- |
| Train | 256 | 0+ | `1a7f2fe0d945` |
| Validation | 64 | 10000+ | `abf079057eb4` |
| Final evaluation | 128 | 20000+ | `8cfcaeea9190` |

The pinned Gym source is commit `c3bac96`. Supervised targets were derived
only from each training record's explicit `circles` list and `target_color`.
Validation and final labels were never added to the training set.

## Results

| Experiment | Update | Validation accuracy | Decision |
| --- | --- | ---: | --- |
| Out-of-box baseline | None | 31.25% (20/64) | Baseline |
| Strict boxed-only prompt | None | 21.88% (14/64) | Discard |
| Greedy decoding | None | 26.56% (17/64) | Discard |
| Answer-first prompt | None | 46.88% (30/64) | Keep |
| GRPO, LR 1e-5 | 4 steps | 26.56% (17/64) | Discard |
| GRPO, LR 1e-6 | 16 steps | 31.25% (20/64) | Discard |
| SFT loss probe | 24 steps | 0.1312 validation loss | Save failed |
| Supervised fine-tuning | 20 steps, weights only | 59.38% (38/64) | Keep |
| Supervised final evaluation | Zero-update, 128 samples | 50.00% (64/128) | Final |

The baseline frequently reasoned for the full generation budget and failed to
emit a parseable answer: 25 of 64 responses were truncated. Requiring the
boxed count at the beginning removed that failure mode. All 64 answer-first
responses parsed, none truncated, and mean response length fell to 21 tokens.
The remaining errors were genuine counting errors, especially at higher
counts.

GRPO briefly reached 53.12% at step 4 in one run, but that run could not save
its optimizer state because of a NeMo-RL v0.5 serializer incompatibility. A
repeat using weights-only checkpoints collapsed to 26.56% at LR 1e-5. Lowering
the rate to 1e-6 held 43.75% through step 12 and then fell to 31.25% at step
16. Both training hypotheses were discarded.

The retained SFT recovery run reduced its fixed 16-example validation loss
from 0.2471 at step 10 to 0.1350 at step 20 (45.4%). Checkpoint save exited 0.
The subsequent evaluator loaded the weights-only checkpoint into the Megatron
policy without optimizer state. Two attempts initially hit `ENOSPC` because
Gym copied roughly 22 GB from its uv cache into an isolated child venv. Adding
hardlink-mode child installs eliminated that duplicate. The completed 64-item
validation pass scored 59.38%; only then was the untouched 128-item final split
run once, scoring 50.00%.

## Retained Supervised Recipe

The 24-step probe reduced its held-out loss from 0.2038 at step 12 to 0.1312
at step 24, but NeMo-RL v0.5 attempted to serialize the optimizer despite the
newer `save_optimizer: false` setting. The in-memory policy was lost when that
save failed. A direct v0.5 weights-only patch was verified in a disposable
container before the retained recovery run.

The final SFT recipe uses the retained answer-first system instruction and a
native Qwen multimodal conversation:

- 20 optimizer steps over the frozen 256-example training set
- global batch 8 and microbatch 1
- peak learning rate 2e-6, four warmup steps, cosine decay to 2e-7
- maximum sequence length 2048
- optimizer fully offloaded to CPU
- activation checkpointing disabled
- 16-example loss sentinel at steps 10 and 20
- weights-only checkpoint at step 20

Activation checkpointing remains disabled because the v0.5 Qwen3-VL bridge
passes non-tensor multimodal metadata through PyTorch recomputation. CPU
optimizer offload provides sufficient memory headroom on the L40S.

## Compatibility Work

The campaign adds a narrow compatibility patch for the validated v0.5 image:

1. Convert Responses API `input_image` parts to Chat Completions `image_url`
   parts and allow those parts in the v0.5 Pydantic message union.
2. Treat Megatron's 4-D causal mask as absent when Qwen3-VL's multimodal rotary
   position helper expects a 2-D padding mask. Megatron still applies causal
   attention in the language model.
3. Avoid serializing the incompatible distributed optimizer scalar state in
   both GRPO and the older SFT save path; retained checkpoints contain policy
   weights and tokenizer state.
4. Install Gym child environments with uv hardlinks, avoiding a roughly 22 GB
   duplicate package copy in each disposable container.
5. Preserve heterogeneous multimodal JSON messages in the v0.5 SFT loader and
   use the tokenizer's native default Qwen chat template.

## Reproduction

From the repository root, the baseline or GRPO launcher is:

```bash
WANDB_MODE=offline \
SYSTEM_PROMPT_FILE=autobench-solution/autoresearch/qwen3_vl_circle_count/prompt_answer_first.txt \
EXPERIMENT=prompt-answer-first \
bash autobench-solution/autoresearch/qwen3_vl_circle_count/run_v05_docker.sh
```

The retained supervised recipe is:

```bash
WANDB_MODE=offline \
EXPERIMENT=sft-supervised-20step-weights-only \
bash autobench-solution/autoresearch/qwen3_vl_circle_count/run_sft_v05_docker.sh
```

The zero-update evaluator loads the saved checkpoint and selects a frozen split
using the launcher's `PRETRAINED_CHECKPOINT_PATH`, `VALIDATION_SPLIT`, and
`EXTRA_OVERRIDES` inputs.

## W&B and Persisted Evidence

Every authoritative run produced a local W&B offline directory. The supplied
credential was not accepted by pinned W&B 0.21.1 because it was not a
40-character API key, so no run was silently dropped and no credential is
included in this report or Git history. The offline runs can be synced after a
compatible key is installed.

Heavy state is kept outside Git under:

```text
/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260902-qwen3-vl-circle-count
```

That directory contains frozen data, run logs, response tables, local W&B
runs, the imported base-model artifact, and the retained trained checkpoint.

## Verification

- Frozen split hashes were recorded before training.
- All prompt-only results completed all 64 validation examples.
- The SFT loader reported exactly 256 training and 64 validation examples.
- The first corrected SFT optimizer step completed at loss 0.5324.
- Launcher scripts pass `bash -n`; the Git diff passes `git diff --check`.
- The 20-step run exited 0 and finalized `step_20` (4.0 GB).
- Validation loss improved from 0.2471 at step 10 to 0.1350 at step 20.
- The retained checkpoint loaded without optimizer state.
- The validation evaluator processed 64/64 examples at 59.38% accuracy.
- The one untouched final evaluator processed 128/128 examples at 50.00%.

## Limitations

The 90% working target is an optimization goal, not a claimed result. This
single-GPU campaign explored a small number of high-value hypotheses within a
fixed five-hour wall-clock budget. The report states the measured final result
even if it falls short of that target.
