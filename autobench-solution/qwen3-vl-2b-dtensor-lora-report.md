# Qwen3-VL-2B DTensor LoRA SFT Report

## Scope

This experiment starts from the original Hugging Face
`Qwen/Qwen3-VL-2B-Instruct` checkpoint and trains with NeMo-RL's DTensor
backend on one NVIDIA L40S (44.42 GiB usable). It uses one raw task image and
one question per example. There is no detector, segmentation, connected-
component analysis, image rewriting, multi-call voting, or answer aggregation.

The selected adaptation is rank-8 LoRA on language attention
`q/k/v/o` projections and vision-attention `qkv/proj` projections. The base
model remains Qwen3-VL-2B; no 4B model is involved. FlashAttention 2 runs under
bfloat16 autocast.

## Microbatch Search

| Recipe | Microbatch | Result | Peak/steady memory | One-step time |
| --- | ---: | --- | ---: | ---: |
| Full SFT, no activation checkpointing | 1 | OOM on step 2 | 44.37/44.42 GiB | 1,893.05s training on step 1 |
| Full SFT, no activation checkpointing | 2 | OOM in backward | 44.16/44.42 GiB | n/a |
| Full SFT, activation checkpointing | 2 | Stable but slower | about 30.6 GiB | Incomplete after 18 min |
| Full SFT, activation checkpointing | 4 | Sustained training | about 28.23 GiB before backward peak | Boundary probe stopped early |
| Full SFT, activation checkpointing | 8 | OOM in backward | 43.23/44.42 GiB | n/a |
| All-linear LoRA | 4 | OOM in forward | 44.40/44.42 GiB | n/a |
| Attention LoRA, raw images | 3 | Stable | about 39.35 GiB | 1,854.60s |
| Attention LoRA, processor `max_pixels=512^2` | 4 | OOM in backward | 44.35/44.42 GiB | n/a |
| Attention LoRA, processor `max_pixels=512^2` | 3 | Stable | about 39.35 GiB | 1,859.79s |

Microbatch 3 is the largest stable setting. Microbatch 4 exhausted the GPU
during backward while trying to allocate another 2.34 GiB. The stable run
held 45--46% reported SM utilization and about 39.35 GiB, leaving roughly
5.1 GiB of device headroom. The worker simultaneously used about one full CPU
core. Together with the flat utilization across smaller batches, this points
to a framework/kernel-path ceiling rather than insufficient batch density.

The standard processor-side 512-pixel cap did not improve the measured path:
its step differed from the raw-input trial by only 5.19 seconds (0.28%) and
used the same memory. It is therefore omitted from the selected recipe.

## Best Stable LoRA Recipe

- Original `Qwen/Qwen3-VL-2B-Instruct` Hugging Face weights.
- DTensor/FSDP2, tensor parallel 1, no activation checkpointing.
- Rank-8 attention LoRA, alpha 32, no dropout.
- bfloat16 with FlashAttention 2.
- Global batch 126 and microbatch 3 (42 accumulation passes).
- AdamW at 2e-5 with four-step linear warmup and cosine decay.
- 200 requested optimizer steps; validation and weights-only checkpoint every
  five steps.

The measured training-only projection is 103.3 hours for 200 steps on this
single L40S, before validation and checkpoint overhead. Consequently a
five-hour allocation cannot complete the requested schedule; it can execute
roughly nine training steps at the measured rate. The configuration retains
all 200 requested steps, with frequent checkpoints so an allocation-limited
run produces resumable weights.

The exact LoRA configuration is
`autoresearch/qwen3_vl_circle_count/vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor-lora-attn-b126-mb3-200.v1.yaml`.
Large assets, logs, and checkpoints live under `/ephemeral`, which is a symlink
to `/data/ephemeral`.

## Decision and Campaign

LoRA was rejected as the final campaign choice. Its completed 1,854.60-second
step is effectively the same duration as the previously measured full-update
steps (about 1,866--1,880 seconds), and its 45--46% SM utilization does not
improve on full SFT's 47--48%. LoRA reduces trainable state and permits a
larger microbatch, but fixed-global-batch throughput stays flat because the
base model's forward/backward work and this DTensor path dominate.

The initial requested full-SFT campaign used global batch 128 and microbatch 1.
Step 1 completed with training loss 0.3208 and validation loss 0.27645; training
took 1,893.05 seconds, validation 231.41 seconds, and checkpointing 52.52
seconds. Its 8.0-GiB weights checkpoint is retained under the `/ephemeral`
campaign directory. Step 2 then OOMed in forward at 44.37/44.42 GiB, proving
that the apparent one-step mb1 feasibility did not survive validation and
checkpoint transitions.

Activation checkpointing was therefore enabled and the global-batch-128
divisors were searched upward. Microbatch 8 OOMed in backward while requesting
4.69 GiB with 43.23 GiB already in use. Microbatch 4 is the selected upper
candidate and the 200-step campaign configuration is
`autoresearch/qwen3_vl_circle_count/vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor1tp1-ac-b128-mb4-200.v1.yaml`.
Its allocation-limited result will be added after the run terminates.
