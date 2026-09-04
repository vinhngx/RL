# Qwen3-VL-2B DTensor LoRA SFT Report

## Outcome

The requested 200-step SFT campaign runs the original Hugging Face
`Qwen/Qwen3-VL-2B-Instruct` checkpoint in the official NeMo-RL v0.6.0 image
with DTensor/FSDP2 and rank-8 all-linear LoRA. It consumes the synthetic
image/question/boxed-answer records directly; no detector, segmentation,
connected-component analysis, image rewriting, voting, or other outer vision
pipeline is used.

The robust single-L40S setting is microbatch 8. Microbatch 16 is faster per
sample but failed on a rare maximum-length batch at step 120. Microbatch 8
preserves the requested global batch of 128 with 16 gradient-accumulation
passes and completed all 200 steps without an OOM. Its best held-out loss was
0.0791 at step 50, compared with the unadapted starting loss
near 3.8 and the completed full-SFT baseline's final validation loss of 3.0596.

Weights and logs live under `/data/ephemeral`; `/ephemeral` points to that
volume. The selected W&B run is [vd6vw9v6](https://wandb.ai/hwinf_dcm/qwen3-vl-circle-count-autoresearch/runs/vd6vw9v6).

## Final Recipe

| Setting | Value |
| --- | --- |
| Base checkpoint | `Qwen/Qwen3-VL-2B-Instruct` (original HF weights) |
| NeMo-RL | official `nvcr.io/nvidia/nemo-rl:v0.6.0` (`sha256:336aa413...95fd3f2`) |
| Entry point | `examples/run_vlm_sft.py` |
| Backend | DTensor/FSDP2 v2, TP=1, CP=1, one NVIDIA L40S |
| Adaptation | LoRA rank 8, alpha 32, dropout 0, all linear layers including `lm_head` |
| Precision | bfloat16 |
| Global / micro batch | 128 / 8 (16 accumulation passes) |
| Sequence limit | 2048 tokens |
| Memory | activation checkpointing enabled; no CPU offload |
| Optimizer | AdamW, LR 1e-4, weight decay 0.1, betas 0.9/0.98, eps 1e-5 |
| Schedule | 4-step linear warmup from 0.1x, then cosine decay |
| Horizon | 200 optimizer steps |
| Validation / checkpoints | every 10 steps; retain best 2 adapters plus final state |
| Data | 87,599 synthetic train and 10,570 validation examples |

The runnable child config is
`autoresearch/qwen3_vl_circle_count/vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor-lora-all-linear-r8-lr1e4-b128-mb8-200-v06.v1.yaml`.
Launch it with `run_sft_v06_docker.sh` by setting `CONFIG_PATH` to that file.

## Results

| Step | Validation loss |
| ---: | ---: |
| 10 | 0.1302 |
| 20 | 0.1101 |
| 30 | 0.0862 |
| 40 | 0.0852 |
| 50 | 0.0791 |
| 60 | 0.0884 |
| 70 | 0.0879 |
| 80 | 0.0917 |
| 90 | 0.0936 |
| 100 | 0.0961 |
| 110 | 0.0862 |
| 120 | 0.0864 |
| 130 | 0.0872 |
| 140 | 0.0865 |
| 150 | 0.0879 |
| 160 | 0.0883 |
| 170 | 0.0888 |
| 180 | 0.0885 |
| 190 | 0.0876 |
| 200 | 0.0888 |

The loss collapsed during the first ten updates and reached its useful
minimum around step 50. Later steps measure plateau/regression rather than
continued improvement, so deployment should use the best validation-selected
adapter rather than the final-step state. Across all 200 updates, policy
training averaged 9.47 seconds and total step time averaged 9.70 seconds. The
range was 6.41--22.69 seconds for policy training; the long tail reflects
variable image/token lengths. Policy training accounted for essentially 100%
of non-checkpoint steps, and W&B recorded 99% GPU utilization at completion.
Adapter-only checkpoint writes took about 0.8 seconds. Each retained adapter
is about 63 MiB on disk (the safetensors payload is 54,296,712 bytes). The run
consumed 25,600 examples and finished in 2,228 seconds of W&B runtime (about
37.1 minutes); final training loss was 0.1315.

## Microbatch and Backend Findings

| Trial | Outcome |
| --- | --- |
| NeMo-RL v0.5 full SFT, mb1 | roughly 1,895 s/update and 46--48% GPU; impractical |
| NeMo-RL v0.6 full SFT, mb4 | completed 200 steps; roughly 8--12 s/update and near 99% GPU |
| NeMo-RL v0.6 full SFT, mb8 | OOM on a variable-length batch at step 33 |
| NeMo-RL v0.6 LoRA, mb16 probe | stable for 2 steps; 10.76 s/update, 100% GPU, about 27.4 GiB sampled |
| NeMo-RL v0.6 LoRA, mb16 campaign | converged quickly, then OOM in log-softmax at step 120 |
| NeMo-RL v0.6 LoRA, mb8 campaign | completed all 200 steps; 9.47 s mean training time, 99% final GPU utilization |

The decisive speedup came from moving to the NeMo-RL v0.6 DTensor VLM path,
not from changing the base model. LoRA then reduced checkpoint cost and made
the requested LR practical. The microbatch search also demonstrates why a
short probe is insufficient for variable-size images: both full-SFT mb8 and
LoRA mb16 looked healthy initially but failed on later long batches.

The only compatibility change applied inside the v0.6 container registers
the `qwen3_vl` model type with the existing image-to-text AutoModel factories.
NeMo-RL v0.6 otherwise handles the multimodal tensors natively.

## Reproduction Assets

- `run_sft_v06_docker.sh`: official v0.6 Docker launcher wrapper.
- `run_sft_v05_docker.sh`: common isolated launcher, now supporting an empty
  broad patch path so the v0.6-native VLM implementation is retained.
- `dtensor_qwen3_vl_v06.patch`: minimal Qwen3-VL factory registration.
- `vlm_sft-qwen3-vl-2b-instruct-1n1g-dtensor-lora-all-linear-r8-lr1e4-b128-mb8-200-v06.v1.yaml`:
  selected robust recipe.
- The mb16 LoRA and mb4 full-SFT configurations are retained as controlled
  comparison artifacts.

The selected run directory is
`/data/ephemeral/nemo-rl/ubuntu/nemo-rl-auto-research/20260903-qwen3-vl-2b-dtensor-v06/sft-2b-lora-all-linear-r8-lr1e4-b128-mb8-200-v06`.
