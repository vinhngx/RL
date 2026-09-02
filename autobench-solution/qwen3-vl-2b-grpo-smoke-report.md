# Qwen3-VL-2B-Instruct GRPO Smoke-Test Report

## Outcome

NeMo-RL successfully completed one end-to-end GRPO optimizer step for
`Qwen/Qwen3-VL-2B-Instruct` on a single NVIDIA L40S. The smoke test exercised
multimodal rollout generation, reward calculation, advantage calculation,
policy and reference log-probability inference, the training forward/backward
pass, and the optimizer update. The process exited with status 0.

The run intentionally used one Geometry3K prompt and two generated responses.
It is a functional integration test, not a convergence or model-quality test.

## Test Environment

| Component | Value |
| --- | --- |
| Host GPU | NVIDIA L40S, 46,068 MiB |
| Host NVIDIA driver | 565.57.01 |
| Base container | `nvcr.io/nvidia/nemo-rl:v0.5.0` |
| Container CUDA | 12.9, using forward compatibility |
| Derived image | `nemo-rl:qwen3vl-smoke-cu129` |
| Derived image ID | `sha256:72d24540e248e8513329be97deab22fa22e0284acf31a32faaff8d2a43d1496d` |
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Algorithm | GRPO |
| Dataset | Geometry3K training split |

NeMo-RL v0.7.0 was not usable on this host because its CUDA 13.2 runtime
required a newer NVIDIA driver. NeMo-RL v0.5.0 provides the required Qwen3-VL
support while retaining a CUDA 12.9 runtime compatible with this machine.

## Reproducibility Files

- `docker/Dockerfile.qwen3vl-smoke-cu129` derives from NeMo-RL v0.5.0 and
  applies the Qwen3-VL attention-mask compatibility patch.
- `examples/smoke/run_qwen3_vl_2b_grpo_docker.sh` builds the image when needed,
  prepares Brev-safe storage, loads local credentials without printing them,
  and launches the one-step smoke test.
- `.gitignore` excludes the local `.env` secret store.

Run the test from the repository root:

```bash
./examples/smoke/run_qwen3_vl_2b_grpo_docker.sh
```

The launcher supports environment overrides such as:

```bash
RUN_ROOT=/ephemeral/nemo-rl/ubuntu/qwen3-vl-2b-smoke-rerun \
    ./examples/smoke/run_qwen3_vl_2b_grpo_docker.sh
```

## Smoke-Test Configuration

| Setting | Value |
| --- | --- |
| Prompts per step | 1 |
| Generations per prompt | 2 |
| Maximum steps | 1 |
| Global training batch | 2 |
| Training microbatch | 1 |
| Maximum total sequence length | 512 |
| Maximum generated tokens | 32 |
| Precision | bfloat16 |
| Training backend | Megatron-Core |
| Generation backend | vLLM 0.11.2 |
| Optimizer placement | CPU offload, fraction 1.0 |
| Activation checkpointing | Disabled |
| Validation | Disabled |
| Training checkpoint save | Disabled |
| W&B and TensorBoard | Disabled |

Activation checkpointing is disabled because the Qwen3-VL Megatron-Bridge
implementation passes a list through the recomputation checkpoint function,
while PyTorch's `save_for_backward` accepts tensors only. CPU optimizer offload
provided sufficient GPU memory headroom without recomputation.

## Compatibility Changes

NeMo-RL v0.5.0 constructs a four-dimensional causal attention mask for policy
log-probability and training forwards. Qwen3-VL's multimodal rotary-position
helper expects a two-dimensional padding mask, or `None` so it can construct an
all-valid mask internally. The derived Docker image sets the mask to `None` at
all three multimodal forward sites used by inference and training.

The container mounts the long Brev paths at short internal paths (`/cache` and
`/runstate`). This also prevents Ray Unix-domain socket paths from exceeding
the operating-system limit.

## Results

The successful step reported:

| Metric | Result |
| --- | ---: |
| Training loss | 0.0450 |
| Generation KL error | 0.0011 |
| Average reward | 0.0375 |
| Mean generation length | 20 tokens |
| Total step time | 29.16 seconds |
| End-to-end throughput | 0.07 samples/s/GPU |
| End-to-end token throughput | 10.84 tokens/s/GPU |
| Policy-training throughput | 24.90 tokens/s/GPU |

The authoritative log contains `SETUP COMPLETE`, `Step 1/1`, `Computing
logprobs`, `Training policy`, the metrics above, and `Max number of steps has
been reached`. It contains no traceback, and the Docker command exited with
status 0.

## Persisted Evidence

Heavy run state remains outside the Git checkout in accordance with the Brev
etiquette skill:

| Artifact | Absolute path |
| --- | --- |
| Shared cache | `/ephemeral/nemo-rl/ubuntu/cache` |
| Run root | `/ephemeral/nemo-rl/ubuntu/qwen3-vl-2b-smoke-20260902` |
| Full log | `/ephemeral/nemo-rl/ubuntu/qwen3-vl-2b-smoke-20260902/logs/run.log` |
| Rollout record | `/ephemeral/nemo-rl/ubuntu/qwen3-vl-2b-smoke-20260902/logs/exp_006/train_data_step1.jsonl` |
| Converted Megatron checkpoint | `/ephemeral/nemo-rl/ubuntu/qwen3-vl-2b-smoke-20260902/artifacts/megatron/Qwen/Qwen3-VL-2B-Instruct` |
| Earlier activation-checkpointing failure | `/ephemeral/nemo-rl/ubuntu/qwen3-vl-2b-smoke-20260902/logs/run.activation-checkpointing-failure.log` |

The converted Megatron checkpoint is the imported base-model checkpoint used
to initialize training. Because this smoke test disables checkpoint saving, it
does not claim to contain the post-update policy weights.

## Verification Performed

- Confirmed the final process exit code was 0.
- Confirmed one rollout batch, reward/advantage processing, log-probability
  inference, and policy training completed.
- Confirmed the rollout JSONL is non-empty (56,748 bytes).
- Confirmed the successful log contains no Python traceback.
- Compile-checked both patched Python modules inside the derived image.
- Confirmed the derived image remains installed and the smoke container exited.
- Syntax-checked the launcher with `bash -n` and checked the Git diff for
  whitespace errors.

## Scope and Limitations

This run proves that the selected NeMo-RL, Megatron-Core, Megatron-Bridge,
vLLM, Qwen3-VL, Docker, CUDA, and Geometry3K paths interoperate for a single RL
update on this host. A longer run is still required to evaluate stability,
checkpoint resumption, reward improvement, or training convergence.
