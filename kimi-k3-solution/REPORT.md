# Smoke Test Report: RL training of Qwen3-VL-2B-Instruct in the NeMo-RL Docker image

Date: 2026-09-02 · Host: Brev instance, 1x NVIDIA L40S (46 GB), 8 vCPU, driver 565.57.01
Task: build the NeMo-RL Docker environment and smoke-test GRPO (RL) training of
`Qwen/Qwen3-VL-2B-Instruct` on a small dataset.

**Result: PASS** — 3 GRPO training steps + end-of-run validation, exit code 0,
in Docker image `nemo-rl:v060-smoke`.

---

## 1. Approach

Recipe base: `examples/configs/vlm_grpo_3B.yaml` (entrypoint
`examples/run_vlm_grpo.py`) with the CLEVR-CoGenT dataset
(`MMInstruction/Clevr_CoGenT_*` on Hugging Face, public) and `clevr-cogent`
reward env (format 0.2 / exact_alnum 0.8).

Smoke overrides (runner script attached; see §5): Qwen3-VL-2B model, batch 16
(4 prompts × 4 generations), seq len 1024 / 512 max new tokens, 3 steps,
8 validation samples, checkpointing and external loggers off.

## 2. Problems found and fixed

### 2.1 Host GPU driver too old for `main` (infra blocker)

`main` pins torch 2.11.0+cu130 / CUDA 13.2 base image (`cuda-dl-base:26.05`).
Host driver 565.57.01 only supports CUDA ≤ 12.7, and the stack cannot run:
torch reports "driver too old (found 12070)" and cuInit returns 803
(driver/runtime mismatch), even with the `cuda-compat-13-2` forward-compat shim
preloaded. The host driver stack is also pinned via apt-holds by instance
provisioning (Crusoe agents), so upgrading it on-box was off the table.

**Fix:** switched the source ref to **NeMo-RL v0.6.0**, whose stack is
cu129-based throughout: base image `cuda-dl-base:25.05-cuda12.9-devel`,
torch 2.10.0+cu129, vLLM 0.17.1, transformers 5.3.0. CUDA 12.x runtimes work
on driver ≥ 525.60.13 via minor-version compatibility, and Qwen3-VL needs
vLLM ≥ 0.11.0 / transformers ≥ 4.57.0, so v0.6.0 fits both constraints.
Checkout lives in a git worktree: `/home/ubuntu/RL-ref-v0.6.0`
(includes recursively-initialized submodules, needed by the Docker build's mcore extra).

### 2.2 Missing model-type mapping for Qwen3-VL (code fix, committed)

`resolve_model_class()` in `nemo_rl/models/policy/utils.py` maps HF
`model_type` → AutoModel class; `qwen3_vl` was absent, so DTensorPolicyWorkerV2
fell back to `AutoModelForCausalLM`, which transformers does *not* register for
`Qwen3VLConfig`:

```
ValueError: Unrecognized configuration class Qwen3VLConfig for this kind of
AutoModel: AutoModelForCausalLM.
```

**Fix (this commit):** added `qwen3_vl` and `qwen3_vl_moe` →
`AutoModelForImageTextToText` (and the `NeMoAutoModelForImageTextToText`
variants) to `AUTOMODEL_FACTORY`, mirroring the existing qwen2_5_vl entries.
Applied to both the v0.6.0 worktree (validated by this run) and `main`.

### 2.3 Ray AF_UNIX socket path too long

Initial run failed in Ray startup:
`MetricsHead failed to start ... validate_socket_filename failed: AF_UNIX path
length cannot exceed 107 bytes`, caused by a long `RAY_TMPDIR`.
**Fix:** bind-mount the experiment ray dir at the short in-container path `/ray`
and export `RAY_TMPDIR=/ray`.

### 2.4 Full fine-tune OOM on one 46 GB GPU

Step 1 (rollout + logprobs + train) passed, but step 2's loss computation OOM'd
(41 GB allocated by the policy worker: fp32 master weights + Adam states for
full FT of 2B params). **Fix for smoke scope:** enable LoRA
(`policy.dtensor_cfg.lora_cfg.enabled=true`), which leaves the optimizer state
tiny while exercising the identical rollout/refit/train pipeline. Full-FT of
Qwen3-VL-2B on 1xL40S would instead need CPU-offload or less colocated vLLM memory.

### 2.5 Validation config bug (v0.6.0)

`validate()` crashed with `UnboundLocalError: additional_metrics_to_report`
at `nemo_rl/algorithms/grpo.py:2314` because it is only assigned inside the
batch loop, and `max_batches = max_val_samples // val_batch_size` was
8 // 256 = 0. **Fix:** set `grpo.val_batch_size=8` for the smoke run
(the upstream fix would initialize the dict before the loop).

### 2.6 Disk pressure

Root volume (also backing `/ephemeral`) fell below 6 GB free mid-run (Ray
warned about object spilling). **Fix:** pruned 61 GB of BuildKit cache from the
superseded cu130 build; steady state now ~41 GB free.

## 3. Results

Model: `Qwen/Qwen3-VL-2B-Instruct` · Dataset: CLEVR-CoGenT train split
· Backend: FSDP2/dtensor (automodel), LoRA dim 8 · Generation: colocated vLLM.

| Step | Train loss | Avg reward | Mean gen len (tok) |
|------|-----------|------------|--------------------|
| 1    | 0.1040    | 0.2469     | 295.2              |
| 2    | 0.1303    | 0.2469     | 207.1              |
| 3    | -0.2561   | 0.2156     | 236.8              |

Validation (8 samples, valA split): **accuracy 0.2375**, avg response length
144.5 tokens, validation time 2.27 s. Sample dumps logged:
`val_data_step3.jsonl` and `train_data_step{1,2,3}.jsonl`.

Rewards/loss on CLEVR-CoGenT at 3 steps are far from converged; this smoke
verifies the full pipeline (VLM data → colocated rollout → refit →
logprob/advantage → train step → validation) end to end.

## 4. How to reproduce

```bash
# 1. v0.6.0 worktree (only needed to reproduce the exact verified env)
cd /home/ubuntu/RL
git worktree add ~/RL-ref-v0.6.0 v0.6.0
git -C ~/RL-ref-v0.6.0 submodule update --init --recursive --depth 1
#   apply the qwen3_vl factory patch (already present in this branch)

# 2. Build image (~2-3 h on 8 vCPU; TE compiles from source)
cd ~/RL-ref-v0.6.0
docker buildx build --build-context nemo-rl=. \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --tag nemo-rl:v060-smoke -f docker/Dockerfile --load .

# 3. Run (script on /ephemeral; /ray mount must stay short for AF_UNIX)
docker run --rm --gpus all --ipc=host \
  -v /ephemeral/nemo-rl/ubuntu:/brev \
  -v $EXP_DIR/ray:/ray \
  -v ~/RL-ref-v0.6.0:/opt/nemo-rl \
  nemo-rl:v060-smoke bash /brev/smoke-vlm/qwen3-vl-2b-grpo-clevr/run_smoke.sh
```

## 5. Artifacts

| Artifact | Path |
|----------|------|
| Runner script | `/ephemeral/nemo-rl/ubuntu/smoke-vlm/qwen3-vl-2b-grpo-clevr/run_smoke.sh` |
| Final run log | `/ephemeral/nemo-rl/ubuntu/smoke-vlm/qwen3-vl-2b-grpo-clevr/logs/run-v060-lora2.log` |
| Training/val samples | `/ephemeral/nemo-rl/ubuntu/smoke-vlm/qwen3-vl-2b-grpo-clevr/logs/nemo-rl/exp_006/` |
| Docker build log | `/ephemeral/nemo-rl/ubuntu/docker-logs/build-v060.log` |
| Shared HF/dataset cache | `/ephemeral/nemo-rl/ubuntu/cache` |
| Docker image | `nemo-rl:v060-smoke` (34.1 GB) |

Note: `/home/ubuntu/RL/.env` does not exist; it is only needed for gated
models/datasets or W&B runs.
