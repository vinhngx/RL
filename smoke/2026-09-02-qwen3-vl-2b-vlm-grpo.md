# Smoke test: GRPO on Qwen/Qwen3-VL-2B-Instruct (CLEVR-CoGenT) — 2026-09-02

**Result: PASS** (RUN_EXIT=0). 3 GRPO steps + end-of-run validation.

## Why v0.6.0 instead of main

Host driver is 565.57.01 (CUDA 12.7) and pinned via apt-holds (Crusoe/Brev
managed). `main`'s stack (torch 2.11+cu130, CUDA 13.2 base image) cannot run
(cuInit 803); cuda-compat shim fails too. Release **v0.6.0** pins
torch 2.10.0+cu129 / vllm 0.17.1 / transformers 5.3.0 on a CUDA 12.9 base,
which works on driver 565 via CUDA-12 minor-version compat and supports
Qwen3-VL (vllm >= 0.11, transformers >= 4.57).

## Changes on top of v0.6.0 (worktree `/home/ubuntu/RL-ref-v0.6.0`)

- `nemo_rl/models/policy/utils.py`: added `qwen3_vl` / `qwen3_vl_moe` to
  `AUTOMODEL_FACTORY` (both plain-HF and NeMo automodel tables), mapping to
  `AutoModelForImageTextToText`. Without this, DTensorPolicyWorkerV2 defaults
  to `AutoModelForCausalLM` and Qwen3VLConfig fails to load.

## Image / run

- Image: `nemo-rl:v060-smoke`, built from the v0.6.0 worktree with
  `--build-arg SKIP_SGLANG_BUILD=1` (vLLM included).
- Runner script: `/ephemeral/nemo-rl/ubuntu/smoke-vlm/qwen3-vl-2b-grpo-clevr/run_smoke.sh`
- Container: `docker run -v /home/ubuntu/RL-ref-v0.6.0:/opt/nemo-rl -v /ephemeral/nemo-rl/ubuntu:/brev -v $EXP_DIR/ray:/ray --gpus all --ipc=host nemo-rl:v060-smoke`
  - `/ray` mount must stay short: Ray AF_UNIX sockets have a 107-byte limit.
- Key overrides: model Qwen/Qwen3-VL-2B-Instruct, dataset clevr-cogent,
  seq len 1024, 4 prompts x 4 gens/step, 3 steps, `grpo.val_batch_size=8`
  (bug at v0.6.0 grpo.py:2314 — `additional_metrics_to_report` unbound when
  `max_val_samples // val_batch_size == 0`), LoRA enabled
  (`policy.dtensor_cfg.lora_cfg.enabled=true`) to fit full-FT OOM on 1xL40S.

## Results (exp_006)

| step | loss   | avg reward | mean gen len |
|------|--------|------------|--------------|
| 1    | 0.1040 | 0.2469     | 295.2        |
| 2    | 0.1303 | 0.2469     | 207.1        |
| 3    | -0.2561| 0.2156     | 236.8        |

Validation (8 samples, valA): accuracy 0.2375, avg response len 144.5 tokens.

## Artifacts (all under /ephemeral)

- Run log: `/ephemeral/nemo-rl/ubuntu/smoke-vlm/qwen3-vl-2b-grpo-clevr/logs/run-v060-lora2.log`
- Samples: `.../logs/nemo-rl/exp_006/{train_data_step1..3,val_data_step3}.jsonl`
- Shared HF/dataset cache: `/ephemeral/nemo-rl/ubuntu/cache`
- Docker build log: `/ephemeral/nemo-rl/ubuntu/docker-logs/build-v060.log`
