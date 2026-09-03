# NeMo-RL Training of Qwen3-VL-2B-Instruct — Software Stack, Version, and Fix

Companion to `REPORT.md` (the GRPO smoke test) and the circle-count campaign
(`session/20260902_131517/`). This note documents **what stack actually ran on
this host and why**, plus every fix needed to make Qwen3-VL train under it.

## 1. Software stack (the version that works here)

| Layer | Version |
|---|---|
| Host GPU / driver | 1x NVIDIA L40S 46GB, driver 565.57.01 (apt-held by the provisioner, ≤ CUDA 12.7) |
| Docker base | `nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04` |
| NeMo-RL | **v0.6.0** release tag (worktree `/home/ubuntu/RL-ref-v0.6.0`, branch `kimi-k3/qwen3vl-smoke-v060` + `autoresearch/*` chain) |
| torch / vLLM / transformers | 2.10.0+cu129, 0.17.1, 5.3.0 |
| Image | `nemo-rl:v060-smoke` (built with `--build-arg SKIP_SGLANG_BUILD=1`; ~34 GB) |
| Trainer | FSDP2/`dtensor` (automodel) policy workers + colocated vLLM generation on the same GPU |
| NeMo Gym | v0.6.0 submodule pin + ported `resources_servers/circle_count` |

## 2. Why v0.6.0 instead of main

- `main` pins torch 2.11+cu130 against a CUDA 13.2 base image. On driver
  565.57.01, everything in that stack fails: torch reports "driver too old
  (12070)", `cuInit` returns 803 inside the guarantee check, and installing
  `cuda-compat-13-2` + LD_PRELOADing `libcuda.so.1` does **not** rescue it.
- The host driver stack is pinned via apt-holds set by instance provisioning,
  so an in-box upgrade was ruled out.
- v0.6.0 is the newest release whose stack is CUDA 12.9 throughout
  (torch 2.10.0+cu129). CUDA 12.x minor-version compatibility lets it run on
  driver ≥ 525.60.13, and the base image enables Forward Compatibility mode:
  `Using CUDA 12.9 driver version 575.51.03 with kernel driver 565.57.01`.
- Qwen3-VL support needs vLLM ≥ 0.11 and transformers ≥ 4.57 — satisfied at
  v0.6.0 (0.17.1 / 5.3.0).

## 3. The fix (code)

`nemo_rl/models/policy/utils.py` — `resolve_model_class()` keys off HF
`model_type` in `AUTOMODEL_FACTORY`; `qwen3_vl` was missing, so VLM workers
fell back to `AutoModelForCausalLM`, which transformers does not register for
`Qwen3VLConfig`:

```
ValueError: Unrecognized configuration class Qwen3VLConfig for this kind of
AutoModel: AutoModelForCausalLM.
```

Fix (mirroring the existing `qwen2_5_vl` entries, both tables):

```python
"qwen3_vl": AutoModelForImageTextToText,
"qwen3_vl_moe": AutoModelForImageTextToText,
"qwen3_vl": NeMoAutoModelForImageTextToText,
"qwen3_vl_moe": NeMoAutoModelForImageTextToText,
```

Committed as `1debc36c6` on main (`/home/ubuntu/RL`) and rebuilt into the
v0.6.0 worktree. Tracked file `qwen3_vl_factory.patch` in this folder.

## 4. Auxiliary fixes the training pipeline needed on this host

| Symptom | Fix |
|---|---|
| Ray dashboard `MetricsHead failed to start` (AF_UNIX > 107 bytes) | bind-mount Ray tmp at the short in-container path `/ray`; set `RAY_TMPDIR=/ray` |
| GRPO step-2 OOM on 46 GB (full FT = fp32 master + Adam ≈ 28 GB) | enable LoRA (`policy.dtensor_cfg.lora_cfg.enabled=true`) for all runs |
| v0.6.0 `validate()` crashes when `max_val_samples < val_batch_size` (`additional_metrics_to_report` unbound, grpo.py:2314) | keep `val_batch_size ≤ max_val_samples` (smoke: 8/8) |
| Circle-count dataset `File name too long` | keep the full `data:image/png;base64,...` URL (do not strip the prefix — `resolve_to_image` keys off it) |
| Gym `ng_reward_profile` missing arg | v0.6.0 Gym needs `+materialized_inputs_jsonl_fpath=...` |
| vLLM rejects vision-tower LoRA (`visual.*` modules ignored; `/v1/load_lora_adapter` 404) | merge adapter into base weights manually (`W + (alpha/r)·B·A`, `merge_lora.py`) and serve the merged model for gym evals |
| Auto-resume picks *latest* ckpt, not *best* | park/shift aside non-best later checkpoints before warm-start phases; use top-k metric (val:accuracy) as fixture of truth |
| `timeout` on `docker run` doesn't kill the container | hard-stop by name with `docker rm -f` at the cap |

## 5. Results delivered on this stack

- GRPO smoke (CLEVR-CoGenT, 3 steps): PASS, end-to-end.
- Circle-count GRPO campaign: 0.365 → 0.615 gym accuracy (phase A lr5e-5 pool).
- Circle-count SFT (user recipe: bs 128 × 200 steps, synthetic data): **0.885**
  gym accuracy; easy band 94.7%, hard band 51.7%.

## 6. Pointers

- Artifacts: `/ephemeral/nemo-rl/ubuntu/{smoke-vlm,circle-count-gym}/`
- Docker build logs: `/ephemeral/nemo-rl/ubuntu/docker-logs/`
- Campaign ledger: `/home/ubuntu/RL/session/20260902_131517/`
- Worktree runtime: `/home/ubuntu/RL-ref-v0.6.0` (`autoresearch/2026-09-02-circle-count-qwen3vl/*` branches)
