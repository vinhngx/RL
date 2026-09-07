# Final Report — NeMo-RL campaign on Qwen3-VL-2B-Instruct (circle-count gym)

Date range: 2026-09-02 → 2026-09-08 · Host: 1x NVIDIA L40S (46 GB), driver 565.57.01
Objective: train Qwen/Qwen3-VL-2B-Instruct to high accuracy in the NeMo-Gym
`circle_count` environment; also: set up the NeMo-RL Docker stack, smoke-test
VLM RL, and document the whole path.

## Headline result

**0.945 accuracy** on the held-out 200-task circle_count gym profile set
(out-of-box: 0.365; +58pp / ~2.6x). Pipeline: SFT from the original checkpoint
(40k synthetic rows, 12 answer templates, bs=128, LoRA, 0.64 epochs → 0.900),
then stacked GRPO windows with fresh LoRA on the merged SFT model (lr 5e-5,
16 gens/prompt, peak pooled by val accuracy): dipped and rebounded to val
0.9453 on the in-training set; gym-verified at **0.945**.

| Candidate | Gym accuracy (200 tasks, temp 1.0) |
|---|---|
| Out-of-box baseline | 0.365 |
| GRPO best (native env shim, step_120) | 0.615 |
| SFT 6k rows / 200 steps | 0.865–0.885 |
| SFT 40k rows / step_200 (of 400) | 0.900 |
| SFT 40k rows / step_400 (overfit) | 0.805 |
| **SFT → GRPO cascade (final)** | **0.945** |

Error signature of the final model: counts ≤7 → 95% correct; counts >7 → 90%
(GRPO-on-SFT is what lifted the hard band from 48% to 90%).

## What had to be solved first (infrastructure)

1. **Driver/CUDA blocker.** Host driver 565.57.01 (CUDA 12.7, apt-held by the
   provisioner) cannot run `main`'s stack (torch 2.11+cu130, CUDA 13.2 base).
   `cuInit` 803 + forward-compat shim fails. **Solution: run NeMo-RL v0.6.0**
   (torch 2.10+cu129, vLLM 0.17.1, transformers 5.3.0) — CUDA 12.9 user space
   works via minor-version forward compat. Image: `nemo-rl:v060-smoke`
   (SKIP_SGLANG_BUILD=1; 34 GB).
2. **Qwen3-VL not loadable on FSDP/dtensor workers.** `AUTOMODEL_FACTORY` in
   `nemo_rl/models/policy/utils.py` lacked `qwen3_vl` → fell back to
   AutoModelForCausalLM → ValueError. Fix committed (this branch) mapping
   `qwen3_vl`/`qwen3_vl_moe` → `AutoModelForImageTextToText` (HF + NeMo
   automodel variants). The fix was validated end-to-end by training runs.
3. **Operational fixes** (details in APPROACH.md §4): Ray AF_UNIX 107-byte
   socket limit (short `/ray` mount), single-GPU 46 GB memory (LoRA), v0.6.0
   `validate()` unbound-variable bug when `max_val_samples < val_batch_size`,
   `data:` URL prefix handling for base64 images, manual LoRA merge before
   serving eval (vLLM ignores vision-tower adapters), moving docker's
   data-root and `/ephemeral` content to the `/data` volume.

## Campaign narrative (auto-research ledger)

Phase 0 (baseline profiling) — gym-CLI rollouts of the untrained model,
`ng_reward_profile` on 200 tasks: 0.365.

GRPO arc (native env shim mirroring the gym verifier 1:1, because VLM+gym
training plumbing exists only on main): LoRA GRPO, 8→16 generations/prompt,
lr 2e-5 → 1e-4, easy-curriculum warm start → best 0.615. Two failures:
hard-band-only GRPO collapses (all-wrong groups = zero advantage); dynamic
sampling added 20% overhead with no gain.

SFT arc (user recipe: bs=128 × 200 steps): 6k templates → **0.885**.
Scale-up 40k rows × 12 templates, 400 steps: step_200 → **0.900** (best),
step_300 → 0.815, step_400 → 0.805. Observations: more data + fewer epochs
outperforms the inverse; the run peaks ~0.64 epochs and overfits thereafter.
Hard-band continuation SFT (fresh LoRA on the merged weights, 11-20 circles,
lr 5e-5) regressed to 0.435 (easy skill destroyed) — discarded.

## Known ceiling & next lever

The hard band (10-20 circles) stays under ~50% across all recipes — pure
"output the count" supervision is data-inefficient for dense counting. The
most promising un-tried lever: **enumerating/grounded supervision** (assistant
lists target circle positions, then answers `\boxed{N}`), or
vision-encoder-weight training at higher capacity.

## Artifact map

- Repo (branch `kimi-k3/circle-count-vlm-support`): `kimi-k3-solution/`
  (`APPROACH.md`, `REPRODUCE.md`, this report), circle-count code
  (datasets/env/reward/prompt/configs); worktree branches
  `autoresearch/2026-09-02-circle-count-qwen3vl/*` hold the per-experiment
  recipe commits; campaign ledger: `session/20260902_131517/`.
- Volume `/data` (symlinked as `/ephemeral/nemo-rl`): datasets
  (train/val/profile jsonl), all run logs under `circle-count-gym/results/`,
  GRPO/SFT adapters under `ckpts*/`, merged eval weights under `merged-*`,
  Docker build logs, and the shareable artifact
  `circle-count-gym/hf-ckpt-sft200.tar.gz` (HF checkpoint of the 0.885 model,
  sha256 44a945d8...).
