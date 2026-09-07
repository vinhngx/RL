# Reproduction Guide — circle-count accuracy 0.885 (SFT recipe)

Highest **verified** accuracy in this campaign: **0.885** on the NeMo-Gym
circle_count verifier (200 held-out tasks, temp 1.0), from SFT of
`Qwen/Qwen3-VL-2B-Instruct` on synthetic gym-generated data.
(OOB baseline: 0.365. GRPO best: 0.615. A 40k-row SFT scale-up is running and
may supersede this.)

Produced on: 1x NVIDIA L40S (46 GB), host driver 565.57.01, image
`nemo-rl:v060-smoke`. All assets use `brev` paths as seen inside containers.

## 0. Stack prerequisites

```bash
# NeMo-RL v0.6.0 with the qwen3_vl fix + circle-count assets
cd /home/ubuntu/RL
git worktree add ~/RL-ref-v0.6.0 v0.6.0
git -C ~/RL-ref-v0.6.0 submodule update --init --recursive --depth 1
git -C ~/RL-ref-v0.6.0 checkout -b circle-count kimi-k3/circle-count-vlm-support

# (this host only: Gym does not have circle_count in the v0.6.0 pin; port it)
cp -r /home/ubuntu/RL/3rdparty/Gym-workspace/Gym/resources_servers/circle_count \
      ~/RL-ref-v0.6.0/3rdparty/Gym-workspace/Gym/resources_servers/

# Build the image (SKIP_SGLANG saves ~30 min; ~2-3 h total on 8 vCPU)
cd ~/RL-ref-v0.6.0
docker buildx build --build-context nemo-rl=. --build-arg SKIP_SGLANG_BUILD=1 \
  --tag nemo-rl:v060-smoke -f docker/Dockerfile --load .
```

> Version recap: v0.6.0 = torch 2.10.0+cu129 / vLLM 0.17.1 / transformers 5.3.0;
> runs on driver 565 via CUDA-12 minor compat. `main` (cu130) does NOT run here.

## 1. Generate the synthetic SFT data

Data generator: `gen_sft.py`, committed at
[`scripts/gen_sft.py`](scripts/gen_sft.py) in this folder (it wraps
`generate_data.make_example` from the gym server with templated answers
ending in `\boxed{N}`). Copy the `scripts/` dir onto the `/brev` mount before
launching containers, e.g.:

```bash
mkdir -p /ephemeral/nemo-rl/ubuntu/circle-count-gym
cp -r /home/ubuntu/RL/kimi-k3-solution/scripts/* /ephemeral/nemo-rl/ubuntu/circle-count-gym/
```

(All runner scripts here — `sft_run.sh`, `train_run.sh`, `profile_run.sh`,
`merge_lora.py`, `merge_and_eval.sh`, `reward_test.py` — are committed in
`kimi-k3-solution/scripts/` and expect that layout.)

```bash
docker run --rm -v /ephemeral/nemo-rl/ubuntu:/brev -v ~/RL-ref-v0.6.0:/opt/nemo-rl \
  nemo-rl:v060-smoke bash -c '
    export HF_HOME=/brev/cache/huggingface
    cd /opt/nemo-rl
    uv run python /brev/circle-count-gym/gen_sft.py 6000 /brev/circle-count-gym/data/sft-train.jsonl 40000
    uv run python /brev/circle-count-gym/gen_sft.py 128  /brev/circle-count-gym/data/sft-val.jsonl  46000'
```

The gym profile set used for scoring: seeds 10000+ (disjoint):
```bash
uv run python ~/RL-ref-v0.6.0/3rdparty/Gym-workspace/Gym/resources_servers/circle_count/generate_data.py \
  --n 200 --seed-offset 10000 --out /brev/circle-count-gym/data/profile.jsonl
```

## 2. Run SFT (200 steps, BS=128, from the original checkpoint)

Config: `examples/configs/sft_circle_count_qwen3vl_2B.yaml` (branch
`kimi-k3/circle-count-vlm-support`; dtensor backend + LoRA dim 8, bs 128,
seq 2048, lr 1e-4, activation checkpointing).

```bash
docker run --rm --gpus all --ipc=host \
  -v /ephemeral/nemo-rl/ubuntu:/brev \
  -v /ephemeral/nemo-rl/ubuntu/circle-count-gym/ray:/ray \
  -v ~/RL-ref-v0.6.0:/opt/nemo-rl \
  nemo-rl:v060-smoke bash /brev/circle-count-gym/sft_run.sh sft.max_num_epochs=5
```

Need 6000/128 ≈ 46.9 steps/epoch ⇒ 5 epochs capped at 200 steps
(~2.4 h). LoRA adapter checkpoints land in
`/brev/circle-count-gym/ckpts-sft/Qwen/Qwen3-VL-2B-Instruct/step_{100,150,200}`.

Key run facts (this instance): train loss 1.21 → 0.05; val_loss 0.13 → 0.05.

## 3. Merge adapter and evaluate with the gym verifier

vLLM can't serve vision-tower LoRA — merge first (`merge_lora.py` under
`/brev/circle-count-gym/`):

```bash
docker run --rm -v /ephemeral/nemo-rl/ubuntu:/brev -v ~/RL-ref-v0.6.0:/opt/nemo-rl \
  nemo-rl:v060-smoke bash -c '
    export HF_HOME=/brev/cache/huggingface; cd /opt/nemo-rl
    CKPT=/brev/circle-count-gym/ckpts-sft/Qwen/Qwen3-VL-2B-Instruct/step_200/policy/weights/model \
    OUT=/brev/circle-count-gym/merged-sft200 \
    uv run python /brev/circle-count-gym/merge_lora.py'
```

Then serve + profile (single shot): 

```bash
docker run --rm --gpus all --ipc=host \
  -v /ephemeral/nemo-rl/ubuntu:/brev -v ~/RL-ref-v0.6.0:/opt/nemo-rl \
  nemo-rl:v060-smoke bash -c '
    CKPT=/brev/circle-count-gym/merged-sft200 TAG=sft200 \
    bash /brev/circle-count-gym/merge_and_eval.sh'
python3 -c "
import json
rows=[json.loads(l) for l in open('/ephemeral/nemo-rl/ubuntu/circle-count-gym/evals/sft200/rollouts.jsonl')]
print('accuracy:', sum(r.get(\"reward\",0) for r in rows)/len(rows))"
```

Expected: **≈ 0.88–0.89** (200 samples, temp 1.0, ±2–3pp sampling noise).
Error split on this run: counts ≤7 → 94.7% correct; counts >7 → 51.7%.

## 4. Notes / gotchas (details in APPROACH.md §4)

- Mount Ray tmp at a short path (`-v .../ray:/ray`, `RAY_TMPDIR=/ray`).
- Inside the container everything runs from `/opt/nemo-rl` (the mounted
  worktree); `/brev` = `/ephemeral/nemo-rl/ubuntu`.
- The checkpoint save requires at least one val pass; with
  `metric_name: val:val_loss` align `save_period` to `val_period`.
- To eval an adapter directly with different checkpoints, pass `CKPT=...` to
  `merge_and_eval.sh`; it merges if the merged dir is absent.
