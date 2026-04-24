# Train a Small VLM on Circle Click with NeMo RL and NeMo Gym

This tutorial trains `Qwen/Qwen2.5-VL-3B-Instruct` with GRPO on the NeMo Gym `circle_click` task. The task shows a synthetic image with several colored circles and rewards the model only when it calls the `click(x, y)` tool inside the target circle.

The NeMo Gym checkout must include `resources_servers/circle_click`; as of this write-up the upstream Gym `main` commit I verified is `c465942` from 2026-04-24.

## 1. Update Gym

From the NeMo RL repo root:

```bash
export UV_CACHE_DIR=/ephemeral/uv-cache
export UV_PROJECT_ENVIRONMENT=/ephemeral/nemo-rl-circle-click-venv

git -C 3rdparty/Gym-workspace/Gym fetch --depth 1 origin main
git -C 3rdparty/Gym-workspace/Gym checkout FETCH_HEAD
test -f 3rdparty/Gym-workspace/Gym/resources_servers/circle_click/generate_data.py
```

## 2. Generate an Easy Curriculum Split

Start with an easier split than the upstream 1000 px examples. This lets a 3B VLM learn the tool-call/coordinate behavior quickly, then you can regenerate with larger images and smaller circles.

```bash
cd 3rdparty/Gym-workspace/Gym
export UV_CACHE_DIR=/ephemeral/uv-cache
export UV_PROJECT_ENVIRONMENT=/ephemeral/nemo-gym-circle-click-venv

uv run python resources_servers/circle_click/generate_data.py \
  --n 4000 \
  --out resources_servers/circle_click/data/train.jsonl \
  --seed-offset 0 \
  --img-size-min 384 --img-size-max 512 \
  --radius-min 38 --radius-max 70

uv run python resources_servers/circle_click/generate_data.py \
  --n 512 \
  --out resources_servers/circle_click/data/validation.jsonl \
  --seed-offset 100000 \
  --img-size-min 384 --img-size-max 512 \
  --radius-min 38 --radius-max 70
```

Prepare the data with NeMo Gym so every row has the `agent_ref` used by NeMo RL:

```bash
config_paths="resources_servers/circle_click/configs/circle_click.yaml,\
../../../tutorial/circle_click_train_datasets.yaml,\
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

uv run ng_prepare_data "+config_paths=[${config_paths}]" \
  +output_dirpath=../../../tutorial/data/circle_click \
  +mode=train_preparation

cd ../../..
```

The training config expects:

```text
tutorial/data/circle_click/train.jsonl
tutorial/data/circle_click/validation.jsonl
```

## 3. Train with NeMo RL

The included config is tuned for a single 48 GB GPU such as L40S/A6000/A40. For smaller GPUs, reduce `grpo.num_prompts_per_step`, `policy.generation_batch_size`, and `policy.generation.vllm_cfg.gpu_memory_utilization`.

```bash
export UV_CACHE_DIR=/ephemeral/uv-cache
export UV_PROJECT_ENVIRONMENT=/ephemeral/nemo-rl-circle-click-venv

uv run --extra nemo_gym --extra vllm \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config tutorial/circle_click_qwen2_5_vl_3b_1gpu.yaml
```

Watch TensorBoard:

```bash
uv run tensorboard --logdir logs/circle_click_qwen2_5_vl_3b
```

The main metric is `val:total_reward/mean`. Because Circle Click uses a binary reward, this is validation accuracy.

## 4. Train to High Accuracy

Use a staged curriculum:

1. Run the config above until `val:total_reward/mean >= 0.90`.
2. Regenerate train/validation at `--img-size-min 768 --img-size-max 1000 --radius-min 70 --radius-max 120`, resume from the best checkpoint, and train until validation is again at least `0.90`.
3. Regenerate at the full upstream difficulty, `--img-size-min 1000 --img-size-max 1000 --radius-min 60 --radius-max 150`, and train until the final validation accuracy is stable.

For the harder stages, keep the same command but override the checkpoint path or resume from the best checkpoint produced in `results/circle_click_qwen2_5_vl_3b`.

## 5. Sanity Check Rollouts

After training, inspect NeMo Gym response logs from the NeMo RL log directory. Successful examples should contain a `function_call` named `click` with integer `x` and `y`, and the full result should have `reward: 1.0` and `hit: true`.

You can also run a small standalone rollout collection against a served checkpoint:

```bash
cd 3rdparty/Gym-workspace/Gym
ng_run "+config_paths=[resources_servers/circle_click/configs/circle_click.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" &
ng_collect_rollouts \
  +agent_name=circle_click_simple_agent \
  +input_jsonl_fpath=resources_servers/circle_click/data/validation.jsonl \
  +output_jsonl_fpath=resources_servers/circle_click/data/validation_rollouts.jsonl \
  +limit=32
```

## Notes

`Qwen/Qwen2.5-VL-3B-Instruct` is the smallest VLM already represented in this NeMo RL tree, which makes it a conservative choice for this recipe. The upstream Circle Click README currently demonstrates `Qwen/Qwen3-VL-8B-Instruct`; switch to that model if you have more GPU memory and want stronger zero-shot tool use.
