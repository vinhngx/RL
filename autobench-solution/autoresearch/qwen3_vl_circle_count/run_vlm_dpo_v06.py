#!/usr/bin/env python3
"""NeMo-RL v0.6 DPO entrypoint that loads an AutoProcessor for VLM data."""

import argparse
import pprint

from omegaconf import OmegaConf

from nemo_rl.algorithms.dpo import MasterConfig, dpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_preference_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal DPO training")
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    pprint.pprint(config)
    config["logger"]["log_dir"] = get_next_experiment_dir(
        config["logger"]["log_dir"]
    )

    init_ray()
    processor = get_tokenizer(config["policy"]["tokenizer"], get_processor=True)
    dataset, validation = setup_preference_data(processor, config["data"])
    (
        policy,
        _cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        dpo_save_state,
        master_config,
    ) = setup(config, processor, dataset, validation)
    dpo_train(
        policy,
        train_dataloader,
        val_dataloader,
        processor,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        dpo_save_state,
    )


if __name__ == "__main__":
    main()
