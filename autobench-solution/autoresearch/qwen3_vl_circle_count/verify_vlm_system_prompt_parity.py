#!/usr/bin/env python3
"""Prove that NeMo-RL VLM preprocessing matches the deployment chat prompt."""

import argparse
import json
from pathlib import Path

import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.multimodal_utils import resolve_to_image
from nemo_rl.data.processors import vlm_hf_data_processor
from nemo_rl.models.policy import TokenizerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    args = parser.parse_args()

    processor = get_tokenizer(
        TokenizerConfig(
            name=args.model,
            chat_template="default",
            chat_template_kwargs={"enable_thinking": False},
        ),
        get_processor=True,
    )
    row = json.loads(args.data.read_text().splitlines()[0])
    datum = vlm_hf_data_processor(
        {"raw": json.dumps(row), "task_name": "circle-count"},
        TaskDataSpec(
            task_name="circle-count", system_prompt_file=args.system_prompt
        ),
        processor,
        max_seq_length=2048,
        idx=0,
    )

    user_content = row["responses_create_params"]["input"][1]["content"]
    image_url = next(x["image_url"] for x in user_content if x["type"] == "input_image")
    question = next(x["text"] for x in user_content if x["type"] == "input_text")
    messages = [
        {"role": "system", "content": args.system_prompt.read_text().strip()},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": question},
            ],
        },
    ]
    expected_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    expected_tokens = processor(
        text=expected_text,
        images=[resolve_to_image(image_url)],
        return_tensors="pt",
    )["input_ids"][0]

    actual_tokens = datum["message_log"][0]["token_ids"]
    assert datum["vllm_content"] == expected_text, "vLLM prompt text differs"
    assert torch.equal(actual_tokens, expected_tokens), "policy token IDs differ"
    assert "<|im_start|>system" in expected_text
    assert "<think>" not in expected_text
    print(
        json.dumps(
            {
                "text_equal": True,
                "token_ids_equal": True,
                "tokens": len(actual_tokens),
                "has_system_turn": True,
                "thinking_disabled": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
