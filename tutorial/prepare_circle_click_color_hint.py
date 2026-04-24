#!/usr/bin/env python3
"""Create Circle Click JSONL files with tool forcing and color-disambiguation hints."""

import argparse
import json
from pathlib import Path


SRC_DIR = Path("tutorial/data/circle_click_1000_clean_forced_tool")
DST_DIR = Path("tutorial/data/circle_click_1000_clean_forced_tool_color_hint")

SYSTEM_HINT = (
    " Carefully distinguish similarly colored circles before clicking. "
    "Purple means the violet/dark purple circle and is not pink, blue, or cyan."
)
PURPLE_USER_HINT = (
    " The requested target is purple: choose the violet/dark purple circle, "
    "not any pink, blue, or cyan circle."
)


def add_hint(row: dict) -> dict:
    params = row["responses_create_params"]
    messages = params["input"]

    system = messages[0]
    if system.get("role") == "system" and SYSTEM_HINT not in system["content"]:
        system["content"] += SYSTEM_HINT

    target_color = row.get("target_color")
    if target_color == "purple":
        user_content = messages[1]["content"]
        for item in user_content:
            if item.get("type") == "input_text" and PURPLE_USER_HINT not in item["text"]:
                item["text"] += PURPLE_USER_HINT
                break

    params["tool_choice"] = {"type": "function", "name": "click"}
    row.setdefault(
        "agent_ref",
        {"type": "responses_api_agents", "name": "circle_click_simple_agent"},
    )
    return row


def rewrite_jsonl(src_dir: Path, dst_dir: Path, name: str) -> None:
    src = src_dir / name
    dst = dst_dir / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as infile, dst.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            row = add_hint(json.loads(line))
            outfile.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Circle Click JSONL files with color-disambiguation hints."
    )
    parser.add_argument("--input-dir", type=Path, default=SRC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DST_DIR)
    args = parser.parse_args()

    rewrite_jsonl(args.input_dir, args.output_dir, "train.jsonl")
    rewrite_jsonl(args.input_dir, args.output_dir, "validation.jsonl")


if __name__ == "__main__":
    main()
