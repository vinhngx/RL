#!/usr/bin/env python3
"""Create Circle Click JSONL files that require the click tool."""

import json
import argparse
from pathlib import Path


SRC_DIR = Path("tutorial/data/circle_click")
DST_DIR = Path("tutorial/data/circle_click_forced_tool")


def rewrite_jsonl(src_dir: Path, dst_dir: Path, name: str) -> None:
    src = src_dir / name
    dst = dst_dir / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as infile, dst.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            row = json.loads(line)
            row["responses_create_params"]["tool_choice"] = {
                "type": "function",
                "name": "click",
            }
            row.setdefault(
                "agent_ref",
                {"type": "responses_api_agents", "name": "circle_click_simple_agent"},
            )
            outfile.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Circle Click JSONL files that require the named click tool."
    )
    parser.add_argument("--input-dir", type=Path, default=SRC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DST_DIR)
    args = parser.parse_args()

    rewrite_jsonl(args.input_dir, args.output_dir, "train.jsonl")
    rewrite_jsonl(args.input_dir, args.output_dir, "validation.jsonl")


if __name__ == "__main__":
    main()
