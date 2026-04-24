#!/usr/bin/env python3
"""Create Circle Click JSONL files that require the click tool."""

import json
from pathlib import Path


SRC_DIR = Path("tutorial/data/circle_click")
DST_DIR = Path("tutorial/data/circle_click_forced_tool")


def rewrite_jsonl(name: str) -> None:
    src = SRC_DIR / name
    dst = DST_DIR / name
    DST_DIR.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as infile, dst.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            row = json.loads(line)
            row["responses_create_params"]["tool_choice"] = {
                "type": "function",
                "name": "click",
            }
            outfile.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    rewrite_jsonl("train.jsonl")
    rewrite_jsonl("validation.jsonl")


if __name__ == "__main__":
    main()
