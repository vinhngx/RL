#!/usr/bin/env python3
"""Generate a balanced raw Circle Count curriculum by target count."""

import argparse
import importlib.util
import json
import random
from pathlib import Path


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("circle_count_generate_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Circle Count generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_count_for_seed(seed: int, colors: dict) -> int:
    """Replay the cheap pre-render RNG draws from make_example exactly."""
    rng = random.Random(seed)
    rng.randint(1000, 1000)
    rng.randint(30, 60)
    num_circles = rng.randint(5, 20)
    num_colors = rng.randint(2, 4)
    palette = rng.sample(list(colors.keys()), min(num_colors, len(colors)))
    color_names = [rng.choice(palette) for _ in range(num_circles)]
    target_color = rng.choice(palette)
    return sum(color == target_color for color in color_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", required=True)
    parser.add_argument("--per-count", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-seeds", type=int, default=1_000_000)
    args = parser.parse_args()

    generator = load_generator(args.generator)
    wanted = set(args.counts)
    buckets: dict[int, list[dict]] = {count: [] for count in args.counts}
    seeds_scanned = 0

    for seed in range(args.seed_start, args.seed_start + args.max_seeds):
        if all(len(rows) == args.per_count for rows in buckets.values()):
            break
        count = target_count_for_seed(seed, generator.COLORS)
        if count in wanted and len(buckets[count]) < args.per_count:
            example = generator.make_example(seed)
            example["curriculum_seed"] = seed
            buckets[count].append(example)
        seeds_scanned += 1
    else:
        missing = {count: args.per_count - len(rows) for count, rows in buckets.items()}
        raise RuntimeError(f"Curriculum incomplete after {args.max_seeds} seeds: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Interleave counts so every optimizer window sees the full difficulty range.
    with args.output.open("w") as destination:
        for row_index in range(args.per_count):
            for count in args.counts:
                destination.write(
                    json.dumps(buckets[count][row_index], separators=(",", ":")) + "\n"
                )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": args.per_count * len(args.counts),
                "counts": {str(count): len(buckets[count]) for count in args.counts},
                "seed_start": args.seed_start,
                "seeds_scanned": seeds_scanned,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
