#!/usr/bin/env python3
"""Compute paired episode outcomes for a baseline/world-model ablation."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition-root",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="one or more transition roots to merge before pairing",
    )
    parser.add_argument("--output-csv", type=pathlib.Path, default=None)
    parser.add_argument("--baseline-controller", default="baseline")
    parser.add_argument("--world-model-controller", default="world_model")
    return parser.parse_args()


def exact_mcnemar_p(baseline_only: int, world_model_only: int) -> float:
    """Two-sided exact McNemar p value conditioned on discordant pairs."""

    if baseline_only < 0 or world_model_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = baseline_only + world_model_only
    if not discordant:
        return 1.0
    tail = min(baseline_only, world_model_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def load_outcomes(directory: pathlib.Path) -> dict[tuple[str, int, int, int], bool]:
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode shards found in {directory}")
    outcomes: dict[tuple[str, int, int, int], bool] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            key = (
                str(archive["suite"]),
                int(archive["task_id"]),
                int(archive["episode_idx"]),
                int(archive["seed"]),
            )
            if key in outcomes:
                raise ValueError(f"duplicate episode key {key} in {directory}")
            outcomes[key] = bool(archive["episode_success"])
    return outcomes


def merge_outcomes(directories: list[pathlib.Path], controller: str) -> dict[tuple[str, int, int, int], bool]:
    merged: dict[tuple[str, int, int, int], bool] = {}
    for directory in directories:
        for key, value in load_outcomes(directory / controller).items():
            if key in merged:
                raise ValueError(f"duplicate episode key {key} across transition roots")
            merged[key] = value
    return merged


def paired_rows(baseline: dict, world_model: dict) -> list[dict]:
    if baseline.keys() != world_model.keys():
        baseline_missing = sorted(world_model.keys() - baseline.keys())[:5]
        world_model_missing = sorted(baseline.keys() - world_model.keys())[:5]
        raise ValueError(
            "episode keys do not match: "
            f"missing from baseline={baseline_missing}, missing from world_model={world_model_missing}"
        )
    suites = sorted({key[0] for key in baseline})
    rows = []
    for suite in suites + ["aggregate"]:
        keys = list(baseline) if suite == "aggregate" else [key for key in baseline if key[0] == suite]
        both_success = sum(baseline[key] and world_model[key] for key in keys)
        baseline_only = sum(baseline[key] and not world_model[key] for key in keys)
        world_model_only = sum(not baseline[key] and world_model[key] for key in keys)
        both_failure = len(keys) - both_success - baseline_only - world_model_only
        baseline_success = both_success + baseline_only
        world_model_success = both_success + world_model_only
        rows.append(
            {
                "suite": suite,
                "episodes": len(keys),
                "baseline_success": baseline_success,
                "world_model_success": world_model_success,
                "delta_pp": 100.0 * (world_model_success - baseline_success) / len(keys),
                "both_success": both_success,
                "baseline_only": baseline_only,
                "world_model_only": world_model_only,
                "both_failure": both_failure,
                "mcnemar_exact_p": exact_mcnemar_p(baseline_only, world_model_only),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    baseline = merge_outcomes(args.transition_root, args.baseline_controller)
    world_model = merge_outcomes(args.transition_root, args.world_model_controller)
    rows = paired_rows(baseline, world_model)
    fieldnames = list(rows[0])
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
