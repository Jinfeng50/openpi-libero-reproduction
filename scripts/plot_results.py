#!/usr/bin/env python3
"""Plot the completed official-checkpoint and fine-tuning LIBERO results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/figures/libero_sr_comparison.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np

    suites = ["Spatial", "Object", "Goal", "LIBERO-10"]
    official = [0.982, 0.988, 0.968, 0.926]
    finetuned = [0.984, 0.984, 0.968, 0.918]
    csv_path = Path(__file__).resolve().parents[1] / "experiments/results.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            row = rows[-1]
            finetuned = [float(row[key]) for key in ("spatial", "object", "goal", "libero_10")]

    x = np.arange(len(suites))
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    width = 0.36
    axis.bar(x - width / 2, official, width, label="Official checkpoint reproduced", color="#4c78a8")
    axis.bar(x + width / 2, finetuned, width, label="Full FT, 2x A800", color="#f58518")
    axis.set_ylim(0.85, 1.0)
    axis.set_ylabel("Success rate")
    axis.set_xticks(x, suites)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
