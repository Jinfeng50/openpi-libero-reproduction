#!/usr/bin/env python3
"""Summarize recorded hybrid gate metadata and sweep confidence thresholds.

The success rate reported for accepted transitions is observational: it is
not a counterfactual estimate of what would have happened under DGTE.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition-root",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="one or more transition roots to merge before analysis",
    )
    parser.add_argument("--controller", default="hybrid")
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--uncertainties", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45])
    return parser.parse_args()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    labels = ("p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100")
    points = np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100])
    return {label: float(value) for label, value in zip(labels, points)}


def load_gate_rows(directory: pathlib.Path) -> tuple[list[dict], int]:
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode shards found in {directory}")
    rows: list[dict] = []
    metadata_shards = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            required = {"gate_margin", "gate_uncertainty", "gate_candidate_count", "episode_success"}
            if not required.issubset(archive.files):
                continue
            metadata_shards += 1
            margin = np.asarray(archive["gate_margin"], dtype=np.float64)
            uncertainty = np.asarray(archive["gate_uncertainty"], dtype=np.float64)
            candidates = np.asarray(archive["gate_candidate_count"], dtype=np.int64)
            success = bool(archive["episode_success"])
            if not (len(margin) == len(uncertainty) == len(candidates)):
                raise ValueError(f"gate arrays have inconsistent lengths in {path}")
            valid = (candidates > 0) & np.isfinite(margin) & np.isfinite(uncertainty)
            rows.extend(
                {
                    "suite": str(archive["suite"]),
                    "episode_success": success,
                    "margin": float(margin[index]),
                    "uncertainty": float(uncertainty[index]),
                    "candidate_count": int(candidates[index]),
                }
                for index in np.flatnonzero(valid)
            )
    if not rows:
        raise ValueError(f"no valid gate metadata found in {directory}")
    return rows, metadata_shards


def merge_gate_rows(directories: list[pathlib.Path], controller: str) -> tuple[list[dict], int]:
    rows: list[dict] = []
    metadata_shards = 0
    for directory in directories:
        directory_rows, directory_shards = load_gate_rows(directory / controller)
        rows.extend(directory_rows)
        metadata_shards += directory_shards
    return rows, metadata_shards


def sweep_gate_thresholds(
    rows: list[dict], margins: list[float], uncertainties: list[float]
) -> list[dict[str, float | int | str | None]]:
    if any(value < 0 for value in margins + uncertainties):
        raise ValueError("thresholds must be non-negative")
    output: list[dict[str, float | int | str]] = []
    for margin_threshold in margins:
        for uncertainty_threshold in uncertainties:
            accepted = [
                row
                for row in rows
                if row["margin"] >= margin_threshold and row["uncertainty"] <= uncertainty_threshold
            ]
            successful = sum(row["episode_success"] for row in accepted)
            output.append(
                {
                    "margin_threshold": float(margin_threshold),
                    "uncertainty_threshold": float(uncertainty_threshold),
                    "accepted_transitions": len(accepted),
                    "acceptance_rate": float(len(accepted) / len(rows)),
                    "accepted_transition_success_rate": float(successful / len(accepted))
                    if accepted
                    else None,
                    "interpretation": "observational transition rate; not a counterfactual policy estimate",
                }
            )
    return output


def build_report(
    rows: list[dict], metadata_shards: int, margins: list[float], uncertainties: list[float]
) -> dict:
    margin_values = np.asarray([row["margin"] for row in rows], dtype=np.float64)
    uncertainty_values = np.asarray([row["uncertainty"] for row in rows], dtype=np.float64)
    return {
        "metadata_shards": metadata_shards,
        "gate_transition_count": len(rows),
        "margin_percentiles": _percentiles(margin_values),
        "uncertainty_percentiles": _percentiles(uncertainty_values),
        "threshold_sweep": sweep_gate_thresholds(rows, margins, uncertainties),
        "interpretation": "accepted-transition success is observational and not counterfactual",
    }


def main() -> None:
    args = parse_args()
    rows, metadata_shards = merge_gate_rows(args.transition_root, args.controller)
    report = build_report(rows, metadata_shards, args.margins, args.uncertainties)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
