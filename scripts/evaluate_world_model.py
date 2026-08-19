#!/usr/bin/env python3
"""Evaluate a Stage-A latent critic without running the simulator."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

PERSONAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERSONAL_ROOT / "src"))
from openpi_libero_reproduction.world_model import LatentChangeCritic
from openpi_libero_reproduction.world_model_data import (
    LatentTransitionDataset,
    collate_latent_transitions,
    split_episode_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=50)
    return parser.parse_args()


def binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    """Return exact rank-based AUROC, or None when only one class is present."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(targets) != len(scores):
        raise ValueError("targets and scores must have equal length")
    positives = targets == 1
    negatives = targets == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if not positive_count or not negative_count:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = ranks[positives].sum()
    return float((rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def calibration_metrics(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if len(targets) != len(probabilities) or not len(targets):
        raise ValueError("calibration inputs must be non-empty and have equal length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(targets[mask].mean()))
    return {
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "ece_10bin": float(ece),
        "mean_probability": float(probabilities.mean()),
        "positive_rate": float(targets.mean()),
    }


def _loader(paths: list[pathlib.Path], batch_size: int, workers: int) -> DataLoader | None:
    if not paths:
        return None
    dataset = LatentTransitionDataset(paths)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_latent_transitions,
    )


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_split(model: LatentChangeCritic, loader: DataLoader | None, device: str, episode_count: int) -> dict:
    if loader is None:
        return {"episode_count": episode_count, "transition_count": 0}
    latent_squared_errors: list[np.ndarray] = []
    terminal_logits: list[np.ndarray] = []
    terminal_targets: list[np.ndarray] = []
    success_logits: list[np.ndarray] = []
    success_targets: list[np.ndarray] = []
    transition_count = 0
    model.eval()
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model(batch["current_latent"], batch["state"], batch["action_chunk"], batch["text_features"])
        latent_squared_errors.append(
            (output.predicted_future_latent - batch["future_latent"]).pow(2).detach().cpu().numpy().reshape(-1)
        )
        terminal_logits.append(output.terminal_logit.detach().cpu().numpy())
        terminal_targets.append(batch["terminal_target"].detach().cpu().numpy())
        success_logits.append(output.success_logit.detach().cpu().numpy())
        success_targets.append(batch["success_target"].detach().cpu().numpy())
        transition_count += len(batch["current_latent"])

    terminal_logit = np.concatenate(terminal_logits)
    terminal_target = np.concatenate(terminal_targets).astype(np.int64)
    success_logit = np.concatenate(success_logits)
    success_target = np.concatenate(success_targets).astype(np.int64)
    terminal_probability = 1.0 / (1.0 + np.exp(-np.clip(terminal_logit, -60.0, 60.0)))
    success_probability = 1.0 / (1.0 + np.exp(-np.clip(success_logit, -60.0, 60.0)))
    return {
        "episode_count": episode_count,
        "transition_count": transition_count,
        "latent_mse": float(np.mean(np.concatenate(latent_squared_errors))),
        "terminal_auroc": binary_auroc(terminal_target, terminal_probability),
        "terminal": calibration_metrics(terminal_target, terminal_probability),
        "success_auroc": binary_auroc(success_target, success_probability),
        "success": calibration_metrics(success_target, success_probability),
    }


@torch.no_grad()
def measure_latency(model: LatentChangeCritic, loader: DataLoader | None, device: str, warmup: int, repeats: int) -> dict | None:
    if loader is None:
        return None
    batch = _to_device(next(iter(loader)), device)
    model.eval()
    for _ in range(warmup):
        model(batch["current_latent"], batch["state"], batch["action_chunk"], batch["text_features"])
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(batch["current_latent"], batch["state"], batch["action_chunk"], batch["text_features"])
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
    batch_size = len(batch["current_latent"])
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "batch_ms_mean": float(np.mean(durations) * 1000.0),
        "batch_ms_p50": float(np.percentile(durations, 50) * 1000.0),
        "per_sample_us_mean": float(np.mean(durations) * 1_000_000.0 / batch_size),
    }


def _episode_summary(paths: Iterable[pathlib.Path], data_dir: pathlib.Path) -> dict[str, int]:
    del data_dir
    paths = list(paths)
    success_count = 0
    transition_count = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            success_count += int(bool(archive["episode_success"]))
            transition_count += len(archive["current_latent"])
    return {"episodes": len(paths), "successful_episodes": success_count, "transitions": transition_count}


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.latency_warmup < 0 or args.latency_repeats <= 0:
        raise ValueError("batch size/workers must be non-negative and latency repeats must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    paths = sorted(args.data_dir.glob("*.npz"))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"no latent shards found in {args.data_dir}")
    train_paths, val_paths, test_paths = split_episode_shards(paths)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    if not config:
        raise ValueError("checkpoint does not contain a critic config")
    model = LatentChangeCritic(**config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    loaders = {
        "train": _loader(train_paths, args.batch_size, args.num_workers),
        "validation": _loader(val_paths, args.batch_size, args.num_workers),
        "test": _loader(test_paths, args.batch_size, args.num_workers),
    }
    split_paths = {"train": train_paths, "validation": val_paths, "test": test_paths}
    report = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "device": args.device,
        "split_rule": "blake2b filename hash; episode shards never cross splits",
        "latent_mse_scope": "within-representation only; latent scales may differ across encoders",
        "latency_scope": "critic forward only on precomputed latents; visual encoder excluded",
        "splits": {},
    }
    for name in ("train", "validation", "test"):
        summary = _episode_summary(split_paths[name], args.data_dir)
        summary["metrics"] = evaluate_split(model, loaders[name], args.device, len(split_paths[name]))
        report["splits"][name] = summary
    latency_loader = loaders["test"] or loaders["validation"] or loaders["train"]
    report["latency"] = measure_latency(
        model,
        latency_loader,
        args.device,
        args.latency_warmup,
        args.latency_repeats,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
