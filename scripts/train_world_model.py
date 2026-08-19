#!/usr/bin/env python3
"""Train the lightweight latent-change and success critic on latent shards."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

PERSONAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERSONAL_ROOT / "src"))
from openpi_libero_reproduction.world_model import LatentChangeCritic, critic_loss
from openpi_libero_reproduction.world_model_data import (
    LatentTransitionDataset,
    collate_latent_transitions,
    split_episode_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--action-source",
        choices=("action_chunk", "selected_actions"),
        default="action_chunk",
        help="action sequence aligned with the future target",
    )
    return parser.parse_args()


def _run_epoch(model, loader, optimizer, device: str | torch.device | None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {"loss": 0.0, "latent_loss": 0.0, "terminal_loss": 0.0, "success_loss": 0.0}
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(batch["current_latent"], batch["state"], batch["action_chunk"], batch["text_features"])
        loss, metrics = critic_loss(
            output,
            future_latent=batch["future_latent"],
            terminal_target=batch["terminal_target"],
            success_target=batch["success_target"],
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        size = len(batch["current_latent"])
        count += size
        totals["loss"] += float(loss.detach()) * size
        for key, value in metrics.items():
            totals[key] += float(value) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    paths = sorted(args.data_dir.glob("*.npz"))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    train_paths, val_paths, test_paths = split_episode_shards(paths)
    if not train_paths:
        raise ValueError("no training shards found")
    train = LatentTransitionDataset(train_paths, action_key=args.action_source)
    val = LatentTransitionDataset(val_paths, action_key=args.action_source) if val_paths else None
    sample = train[0]
    latent_dim = int(sample["current_latent"].shape[0])
    state_dim = int(sample["state"].shape[0])
    action_horizon, action_dim = map(int, sample["action_chunk"].shape)
    model = LatentChangeCritic(
        latent_dim=latent_dim,
        state_dim=state_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_latent_transitions,
    )
    val_loader = (
        DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_latent_transitions)
        if val is not None and len(val)
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, args.device)
        val_metrics = _run_epoch(model, val_loader, None, args.device) if val_loader else {}
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}, sort_keys=True))
        score = val_metrics.get("loss", train_metrics["loss"])
        if score <= best_val:
            best_val = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "latent_dim": latent_dim,
                        "state_dim": state_dim,
                        "action_horizon": action_horizon,
                        "action_dim": action_dim,
                        "text_dim": 256,
                        "hidden_dim": args.hidden_dim,
                        "action_source": args.action_source,
                    },
                    "train_files": [path.name for path in train_paths],
                    "val_files": [path.name for path in val_paths],
                    "test_files": [path.name for path in test_paths],
                },
                args.output_dir / "critic.pt",
            )


if __name__ == "__main__":
    main()
