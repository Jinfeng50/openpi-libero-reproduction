#!/usr/bin/env python3
"""Precompute frozen visual latents for recorded transition shards."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

PERSONAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERSONAL_ROOT / "src"))
from openpi_libero_reproduction.transition_dataset import load_episode_shard
from openpi_libero_reproduction.world_model import FrozenResNet18Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-weights", choices=("default", "none"), default="default")
    return parser.parse_args()


def _encode(encoder, images: np.ndarray, wrist_images: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        image = torch.from_numpy(images[start : start + batch_size]).to(device)
        wrist = torch.from_numpy(wrist_images[start : start + batch_size]).to(device)
        outputs.append(encoder(image, wrist).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, encoder.output_dim), dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    paths = sorted(args.data_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode shards found in {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = FrozenResNet18Encoder(pretrained=args.encoder_weights == "default").to(args.device)
    for path in paths:
        data = load_episode_shard(path)
        latent = {
            "current_latent": _encode(encoder, data["image"], data["wrist_image"], args.batch_size, args.device),
            "future_latent": _encode(
                encoder, data["future_image"], data["future_wrist_image"], args.batch_size, args.device
            ),
        }
        for key in (
            "state",
            "action_chunk",
            "terminal_within_horizon",
            "episode_success",
            "prompt",
            "suite",
            "controller",
            "task_id",
            "episode_idx",
            "seed",
        ):
            if key in data:
                latent[key] = data[key]
        destination = args.output_dir / path.name
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **latent)
        temporary.replace(destination)
        print(f"{path.name}: {len(data['image'])} transitions -> {destination}")


if __name__ == "__main__":
    main()
