"""Datasets and deterministic episode splits for the latent critic stage."""

from __future__ import annotations

import hashlib
import pathlib
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from .transition_dataset import load_episode_shard
from .world_model import hashed_text_features


def split_episode_shards(
    paths: list[pathlib.Path], *, val_fraction: float = 0.2, test_fraction: float = 0.1
) -> tuple[list[pathlib.Path], list[pathlib.Path], list[pathlib.Path]]:
    """Split by stable filename hash so transitions from one episode never cross splits."""

    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than one")
    train: list[pathlib.Path] = []
    val: list[pathlib.Path] = []
    test: list[pathlib.Path] = []
    for path in sorted(paths):
        bucket = int(hashlib.blake2b(path.name.encode(), digest_size=8).hexdigest(), 16) / 2**64
        if bucket < test_fraction:
            test.append(path)
        elif bucket < test_fraction + val_fraction:
            val.append(path)
        else:
            train.append(path)
    if paths and not train:
        train.append((test or val).pop())
    return train, val, test


class LatentTransitionDataset(Dataset):
    """Flatten precomputed episode latent shards into training examples."""

    def __init__(self, paths: list[pathlib.Path], *, text_dim: int = 256) -> None:
        self.paths = list(paths)
        self.text_dim = text_dim
        self._index: list[tuple[int, int]] = []
        self._lengths: list[int] = []
        expected_shapes: dict[str, tuple[int, ...]] | None = None
        for shard_id, path in enumerate(self.paths):
            with np.load(path, allow_pickle=False) as archive:
                if "current_latent" not in archive or "future_latent" not in archive:
                    raise ValueError(f"latent arrays are missing from {path}")
                length = len(archive["current_latent"])
                shapes = {
                    key: tuple(archive[key].shape[1:])
                    for key in ("current_latent", "future_latent", "state", "action_chunk")
                }
                if expected_shapes is None:
                    expected_shapes = shapes
                elif shapes != expected_shapes:
                    raise ValueError(f"latent/state/action shapes differ in {path}: {shapes} != {expected_shapes}")
            self._lengths.append(length)
            self._index.extend((shard_id, row) for row in range(length))

    def __len__(self) -> int:
        return len(self._index)

    @lru_cache(maxsize=8)
    def _load(self, shard_id: int) -> dict[str, np.ndarray]:
        return load_latent_shard(self.paths[shard_id])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        shard_id, row = self._index[index]
        shard = self._load(shard_id)
        prompt = str(shard["prompt"])
        return {
            "current_latent": torch.from_numpy(np.asarray(shard["current_latent"][row], dtype=np.float32)),
            "future_latent": torch.from_numpy(np.asarray(shard["future_latent"][row], dtype=np.float32)),
            "state": torch.from_numpy(np.asarray(shard["state"][row], dtype=np.float32)),
            "action_chunk": torch.from_numpy(np.asarray(shard["action_chunk"][row], dtype=np.float32)),
            "terminal_target": torch.tensor(float(shard["terminal_within_horizon"][row])),
            "success_target": torch.tensor(float(shard["episode_success"])),
            "prompt": prompt,
        }


def collate_latent_transitions(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor]:
    prompts = [str(item["prompt"]) for item in batch]
    result = {
        key: torch.stack([item[key] for item in batch])
        for key in (
            "current_latent",
            "future_latent",
            "state",
            "action_chunk",
            "terminal_target",
            "success_target",
        )
    }
    result["text_features"] = hashed_text_features(prompts, dimension=256)
    return result


def load_latent_shard(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    for key in ("current_latent", "future_latent", "state", "action_chunk", "terminal_within_horizon"):
        if key not in data:
            raise ValueError(f"missing {key} in latent shard {path}")
    return data
