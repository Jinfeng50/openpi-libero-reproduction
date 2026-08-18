"""Episode-level transition recording for latent world-model training."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


class EpisodeTransitionRecorder:
    """Collect replan-boundary transitions and atomically write one NPZ shard."""

    def __init__(
        self,
        output_dir: str | pathlib.Path,
        *,
        suite: str,
        controller: str,
        task_id: int,
        episode_idx: int,
        prompt: str,
        seed: int,
        replan_steps: int,
    ) -> None:
        if replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.output_dir = pathlib.Path(output_dir)
        self.suite = suite
        self.controller = controller
        self.task_id = int(task_id)
        self.episode_idx = int(episode_idx)
        self.prompt = str(prompt)
        self.seed = int(seed)
        self.replan_steps = int(replan_steps)
        self._records: list[dict[str, Any]] = []

    def add(
        self,
        *,
        image: np.ndarray,
        wrist_image: np.ndarray,
        future_image: np.ndarray,
        future_wrist_image: np.ndarray,
        state: np.ndarray,
        future_state: np.ndarray,
        action_chunk: np.ndarray,
        selected_actions: np.ndarray,
        executed_steps: int,
        start_step: int,
        future_step: int,
        terminal_within_horizon: bool,
    ) -> None:
        image = _validate_image("image", image)
        wrist_image = _validate_image("wrist_image", wrist_image)
        future_image = _validate_image("future_image", future_image)
        future_wrist_image = _validate_image("future_wrist_image", future_wrist_image)
        if image.shape != future_image.shape or wrist_image.shape != future_wrist_image.shape:
            raise ValueError("current and future image shapes must match")

        state = _validate_vector("state", state)
        future_state = _validate_vector("future_state", future_state)
        if state.shape != future_state.shape:
            raise ValueError("current and future state shapes must match")

        action_chunk = _validate_actions("action_chunk", action_chunk)
        selected_actions = _validate_actions("selected_actions", selected_actions)
        if action_chunk.shape[1] != selected_actions.shape[1]:
            raise ValueError("action dimensions must match")
        if len(selected_actions) != self.replan_steps:
            raise ValueError(
                f"selected_actions must contain {self.replan_steps} actions, got {len(selected_actions)}"
            )
        if not 0 < executed_steps <= self.replan_steps:
            raise ValueError("executed_steps must be in [1, replan_steps]")
        if future_step - start_step != executed_steps:
            raise ValueError("future_step - start_step must equal executed_steps")

        if self._records:
            first = self._records[0]
            for key, value in (
                ("image", image),
                ("wrist_image", wrist_image),
                ("state", state),
                ("action_chunk", action_chunk),
                ("selected_actions", selected_actions),
            ):
                if value.shape != first[key].shape:
                    raise ValueError(f"{key} shape changed within the episode")

        self._records.append(
            {
                "image": image.copy(),
                "wrist_image": wrist_image.copy(),
                "future_image": future_image.copy(),
                "future_wrist_image": future_wrist_image.copy(),
                "state": state.copy(),
                "future_state": future_state.copy(),
                "action_chunk": action_chunk.copy(),
                "selected_actions": selected_actions.copy(),
                "executed_steps": int(executed_steps),
                "start_step": int(start_step),
                "future_step": int(future_step),
                "terminal_within_horizon": bool(terminal_within_horizon),
            }
        )

    def finish(self, *, episode_success: bool) -> pathlib.Path | None:
        """Write the episode shard and return its path, or None if it is empty."""

        if not self._records:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self.suite}_{self.controller}_task_{self.task_id:02d}_"
            f"episode_{self.episode_idx:03d}_seed_{self.seed}.npz"
        )
        destination = self.output_dir / filename
        temporary = destination.with_suffix(".npz.tmp")
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "suite": np.asarray(self.suite),
            "controller": np.asarray(self.controller),
            "task_id": np.asarray(self.task_id, dtype=np.int32),
            "episode_idx": np.asarray(self.episode_idx, dtype=np.int32),
            "prompt": np.asarray(self.prompt),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "replan_steps": np.asarray(self.replan_steps, dtype=np.int32),
            "episode_success": np.asarray(bool(episode_success)),
        }
        for key in (
            "image",
            "wrist_image",
            "future_image",
            "future_wrist_image",
            "state",
            "future_state",
            "action_chunk",
            "selected_actions",
        ):
            arrays[key] = np.stack([record[key] for record in self._records])
        for key, dtype in (
            ("executed_steps", np.int32),
            ("start_step", np.int32),
            ("future_step", np.int32),
            ("terminal_within_horizon", np.bool_),
        ):
            arrays[key] = np.asarray([record[key] for record in self._records], dtype=dtype)

        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
        return destination


def load_episode_shard(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load and validate a recorder shard without enabling pickle."""

    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported transition schema in {path}")
    required = {
        "image",
        "wrist_image",
        "future_image",
        "future_wrist_image",
        "state",
        "future_state",
        "action_chunk",
        "selected_actions",
        "executed_steps",
        "terminal_within_horizon",
        "episode_success",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"missing arrays in {path}: {sorted(missing)}")
    count = len(data["image"])
    for key in required.difference({"episode_success"}):
        if len(data[key]) != count:
            raise ValueError(f"{key} has inconsistent transition count in {path}")
    return data


def _validate_image(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [height, width, 3]")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8")
    return np.ascontiguousarray(array)


def _validate_vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    return array


def _validate_actions(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape [horizon, action_dim]")
    return array

