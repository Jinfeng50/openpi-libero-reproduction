"""Training-free temporal action fusion for overlapping openpi action chunks.

The LIBERO client normally executes the first ``replan_steps`` actions from a
newly predicted chunk and discards the rest.  With a 50-step action horizon,
this leaves two predictions for the same simulator timestep whenever chunks
overlap.  DGTE (Disagreement-Gated Temporal Ensemble) keeps those predictions,
weights newer chunks more strongly, and reacts faster when the predictions
disagree.  The gripper dimension is copied from the newest chunk because it is
an absolute, effectively discrete command in LIBERO.

The class is deliberately independent of openpi, websocket clients, and
robosuite so that the fusion logic can be tested on CPU and reused by other
clients.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DGTEConfig:
    """Parameters for :class:`DisagreementGatedTemporalEnsembler`.

    ``continuous_dims`` defaults to all dimensions except ``gripper_index``.
    ``disagreement_threshold`` is measured in mean absolute action units over
    those continuous dimensions.  Set ``gate_strength=0`` to obtain a plain
    freshness-weighted temporal ensemble.
    """

    replan_steps: int = 5
    decay: float = 0.7
    disagreement_threshold: float = 0.08
    gate_strength: float = 3.0
    gripper_index: int = -1
    max_history: int = 16

    def __post_init__(self) -> None:
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        if self.decay < 0:
            raise ValueError("decay must be non-negative")
        if self.disagreement_threshold <= 0:
            raise ValueError("disagreement_threshold must be positive")
        if self.gate_strength < 0:
            raise ValueError("gate_strength must be non-negative")
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")


class DisagreementGatedTemporalEnsembler:
    """Fuse predictions from action chunks with a common absolute time axis."""

    def __init__(self, config: DGTEConfig | None = None) -> None:
        self.config = config or DGTEConfig()
        self.reset()

    def reset(self) -> None:
        """Discard all chunks and per-episode diagnostics."""
        self._chunks: list[tuple[int, np.ndarray]] = []
        self._action_dim: int | None = None
        self._latest_start: int | None = None
        self.last_disagreement = 0.0
        self.last_candidate_count = 0

    @property
    def action_dim(self) -> int | None:
        return self._action_dim

    def add_chunk(self, actions: np.ndarray | Iterable[Iterable[float]], start_step: int) -> None:
        """Register a newly inferred chunk beginning at ``start_step``.

        Chunks must be two-dimensional ``[horizon, action_dim]`` arrays.  A
        copy is stored so callers may safely reuse their websocket response.
        Inference calls are expected to advance monotonically in simulator
        time; equal starts are allowed for retries but replace the old chunk.
        """

        if isinstance(start_step, bool) or not isinstance(start_step, (int, np.integer)):
            raise TypeError("start_step must be an integer")
        start_step = int(start_step)
        if start_step < 0:
            raise ValueError("start_step must be non-negative")

        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] == 0:
            raise ValueError("actions must have shape [horizon, action_dim]")
        if not np.isfinite(chunk).all():
            raise ValueError("actions must contain only finite values")
        if self._action_dim is None:
            self._action_dim = int(chunk.shape[1])
        elif chunk.shape[1] != self._action_dim:
            raise ValueError(f"action_dim changed from {self._action_dim} to {chunk.shape[1]}")

        self._chunks = [(s, c) for s, c in self._chunks if s != start_step]
        self._chunks.append((start_step, chunk.copy()))
        self._chunks.sort(key=lambda item: item[0])
        self._latest_start = max(start_step, self._latest_start or start_step)
        if len(self._chunks) > self.config.max_history:
            self._chunks = self._chunks[-self.config.max_history :]

    def action_at(self, step: int) -> np.ndarray:
        """Return the fused action predicted for absolute simulator ``step``."""

        if self._action_dim is None or not self._chunks:
            raise RuntimeError("add_chunk must be called before action_at")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        step = int(step)

        candidates: list[tuple[int, np.ndarray]] = []
        for source_step, chunk in self._chunks:
            offset = step - source_step
            if 0 <= offset < len(chunk):
                candidates.append((source_step, chunk[offset]))
        if not candidates:
            raise KeyError(f"no action chunk covers step {step}")

        newest_source, newest = max(candidates, key=lambda item: item[0])
        if len(candidates) == 1:
            self.last_disagreement = 0.0
            self.last_candidate_count = 1
            return newest.copy()

        continuous = self._continuous_indices()
        values = np.stack([candidate for _, candidate in candidates], axis=0)
        if continuous:
            disagreement = float(np.mean(np.abs(values[:, continuous] - newest[continuous])))
        else:
            disagreement = 0.0
        self.last_disagreement = disagreement
        self.last_candidate_count = len(candidates)

        latest_start = self._latest_start if self._latest_start is not None else newest_source
        ages = np.asarray(
            [(latest_start - source) / self.config.replan_steps for source, _ in candidates], dtype=np.float32
        )
        weights = np.exp(-self.config.decay * np.maximum(ages, 0.0))
        if self.config.gate_strength:
            gate = min(1.0, disagreement / self.config.disagreement_threshold)
            for index, (source, _) in enumerate(candidates):
                if source == newest_source:
                    weights[index] *= 1.0 + self.config.gate_strength * gate
                    break
        weights /= weights.sum()

        fused = newest.copy()
        if continuous:
            fused[continuous] = np.average(values[:, continuous], axis=0, weights=weights)
        # Preserve the newest absolute gripper command rather than averaging
        # -1/+1 into an invalid intermediate command.
        if self.config.gripper_index is not None:
            fused[self._normalise_gripper_index()] = newest[self._normalise_gripper_index()]
        return fused.astype(np.float32, copy=False)

    def next_actions(self, start_step: int, count: int) -> np.ndarray:
        """Return ``count`` fused actions and prune chunks in the past."""

        if count <= 0:
            raise ValueError("count must be positive")
        result = np.stack([self.action_at(start_step + offset) for offset in range(count)])
        cutoff = start_step + count
        self._chunks = [(s, c) for s, c in self._chunks if s + len(c) > cutoff]
        return result

    def chunks_covering(self, step: int) -> list[tuple[int, np.ndarray]]:
        """Return copies of all registered chunks that cover ``step``."""

        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        step = int(step)
        result = []
        for source_step, chunk in self._chunks:
            offset = step - source_step
            if 0 <= offset < len(chunk):
                result.append((source_step, chunk.copy()))
        return result

    def prune_before(self, step: int) -> None:
        """Discard chunks that cannot cover ``step`` or any later timestep."""

        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        step = int(step)
        self._chunks = [(source, chunk) for source, chunk in self._chunks if source + len(chunk) > step]

    def _normalise_gripper_index(self) -> int:
        assert self._action_dim is not None
        index = self.config.gripper_index
        if index < 0:
            index += self._action_dim
        if not 0 <= index < self._action_dim:
            raise ValueError(f"gripper_index {self.config.gripper_index} out of range for dim {self._action_dim}")
        return index

    def _continuous_indices(self) -> list[int]:
        assert self._action_dim is not None
        gripper = self._normalise_gripper_index()
        return [index for index in range(self._action_dim) if index != gripper]


def freshness_weight(age_steps: int, *, replan_steps: int = 5, decay: float = 0.7) -> float:
    """Expose the deterministic freshness weighting for analysis scripts."""

    if age_steps < 0 or replan_steps <= 0 or decay < 0:
        raise ValueError("age_steps >= 0, replan_steps > 0 and decay >= 0 are required")
    return math.exp(-decay * age_steps / replan_steps)
