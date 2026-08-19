"""Inference-only action-chunk selection with the Stage-A latent critic."""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from .world_model import FrozenResNet18Encoder, LatentChangeCritic, hashed_text_features


def gate_world_model_choice(
    scores: np.ndarray,
    uncertainty: np.ndarray,
    *,
    margin_threshold: float,
    uncertainty_threshold: float,
) -> tuple[bool, float, float]:
    """Accept a world-model choice only with enough candidates and confidence."""

    scores = np.asarray(scores, dtype=np.float32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    if scores.ndim != 1 or uncertainty.ndim != 1 or len(scores) != len(uncertainty):
        raise ValueError("scores and uncertainty must be equal-length vectors")
    if not len(scores) or not np.isfinite(scores).all() or not np.isfinite(uncertainty).all():
        raise ValueError("scores and uncertainty must be non-empty and finite")
    if margin_threshold < 0 or uncertainty_threshold < 0:
        raise ValueError("gate thresholds must be non-negative")
    best_index = int(np.argmax(scores))
    best_uncertainty = float(uncertainty[best_index])
    if len(scores) < 2:
        return False, 0.0, best_uncertainty
    ranked = np.sort(scores)
    margin = float(ranked[-1] - ranked[-2])
    accepted = margin >= margin_threshold and best_uncertainty <= uncertainty_threshold
    return accepted, margin, best_uncertainty


def align_action_chunk(action_chunk: np.ndarray, offset: int, horizon: int) -> np.ndarray:
    """Align a chunk to the current timestep and pad its short tail."""

    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or not len(chunk) or not np.isfinite(chunk).all():
        raise ValueError("action_chunk must have finite shape [horizon, action_dim]")
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if offset >= len(chunk):
        raise ValueError("offset must point inside action_chunk")
    aligned = chunk[int(offset) : int(offset) + horizon]
    if len(aligned) < horizon:
        aligned = np.concatenate((aligned, np.repeat(aligned[-1:], horizon - len(aligned), axis=0)), axis=0)
    return np.ascontiguousarray(aligned)


class WorldModelActionSelector:
    """Score overlapping policy chunks using only the current observation."""

    def __init__(
        self,
        checkpoint: str | pathlib.Path,
        *,
        device: str = "cpu",
        encoder_weights: str = "default",
        uncertainty_penalty: float = 0.1,
    ) -> None:
        if uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty must be non-negative")
        if encoder_weights not in {"default", "none"}:
            raise ValueError("encoder_weights must be 'default' or 'none'")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the world-model selector but is unavailable")
        checkpoint = pathlib.Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"world-model checkpoint does not exist: {checkpoint}")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = dict(payload.get("config", {}))
        if not config:
            raise ValueError(f"world-model checkpoint has no model config: {checkpoint}")
        # action_source is checkpoint metadata used by data/evaluation tooling, not the critic model.
        config = {key: value for key, value in config.items() if key != "action_source"}
        self.device = device
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.encoder = FrozenResNet18Encoder(pretrained=encoder_weights == "default").to(device).eval()
        self.model = LatentChangeCritic(**config).to(device).eval()
        self.model.load_state_dict(payload["model"])

    @property
    def action_horizon(self) -> int:
        return self.model.action_horizon

    @torch.no_grad()
    def score_chunks(
        self,
        *,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prompt: str,
        action_chunks: list[np.ndarray],
    ) -> np.ndarray:
        """Return one uncertainty-penalized score for each candidate chunk."""

        return self.score_chunks_with_diagnostics(
            image=image,
            wrist_image=wrist_image,
            state=state,
            prompt=prompt,
            action_chunks=action_chunks,
        )["score"]

    @torch.no_grad()
    def score_chunks_with_diagnostics(
        self,
        *,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prompt: str,
        action_chunks: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return scores plus success probability and latent uncertainty."""

        if not action_chunks:
            raise ValueError("at least one action chunk is required")
        chunks = np.asarray(action_chunks, dtype=np.float32)
        if chunks.ndim != 3 or not np.isfinite(chunks).all():
            raise ValueError("action_chunks must have shape [candidates, horizon, action_dim]")
        if np.asarray(image).dtype != np.uint8 or np.asarray(wrist_image).dtype != np.uint8:
            raise ValueError("image and wrist_image must be uint8")
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("state must be a finite vector")
        count = len(chunks)
        image_batch = torch.from_numpy(np.repeat(np.asarray(image)[None], count, axis=0)).to(self.device)
        wrist_batch = torch.from_numpy(np.repeat(np.asarray(wrist_image)[None], count, axis=0)).to(self.device)
        state_batch = torch.from_numpy(np.repeat(state[None], count, axis=0)).to(self.device)
        action_batch = torch.from_numpy(chunks).to(self.device)
        text_batch = hashed_text_features([str(prompt)] * count, dimension=self.model.text_dim).to(self.device)
        current_latent = self.encoder(image_batch, wrist_batch)
        output = self.model(current_latent, state_batch, action_batch, text_batch)
        success_probability = torch.sigmoid(output.success_logit)
        uncertainty = torch.exp(0.5 * output.latent_log_variance)
        score = success_probability - self.uncertainty_penalty * uncertainty
        return {
            "score": score.cpu().numpy(),
            "success_probability": success_probability.cpu().numpy(),
            "uncertainty": uncertainty.cpu().numpy(),
        }

    def select_chunk(self, **kwargs) -> tuple[int, np.ndarray]:
        scores = self.score_chunks(**kwargs)
        index = int(np.argmax(scores))
        return index, scores
