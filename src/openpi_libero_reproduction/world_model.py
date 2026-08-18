"""A lightweight latent-change and success critic for frozen VLA policies."""

from __future__ import annotations

import hashlib
import re
from typing import NamedTuple

import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18


class CriticOutput(NamedTuple):
    predicted_future_latent: torch.Tensor
    latent_log_variance: torch.Tensor
    terminal_logit: torch.Tensor
    success_logit: torch.Tensor


def hashed_text_features(prompts: list[str], dimension: int = 256) -> torch.Tensor:
    """Build deterministic signed bag-of-token features without a text model."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    result = torch.zeros((len(prompts), dimension), dtype=torch.float32)
    for row, prompt in enumerate(prompts):
        for token in re.findall(r"[a-z0-9]+", prompt.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            result[row, value % dimension] += 1.0 if value & 1 else -1.0
    return torch.nn.functional.normalize(result, dim=-1)


class FrozenResNet18Encoder(nn.Module):
    """Shared frozen encoder for agent and wrist images."""

    output_dim = 1024

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone.eval()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor, wrist_image: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.backbone(self._prepare(image)), self.backbone(self._prepare(wrist_image))), dim=-1)

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have four dimensions")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] != 3:
            raise ValueError("images must be BCHW or BHWC RGB tensors")
        images = images.to(device=self.mean.device, dtype=torch.float32) / 255.0
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return (images - self.mean) / self.std


class LatentChangeCritic(nn.Module):
    """Predict future visual change, short-horizon termination, and success."""

    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        state_dim: int = 8,
        action_horizon: int = 10,
        action_dim: int = 7,
        text_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("state_dim", state_dim),
            ("action_horizon", action_horizon),
            ("action_dim", action_dim),
            ("text_dim", text_dim),
            ("hidden_dim", hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.text_dim = text_dim
        input_dim = latent_dim + state_dim + action_horizon * action_dim + text_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_head = nn.Linear(hidden_dim, 1)
        self.terminal_head = nn.Linear(hidden_dim, 1)
        self.success_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        current_latent: torch.Tensor,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        text_features: torch.Tensor,
    ) -> CriticOutput:
        self._validate_inputs(current_latent, state, action_chunk, text_features)
        features = torch.cat(
            (current_latent, state, action_chunk.flatten(start_dim=1), text_features),
            dim=-1,
        )
        hidden = self.trunk(features)
        delta = self.delta_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 5.0).squeeze(-1)
        return CriticOutput(
            predicted_future_latent=current_latent + delta,
            latent_log_variance=log_variance,
            terminal_logit=self.terminal_head(hidden).squeeze(-1),
            success_logit=self.success_head(hidden).squeeze(-1),
        )

    def score(self, output: CriticOutput, uncertainty_penalty: float = 0.1) -> torch.Tensor:
        if uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty must be non-negative")
        return torch.sigmoid(output.success_logit) - uncertainty_penalty * torch.exp(
            0.5 * output.latent_log_variance
        )

    def _validate_inputs(self, latent, state, actions, text) -> None:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"current_latent must have shape [batch, {self.latent_dim}]")
        batch = latent.shape[0]
        if state.shape != (batch, self.state_dim):
            raise ValueError(f"state must have shape [batch, {self.state_dim}]")
        if actions.shape != (batch, self.action_horizon, self.action_dim):
            raise ValueError(
                f"action_chunk must have shape [batch, {self.action_horizon}, {self.action_dim}]"
            )
        if text.shape != (batch, self.text_dim):
            raise ValueError(f"text_features must have shape [batch, {self.text_dim}]")


def critic_loss(
    output: CriticOutput,
    *,
    future_latent: torch.Tensor,
    terminal_target: torch.Tensor,
    success_target: torch.Tensor,
    terminal_weight: float = 1.0,
    success_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Heteroscedastic latent regression plus two binary objectives."""

    if future_latent.shape != output.predicted_future_latent.shape:
        raise ValueError("future_latent shape must match predicted_future_latent")
    squared_error = (output.predicted_future_latent - future_latent).pow(2).mean(dim=-1)
    latent_loss = 0.5 * (
        torch.exp(-output.latent_log_variance) * squared_error + output.latent_log_variance
    ).mean()
    terminal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.terminal_logit, terminal_target.float()
    )
    success_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.success_logit, success_target.float()
    )
    total = latent_loss + terminal_weight * terminal_loss + success_weight * success_loss
    return total, {
        "latent_loss": latent_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "success_loss": success_loss.detach(),
    }
