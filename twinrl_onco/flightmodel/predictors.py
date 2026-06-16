from __future__ import annotations

import torch
from torch import nn

from twinrl_onco.flightplan import ModelPlan


class _Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RewardHead(nn.Module):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__()
        self.head = _Head(model.hidden_dim + model.latent_dim, model.hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([hidden, latent], dim=-1)).squeeze(-1)


class ContinuationHead(nn.Module):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__()
        self.head = _Head(model.hidden_dim + model.latent_dim, model.hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([hidden, latent], dim=-1)).squeeze(-1)


class ProgressionHead(nn.Module):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__()
        self.head = _Head(model.hidden_dim + model.latent_dim, model.hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([hidden, latent], dim=-1)).squeeze(-1)

    def probability(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden, latent))
