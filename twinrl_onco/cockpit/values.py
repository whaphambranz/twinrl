from __future__ import annotations

import torch
from torch import nn

from twinrl_onco.flightplan import ModelPlan


class _Value(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([hidden, latent], dim=-1)).squeeze(-1)


class HighValue(_Value):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__(model.hidden_dim + model.latent_dim, model.hidden_dim)


class LowValue(_Value):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__(model.hidden_dim + model.latent_dim, model.hidden_dim)
