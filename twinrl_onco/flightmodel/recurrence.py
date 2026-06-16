from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from twinrl_onco.flightplan import ModelPlan


class RecurrentCore(Protocol):
    def reset(self, batch: int, device: torch.device) -> torch.Tensor: ...

    def step(
        self, hidden: torch.Tensor, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor: ...


class GruCore(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(latent_dim + action_dim, hidden_dim)

    def reset(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def step(
        self, hidden: torch.Tensor, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.cell(torch.cat([latent, action], dim=-1), hidden)


class TransformerCore(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(latent_dim + action_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self._history: list[torch.Tensor] = []

    def reset(self, batch: int, device: torch.device) -> torch.Tensor:
        self._history = []
        return torch.zeros(batch, self.hidden_dim, device=device)

    def step(
        self, hidden: torch.Tensor, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        token = self.embed(torch.cat([latent, action], dim=-1))
        self._history.append(token)
        seq = torch.stack(self._history, dim=0)
        encoded = self.encoder(seq)
        return encoded[-1]


def build_core(model: ModelPlan) -> RecurrentCore:
    if model.backbone == "transformer":
        return TransformerCore(model.latent_dim, model.action_dim, model.hidden_dim, model.n_heads)
    return GruCore(model.latent_dim, model.action_dim, model.hidden_dim)
