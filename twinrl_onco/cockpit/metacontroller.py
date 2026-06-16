from __future__ import annotations

import torch
from torch import nn

from twinrl_onco.flightplan import ModelPlan, PolicyPlan


class MetaController(nn.Module):
    def __init__(self, model: ModelPlan, policy: PolicyPlan) -> None:
        super().__init__()
        self.n_phases = len(policy.phases)
        self.proj = nn.Linear(model.hidden_dim + model.latent_dim, self.n_phases)

    def logits(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([hidden, latent], dim=-1))

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(hidden, latent), dim=-1)
