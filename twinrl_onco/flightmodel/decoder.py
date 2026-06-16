from __future__ import annotations

import math

import torch
from torch import nn

from twinrl_onco.flightplan import ModelPlan

LOG_2PI = math.log(2.0 * math.pi)


class ObservationDecoder(nn.Module):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__()
        in_dim = model.hidden_dim + model.latent_dim
        hidden = model.hidden_dim
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.gene = nn.Linear(hidden, model.gene_dim)
        self.flow = nn.Linear(hidden, model.flow_dim)
        self.clinical = nn.Linear(hidden, model.clinical_dim)

    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.trunk(torch.cat([hidden, latent], dim=-1))
        return {
            "gene": self.gene(feat),
            "flow": self.flow(feat),
            "clinical": self.clinical(feat),
        }


def gaussian_log_prob(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    se = (target - mean).pow(2)
    return (-0.5 * (se + LOG_2PI)).sum(dim=-1)


def reconstruction_log_prob(
    recon: dict[str, torch.Tensor], obs: dict[str, torch.Tensor]
) -> torch.Tensor:
    total = gaussian_log_prob(recon["gene"], obs["gene"])
    total = total + gaussian_log_prob(recon["flow"], obs["flow"])
    total = total + gaussian_log_prob(recon["clinical"], obs["clinical"])
    return total
