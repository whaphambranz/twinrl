from __future__ import annotations

import torch
from torch import nn

from twinrl_onco.flightplan import ModelPlan, PolicyPlan
from twinrl_onco.simenv.actions import ActionGrid

NEG_INF = -1e9


class LowLevelController(nn.Module):
    def __init__(self, model: ModelPlan, policy: PolicyPlan, grid: ActionGrid) -> None:
        super().__init__()
        self.grid = grid
        self.phase_specific = policy.phase_specific_lowlevel
        self.hierarchical = policy.hierarchical
        in_dim = model.hidden_dim + model.latent_dim
        n_phases = len(policy.phases)
        if self.phase_specific and self.hierarchical:
            self.heads = nn.ModuleList([nn.Linear(in_dim, grid.n_actions) for _ in range(n_phases)])
        else:
            self.heads = nn.ModuleList([nn.Linear(in_dim, grid.n_actions)])
        self.register_buffer("masks", grid.all_phase_masks())
        self.masks: torch.Tensor

    def logits(
        self, hidden: torch.Tensor, latent: torch.Tensor, phase_idx: torch.Tensor
    ) -> torch.Tensor:
        feat = torch.cat([hidden, latent], dim=-1)
        if self.phase_specific and self.hierarchical:
            stacked = torch.stack([head(feat) for head in self.heads], dim=1)
            gathered = stacked.gather(
                1, phase_idx.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            ).squeeze(1)
        else:
            gathered = self.heads[0](feat)
        if self.hierarchical:
            mask = self.masks[phase_idx]
            gathered = torch.where(mask, gathered, torch.full_like(gathered, NEG_INF))
        return gathered

    def forward(
        self, hidden: torch.Tensor, latent: torch.Tensor, phase_idx: torch.Tensor
    ) -> torch.Tensor:
        return torch.log_softmax(self.logits(hidden, latent, phase_idx), dim=-1)
