from __future__ import annotations

import torch
from torch import nn

from twinrl_onco.cockpit.lowlevel import LowLevelController
from twinrl_onco.cockpit.metacontroller import MetaController
from twinrl_onco.cockpit.values import HighValue, LowValue
from twinrl_onco.flightplan import ModelPlan, PolicyPlan
from twinrl_onco.simenv.actions import ActionGrid


def _sample(log_prob: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    probs = log_prob.exp().clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def _gather(log_prob: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return log_prob.gather(-1, index.unsqueeze(-1)).squeeze(-1)


class HierarchicalPolicy(nn.Module):
    def __init__(self, model: ModelPlan, policy: PolicyPlan, grid: ActionGrid) -> None:
        super().__init__()
        self.grid = grid
        self.hierarchical = policy.hierarchical
        self.meta = MetaController(model, policy)
        self.low = LowLevelController(model, policy, grid)
        self.high_value = HighValue(model)
        self.low_value = LowValue(model)

    def act(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = hidden.shape[0]
        if self.hierarchical:
            logp_high = self.meta(hidden, latent)
            phase = _sample(logp_high, generator)
            logp_high_a = _gather(logp_high, phase)
        else:
            phase = torch.zeros(batch, dtype=torch.long, device=hidden.device)
            logp_high_a = torch.zeros(batch, device=hidden.device)
        logp_low = self.low(hidden, latent, phase)
        action = _sample(logp_low, generator)
        logp_low_a = _gather(logp_low, action)
        dose_mg, combo_idx = self.grid.decode(action)
        features = self.grid.features(dose_mg, combo_idx, phase)
        return {
            "phase": phase,
            "action": action,
            "dose_mg": dose_mg,
            "combo_idx": combo_idx,
            "logp_high": logp_high_a,
            "logp_low": logp_low_a,
            "features": features,
        }

    def log_probs(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        phase: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hierarchical:
            logp_high = _gather(self.meta(hidden, latent), phase)
        else:
            logp_high = torch.zeros(hidden.shape[0], device=hidden.device)
        logp_low = _gather(self.low(hidden, latent, phase), action)
        return logp_high, logp_low
