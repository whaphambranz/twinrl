from __future__ import annotations

from dataclasses import dataclass

import torch

from twinrl_onco.flightplan import TrainPlan
from twinrl_onco.tower import TwinRLOnco


@dataclass
class Optimizers:
    world: torch.optim.Optimizer
    actor: torch.optim.Optimizer
    critic: torch.optim.Optimizer


def build_optimizers(model: TwinRLOnco, plan: TrainPlan) -> Optimizers:
    world_params = list(model.world.parameters())
    actor_params = list(model.policy.meta.parameters()) + list(model.policy.low.parameters())
    critic_params = list(model.policy.high_value.parameters()) + list(
        model.policy.low_value.parameters()
    )
    return Optimizers(
        world=torch.optim.Adam(world_params, lr=plan.world_lr),
        actor=torch.optim.Adam(actor_params, lr=plan.actor_lr),
        critic=torch.optim.Adam(critic_params, lr=plan.critic_lr),
    )


def clip_grads(params: list[torch.nn.Parameter], max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(params, max_norm)


class EarlyStopper:
    def __init__(self, patience: int, enabled: bool = True) -> None:
        self.patience = patience
        self.enabled = enabled
        self.best = float("inf")
        self.waited = 0

    def update(self, value: float) -> bool:
        if not self.enabled:
            return False
        if value < self.best - 1e-5:
            self.best = value
            self.waited = 0
            return False
        self.waited += 1
        return self.waited >= self.patience
