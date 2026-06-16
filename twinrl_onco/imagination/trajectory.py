from __future__ import annotations

import math

import torch

from twinrl_onco.cockpit.policy import HierarchicalPolicy
from twinrl_onco.flightmodel.vrssm import VRSSM
from twinrl_onco.imagination.rollout import imagine


def rollout_std(n_rollouts: int) -> float:
    return 0.28 / math.sqrt(n_rollouts)


def aggregate_progression(progression: torch.Tensor) -> torch.Tensor:
    return progression.mean(dim=0)


def monte_carlo_progression(
    world: VRSSM,
    policy: HierarchicalPolicy,
    hidden0: torch.Tensor,
    latent0: torch.Tensor,
    horizon: int,
    n_rollouts: int,
    progression_horizon: int,
    stochastic: bool,
    use_rollouts: bool = True,
    aggregate: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    steps = min(progression_horizon, horizon)
    samples = n_rollouts if use_rollouts else 1
    per_rollout = []
    for _ in range(samples):
        rollout = imagine(world, policy, hidden0, latent0, horizon, stochastic, generator)
        window = rollout.progression[:steps]
        cumulative = 1.0 - torch.prod(1.0 - window, dim=0)
        per_rollout.append(cumulative)
    stacked = torch.stack(per_rollout)
    if not aggregate:
        return stacked[0]
    return aggregate_progression(stacked)
