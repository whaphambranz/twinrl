from __future__ import annotations

from dataclasses import dataclass

import torch

from twinrl_onco.cockpit.policy import HierarchicalPolicy
from twinrl_onco.flightmodel.latent import reparameterize
from twinrl_onco.flightmodel.vrssm import VRSSM


@dataclass
class ImaginedRollout:
    hidden: torch.Tensor
    latent: torch.Tensor
    phase: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    continuation: torch.Tensor
    progression: torch.Tensor
    hidden_last: torch.Tensor
    latent_last: torch.Tensor


def imagine(
    world: VRSSM,
    policy: HierarchicalPolicy,
    hidden0: torch.Tensor,
    latent0: torch.Tensor,
    horizon: int,
    stochastic: bool,
    generator: torch.Generator | None = None,
) -> ImaginedRollout:
    hidden, latent = hidden0, latent0
    hs, zs, phases, actions, rewards, conts, progs = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(horizon):
            decision = policy.act(hidden, latent, generator)
            a_emb = world.encode_actions(decision["features"])
            reward = world.reward_head(hidden, latent)
            cont = torch.sigmoid(world.continuation_head(hidden, latent))
            prog = world.progression_head.probability(hidden, latent)
            hs.append(hidden)
            zs.append(latent)
            phases.append(decision["phase"])
            actions.append(decision["action"])
            rewards.append(reward)
            conts.append(cont)
            progs.append(prog)
            hidden = world.core.step(hidden, latent, a_emb)
            prior_mean, prior_std = world.prior(hidden)
            latent = reparameterize(prior_mean, prior_std, generator) if stochastic else prior_mean
    return ImaginedRollout(
        hidden=torch.stack(hs),
        latent=torch.stack(zs),
        phase=torch.stack(phases),
        action=torch.stack(actions),
        reward=torch.stack(rewards),
        continuation=torch.stack(conts),
        progression=torch.stack(progs),
        hidden_last=hidden,
        latent_last=latent,
    )
