from __future__ import annotations

import torch


def discounted_return(reward: torch.Tensor, gamma: float) -> torch.Tensor:
    steps = reward.shape[0]
    weights = gamma ** torch.arange(steps, device=reward.device).float()
    return (reward * weights.unsqueeze(-1)).sum(dim=0)


def lambda_returns(
    reward: torch.Tensor,
    continuation: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    steps = reward.shape[0]
    out = torch.zeros_like(reward)
    nxt = values[-1]
    for t in reversed(range(steps)):
        bootstrap = (1.0 - lam) * values[t + 1] + lam * nxt
        nxt = reward[t] + gamma * continuation[t] * bootstrap
        out[t] = nxt
    return out


def actor_loss(log_prob: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    return -(log_prob * advantage.detach()).sum(dim=0).mean()


def critic_loss(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (values - target.detach()).pow(2).sum(dim=0).mean()
