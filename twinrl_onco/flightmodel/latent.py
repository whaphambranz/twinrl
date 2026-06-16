from __future__ import annotations

import torch
from torch import nn

MIN_STD = 0.1


class GaussianState(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.mean = nn.Linear(hidden, latent_dim)
        self.std = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        mean = self.mean(feat)
        std = torch.nn.functional.softplus(self.std(feat)) + MIN_STD
        return mean, std


class Posterior(nn.Module):
    def __init__(self, hidden_dim: int, embed_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.state = GaussianState(hidden_dim + embed_dim, latent_dim, hidden_dim)

    def forward(
        self, hidden: torch.Tensor, embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state(torch.cat([hidden, embed], dim=-1))


class Prior(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.state = GaussianState(hidden_dim, latent_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state(hidden)


def reparameterize(
    mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    noise = torch.randn(mean.shape, generator=generator, device=mean.device)
    return mean + std * noise


def gaussian_kl(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    var_post = post_std.pow(2)
    var_prior = prior_std.pow(2)
    term = (
        torch.log(prior_std / post_std)
        + (var_post + (post_mean - prior_mean).pow(2)) / (2.0 * var_prior)
        - 0.5
    )
    return term.sum(dim=-1)
