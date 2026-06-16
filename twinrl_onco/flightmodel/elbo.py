from __future__ import annotations

import torch


def vrssm_objective(
    terms: dict[str, torch.Tensor], beta: float, use_kl: bool
) -> tuple[torch.Tensor, dict[str, float]]:
    recon = terms["recon_lp"].sum(dim=0).mean()
    reward = terms["reward_lp"].sum(dim=0).mean()
    cont = terms["cont_lp"].sum(dim=0).mean()
    kl = terms["kl"].sum(dim=0).mean()
    kl_term = beta * kl if use_kl else torch.zeros((), device=kl.device)
    elbo = recon - kl_term + reward + cont
    loss = -elbo
    report = {
        "recon": float(recon.detach()),
        "kl": float(kl.detach()),
        "reward_ll": float(reward.detach()),
        "cont_ll": float(cont.detach()),
        "elbo": float(elbo.detach()),
    }
    return loss, report


def progression_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)


def active_units(post_std: torch.Tensor, threshold: float = 0.1) -> int:
    variance = post_std.detach().pow(2).mean(dim=(0, 1))
    return int((variance > threshold).sum().item())
