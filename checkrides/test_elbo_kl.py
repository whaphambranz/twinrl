from __future__ import annotations

import torch

from twinrl_onco.flightmodel.elbo import vrssm_objective
from twinrl_onco.flightmodel.latent import gaussian_kl


def test_kl_zero_for_identical():
    mean = torch.randn(5, 4)
    std = torch.rand(5, 4) + 0.2
    kl = gaussian_kl(mean, std, mean.clone(), std.clone())
    assert torch.allclose(kl, torch.zeros(5), atol=1e-5)


def test_kl_matches_torch_distributions():
    pm, ps = torch.randn(8, 3), torch.rand(8, 3) + 0.3
    qm, qs = torch.randn(8, 3), torch.rand(8, 3) + 0.3
    p = torch.distributions.Normal(pm, ps)
    q = torch.distributions.Normal(qm, qs)
    reference = torch.distributions.kl_divergence(p, q).sum(-1)
    assert torch.allclose(gaussian_kl(pm, ps, qm, qs), reference, atol=1e-5)


def test_beta_scales_kl_term():
    terms = {
        "recon_lp": torch.zeros(3, 4),
        "reward_lp": torch.zeros(3, 4),
        "cont_lp": torch.zeros(3, 4),
        "kl": torch.ones(3, 4),
    }
    loss_low, rep_low = vrssm_objective(terms, beta=0.1, use_kl=True)
    loss_high, _ = vrssm_objective(terms, beta=0.5, use_kl=True)
    loss_off, _ = vrssm_objective(terms, beta=0.5, use_kl=False)
    assert float(loss_high) > float(loss_low)
    assert abs(float(loss_off)) < 1e-6
    assert rep_low["kl"] == 3.0
