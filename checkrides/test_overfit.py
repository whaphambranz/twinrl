from __future__ import annotations

import torch

from twinrl_onco.flightmodel.elbo import vrssm_objective
from twinrl_onco.telemetry import set_seed


def test_world_model_overfits_single_batch(model, trajectory, features):
    set_seed(0)
    params = list(model.world.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    obs = {"gene": trajectory.gene, "flow": trajectory.flow, "clinical": trajectory.clinical}

    def loss_value():
        state = model.world.observe(trajectory.gene, trajectory.flow, trajectory.clinical, features)
        terms = model.world.per_step_terms(state, obs, trajectory.reward, trajectory.continuation)
        loss, _ = vrssm_objective(terms, model.config.model.beta, True)
        return loss

    first = float(loss_value().detach())
    for _ in range(80):
        opt.zero_grad()
        loss = loss_value()
        loss.backward()
        opt.step()
    last = float(loss_value().detach())
    assert last < first
    assert (first - last) > 1.0
