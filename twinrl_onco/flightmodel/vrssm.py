from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from twinrl_onco.flightmodel.decoder import ObservationDecoder, reconstruction_log_prob
from twinrl_onco.flightmodel.encoders import ActionEncoder, ObservationEncoder
from twinrl_onco.flightmodel.latent import (
    Posterior,
    Prior,
    gaussian_kl,
    reparameterize,
)
from twinrl_onco.flightmodel.predictors import (
    ContinuationHead,
    ProgressionHead,
    RewardHead,
)
from twinrl_onco.flightmodel.recurrence import RecurrentCore, build_core
from twinrl_onco.flightplan import ModelPlan
from twinrl_onco.simenv.actions import ActionGrid


@dataclass
class WorldState:
    hidden: torch.Tensor
    latent: torch.Tensor
    post_mean: torch.Tensor
    post_std: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor


class VRSSM(nn.Module):
    def __init__(self, model: ModelPlan, grid: ActionGrid) -> None:
        super().__init__()
        self.model = model
        self.encoder = ObservationEncoder(model)
        self.action_encoder = ActionEncoder(grid.feature_dim, model.action_dim)
        self.core: RecurrentCore = build_core(model)
        self.posterior = Posterior(model.hidden_dim, model.fuse_dim, model.latent_dim)
        self.prior = Prior(model.hidden_dim, model.latent_dim)
        self.decoder = ObservationDecoder(model)
        self.reward_head = RewardHead(model)
        self.continuation_head = ContinuationHead(model)
        self.progression_head = ProgressionHead(model)
        self.latent_dim = model.latent_dim
        self.action_dim = model.action_dim

    def encode_actions(self, action_features: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(action_features)

    def observe(
        self,
        gene: torch.Tensor,
        flow: torch.Tensor,
        clinical: torch.Tensor,
        action_features: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> WorldState:
        steps, batch = gene.shape[0], gene.shape[1]
        device = gene.device
        hidden = self.core.reset(batch, device)
        z_prev = torch.zeros(batch, self.latent_dim, device=device)
        action_emb = self.encode_actions(action_features)
        zero_action = torch.zeros(batch, self.action_dim, device=device)

        hs, zs = [], []
        pm, ps, qm, qs = [], [], [], []
        for t in range(steps):
            a_prev = zero_action if t == 0 else action_emb[t - 1]
            hidden = self.core.step(hidden, z_prev, a_prev)
            embed = self.encoder(gene[t], flow[t], clinical[t])
            post_mean, post_std = self.posterior(hidden, embed)
            prior_mean, prior_std = self.prior(hidden)
            if self.model.use_stochastic:
                z = reparameterize(post_mean, post_std, generator)
            else:
                z = post_mean
            hs.append(hidden)
            zs.append(z)
            pm.append(post_mean)
            ps.append(post_std)
            qm.append(prior_mean)
            qs.append(prior_std)
            z_prev = z
        return WorldState(
            hidden=torch.stack(hs),
            latent=torch.stack(zs),
            post_mean=torch.stack(pm),
            post_std=torch.stack(ps),
            prior_mean=torch.stack(qm),
            prior_std=torch.stack(qs),
        )

    def per_step_terms(
        self,
        state: WorldState,
        obs: dict[str, torch.Tensor],
        reward: torch.Tensor,
        continuation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        steps, batch = state.hidden.shape[0], state.hidden.shape[1]
        flat_h = state.hidden.reshape(steps * batch, -1)
        flat_z = state.latent.reshape(steps * batch, -1)
        recon = self.decoder(flat_h, flat_z)
        recon = {k: v.reshape(steps, batch, -1) for k, v in recon.items()}
        recon_lp = reconstruction_log_prob(recon, obs)
        r_hat = self.reward_head(flat_h, flat_z).reshape(steps, batch)
        c_logit = self.continuation_head(flat_h, flat_z).reshape(steps, batch)
        reward_lp = -0.5 * (reward - r_hat).pow(2)
        cont_lp = -torch.nn.functional.binary_cross_entropy_with_logits(
            c_logit, continuation, reduction="none"
        )
        kl = gaussian_kl(state.post_mean, state.post_std, state.prior_mean, state.prior_std)
        return {
            "recon_lp": recon_lp,
            "reward_lp": reward_lp,
            "cont_lp": cont_lp,
            "kl": kl,
        }

    def progression_logits(self, state: WorldState) -> torch.Tensor:
        steps, batch = state.hidden.shape[0], state.hidden.shape[1]
        flat_h = state.hidden.reshape(steps * batch, -1)
        flat_z = state.latent.reshape(steps * batch, -1)
        return self.progression_head(flat_h, flat_z).reshape(steps, batch)
