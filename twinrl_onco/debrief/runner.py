from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from twinrl_onco.groundstation.cohort import Cohort
from twinrl_onco.imagination.trajectory import monte_carlo_progression
from twinrl_onco.simenv.environment import TreatmentEnvironment
from twinrl_onco.simenv.pkpd import response_category
from twinrl_onco.telemetry import generator_for
from twinrl_onco.tower import TwinRLOnco, make_environment


@dataclass
class EvalBundle:
    prog_score: dict[int, np.ndarray]
    prog_true: dict[int, np.ndarray]
    ttp: np.ndarray
    pfs: dict[int, np.ndarray]
    drug_exposure: np.ndarray
    qalm: np.ndarray
    grade3_ae: np.ndarray
    complete_response: np.ndarray
    phase_seq: np.ndarray
    dose_seq: np.ndarray
    world_quality: dict[str, dict[str, np.ndarray]]


def _encode_initial(
    model: TwinRLOnco, env: TreatmentEnvironment, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    obs = env.reset()
    h = model.world.core.reset(env.n, torch.device("cpu"))
    z = torch.zeros(env.n, model.config.model.latent_dim)
    a_emb = torch.zeros(env.n, model.config.model.action_dim)
    h = model.world.core.step(h, z, a_emb)
    embed = model.world.encoder(obs["gene"], obs["flow"], obs["clinical"])
    mean, _ = model.world.posterior(h, embed)
    return h.detach(), mean.detach()


@torch.no_grad()
def collect_eval(model: TwinRLOnco, cohort: Cohort, index: torch.Tensor, seed: int) -> EvalBundle:
    cfg = model.config
    gen = generator_for(seed)
    env = make_environment(model, cohort, index, seed=seed)
    h0, z0 = _encode_initial(model, env, gen)

    prog_score: dict[int, np.ndarray] = {}
    for horizon in (12, 24):
        score = monte_carlo_progression(
            model.world,
            model.policy,
            h0,
            z0,
            cfg.imagination.horizon,
            cfg.imagination.n_rollouts,
            horizon,
            cfg.model.use_stochastic,
            cfg.imagination.use_rollouts,
            cfg.imagination.aggregate,
            gen,
        )
        prog_score[horizon] = score.cpu().numpy()

    n = env.n
    length = cfg.sim.episode_length
    obs = env.reset()
    h = model.world.core.reset(n, torch.device("cpu"))
    z = torch.zeros(n, cfg.model.latent_dim)
    a_emb = torch.zeros(n, cfg.model.action_dim)
    burden_ratio = torch.ones(n)
    ttp = torch.full((n,), float(length))
    progressed = torch.zeros(n, dtype=torch.bool)
    drug = torch.zeros(n)
    qalm = torch.zeros(n)
    ae = torch.zeros(n, dtype=torch.bool)
    cr = torch.zeros(n, dtype=torch.bool)
    pfs = {12: torch.ones(n), 24: torch.ones(n)}
    phases, doses = [], []
    recon_err: dict[str, list[torch.Tensor]] = {"gene": [], "lymph": [], "hgb": [], "plt": []}
    true_vals: dict[str, list[torch.Tensor]] = {"gene": [], "lymph": [], "hgb": [], "plt": []}
    top = float(max(model.grid.dose_levels_mg))

    for t in range(length):
        h = model.world.core.step(h, z, a_emb)
        embed = model.world.encoder(obs["gene"], obs["flow"], obs["clinical"])
        mean, _ = model.world.posterior(h, embed)
        z = mean
        recon = model.world.decoder(h, z)
        recon_err["gene"].append(recon["gene"][:, :100])
        true_vals["gene"].append(obs["gene"][:, :100])
        if obs["clinical"].shape[1] > 13:
            recon_err["lymph"].append(recon["clinical"][:, 10])
            recon_err["hgb"].append(recon["clinical"][:, 11])
            recon_err["plt"].append(recon["clinical"][:, 12])
            true_vals["lymph"].append(obs["clinical"][:, 10])
            true_vals["hgb"].append(obs["clinical"][:, 11])
            true_vals["plt"].append(obs["clinical"][:, 12])
        decision = model.policy.act(h, z, gen)
        phases.append(decision["phase"])
        doses.append(decision["dose_mg"])
        drug = drug + decision["dose_mg"] / top
        ae = ae | (decision["dose_mg"] >= top)
        obs, reward, cont, prog = env.step(decision["dose_mg"], decision["combo_idx"])
        burden_ratio = (env.sensitive + env.resistant) / env.baseline
        category = response_category(burden_ratio, cfg.sim)
        cr = cr | (category == 0)
        qalm = qalm + (burden_ratio < cfg.sim.pr_threshold).float()
        newly = (prog > 0.5) & (~progressed)
        ttp = torch.where(newly, torch.full_like(ttp, float(t + 1)), ttp)
        progressed = progressed | (prog > 0.5)
        for hz in (12, 24):
            if t < hz:
                pfs[hz] = torch.where(prog > 0.5, torch.zeros_like(pfs[hz]), pfs[hz])
        a_emb = model.world.encode_actions(decision["features"])

    prog_true = {
        12: (ttp <= 12).float().cpu().numpy(),
        24: (ttp <= 24).float().cpu().numpy(),
    }

    quality: dict[str, dict[str, np.ndarray]] = {}
    for key in recon_err:
        if not recon_err[key]:
            continue
        pred = torch.stack(recon_err[key])
        true = torch.stack(true_vals[key])
        quality[key] = {
            "pred": pred.cpu().numpy().reshape(-1),
            "true": true.cpu().numpy().reshape(-1),
        }

    return EvalBundle(
        prog_score=prog_score,
        prog_true=prog_true,
        ttp=ttp.cpu().numpy(),
        pfs={k: v.cpu().numpy() for k, v in pfs.items()},
        drug_exposure=drug.cpu().numpy(),
        qalm=qalm.cpu().numpy(),
        grade3_ae=ae.float().cpu().numpy(),
        complete_response=cr.float().cpu().numpy(),
        phase_seq=torch.stack(phases).cpu().numpy(),
        dose_seq=torch.stack(doses).cpu().numpy(),
        world_quality=quality,
    )
