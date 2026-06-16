from __future__ import annotations

import torch

from twinrl_onco.flightplan import SimPlan


def drug_effect(
    dose_mg: torch.Tensor,
    ec50_mg: torch.Tensor,
    mcl1: torch.Tensor,
    combo_boost: torch.Tensor,
    plan: SimPlan,
) -> torch.Tensor:
    dose = dose_mg.clamp_min(0.0)
    hill = plan.hill
    numer = dose.pow(hill)
    denom = ec50_mg.clamp_min(1.0).pow(hill) + numer
    raw = plan.emax * numer / denom.clamp_min(1e-6)
    rescued = raw * (1.0 - plan.mcl1_rescue * mcl1)
    return (rescued + combo_boost).clamp(0.0, 0.999)


def step_populations(
    sensitive: torch.Tensor,
    resistant: torch.Tensor,
    dose_mg: torch.Tensor,
    combo_idx: torch.Tensor,
    growth_s: torch.Tensor,
    growth_r: torch.Tensor,
    ec50_mg: torch.Tensor,
    mcl1: torch.Tensor,
    resist_prop: torch.Tensor,
    plan: SimPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    boost = torch.where(
        combo_idx > 0,
        torch.full_like(sensitive, plan.rituximab_boost),
        torch.zeros_like(sensitive),
    )
    effect = drug_effect(dose_mg, ec50_mg, mcl1, boost, plan)
    kill_s = effect
    kill_r = effect * plan.resistant_kill_ratio
    emergence = resist_prop * effect * sensitive
    next_s = (sensitive * torch.exp(growth_s - kill_s) - emergence).clamp_min(0.0)
    next_r = resistant * torch.exp(growth_r - kill_r) + emergence
    return next_s, next_r


def response_category(ratio: torch.Tensor, plan: SimPlan) -> torch.Tensor:
    cat = torch.full_like(ratio, 2.0)
    cat = torch.where(ratio < plan.pr_threshold, torch.full_like(ratio, 1.0), cat)
    cat = torch.where(ratio < plan.cr_threshold, torch.zeros_like(ratio), cat)
    cat = torch.where(ratio > plan.pd_threshold, torch.full_like(ratio, 3.0), cat)
    return cat
