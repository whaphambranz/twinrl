from __future__ import annotations

import torch

from twinrl_onco.flightplan import SimPlan

RESPONSE_REWARD = torch.tensor([1.0, 0.5, 0.0, -1.0])


def response_term(category: torch.Tensor) -> torch.Tensor:
    table = RESPONSE_REWARD.to(category.device)
    return table[category.long()]


def toxicity_term(dose_mg: torch.Tensor, combo_idx: torch.Tensor, top_dose: float) -> torch.Tensor:
    severity = dose_mg / top_dose
    combo_pen = 0.2 * (combo_idx > 0).float()
    return -(severity + combo_pen)


def progression_term(ratio: torch.Tensor, plan: SimPlan) -> torch.Tensor:
    prob = torch.sigmoid(6.0 * (ratio - plan.pd_threshold))
    return -prob


def qol_term(dose_mg: torch.Tensor, ratio: torch.Tensor, plan: SimPlan) -> torch.Tensor:
    on_holiday = (dose_mg <= 0.0).float()
    in_response = (ratio < plan.pr_threshold).float()
    return on_holiday * in_response


def compose_reward(
    category: torch.Tensor,
    ratio: torch.Tensor,
    dose_mg: torch.Tensor,
    combo_idx: torch.Tensor,
    top_dose: float,
    plan: SimPlan,
) -> torch.Tensor:
    resp = response_term(category)
    tox = toxicity_term(dose_mg, combo_idx, top_dose)
    prog = progression_term(ratio, plan)
    qol = qol_term(dose_mg, ratio, plan)
    return plan.w_resp * resp + plan.w_tox * tox + plan.w_prog * prog + plan.w_qol * qol
