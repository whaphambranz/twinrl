from __future__ import annotations

import torch

from twinrl_onco.flightplan import SimPlan
from twinrl_onco.simenv.pkpd import drug_effect, step_populations
from twinrl_onco.simenv.reward import compose_reward, response_term


def test_response_reward_table():
    cats = torch.tensor([0.0, 1.0, 2.0, 3.0])
    rewards = response_term(cats)
    assert torch.allclose(rewards, torch.tensor([1.0, 0.5, 0.0, -1.0]))


def test_drug_effect_monotone_in_dose():
    plan = SimPlan()
    ec50 = torch.full((4,), plan.ec50_mg)
    mcl1 = torch.zeros(4)
    boost = torch.zeros(4)
    doses = torch.tensor([0.0, 100.0, 200.0, 400.0])
    eff = drug_effect(doses, ec50, mcl1, boost, plan)
    assert torch.all(eff[1:] >= eff[:-1])


def test_higher_dose_reduces_burden():
    plan = SimPlan()
    s = torch.tensor([1.0, 1.0])
    r = torch.tensor([0.05, 0.05])
    combo = torch.zeros(2, dtype=torch.long)
    gs = torch.full((2,), plan.growth_sensitive)
    gr = torch.full((2,), plan.growth_resistant)
    ec50 = torch.full((2,), plan.ec50_mg)
    mcl1 = torch.zeros(2)
    resist = torch.full((2,), plan.gly101val_rate)
    low = torch.tensor([0.0, 0.0])
    high = torch.tensor([400.0, 400.0])
    s_low, r_low = step_populations(s, r, low, combo, gs, gr, ec50, mcl1, resist, plan)
    s_high, r_high = step_populations(s, r, high, combo, gs, gr, ec50, mcl1, resist, plan)
    assert float((s_high + r_high).mean()) < float((s_low + r_low).mean())


def test_compose_reward_holiday_bonus():
    plan = SimPlan()
    cat = torch.tensor([0.0])
    ratio = torch.tensor([0.1])
    dose = torch.tensor([0.0])
    combo = torch.zeros(1, dtype=torch.long)
    reward = compose_reward(cat, ratio, dose, combo, 400.0, plan)
    assert float(reward) > 0.0
