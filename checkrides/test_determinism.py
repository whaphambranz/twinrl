from __future__ import annotations

import torch

from twinrl_onco.flightdeck.trainer import Trainer
from twinrl_onco.groundstation.cohort import build_cohort
from twinrl_onco.telemetry import set_seed


def test_cohort_reproducible(config):
    set_seed(0)
    a = build_cohort(config.data, config.model, seed=0)
    set_seed(123)
    b = build_cohort(config.data, config.model, seed=0)
    assert torch.allclose(a.gene, b.gene)
    assert torch.allclose(a.sim_params["ec50_mg"], b.sim_params["ec50_mg"])


def test_training_reproducible(config):
    r1 = Trainer(config, torch.device("cpu")).fit()
    r2 = Trainer(config, torch.device("cpu")).fit()
    assert abs(r1.final_loss - r2.final_loss) < 1e-4
