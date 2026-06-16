from __future__ import annotations

import torch

from twinrl_onco.debrief.runner import collect_eval
from twinrl_onco.debrief.therapy import therapy_report
from twinrl_onco.flightdeck.trainer import Trainer
from twinrl_onco.groundstation.folds import fold_indices


def test_training_two_steps_reduces_loss(config):
    result = Trainer(config, torch.device("cpu")).fit()
    assert len(result.history) >= 2
    assert result.history[-1]["world_loss"] < result.history[0]["world_loss"]


def test_evaluation_pipeline_runs(config, model, cohort):
    _, test_idx = fold_indices(cohort, config.data.n_folds, config.data.seed)[0]
    bundle = collect_eval(model, cohort, test_idx, seed=5)
    report = therapy_report(bundle)
    assert "median_ttp" in report
    assert bundle.prog_score[12].shape[0] == test_idx.numel()
