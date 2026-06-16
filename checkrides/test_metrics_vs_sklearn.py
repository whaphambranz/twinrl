from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from twinrl_onco.debrief.metrics import auprc, auroc, brier, regression_scores


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=200)
    s = 0.6 * y + rng.random(200)
    return y, s


def test_auroc_matches_sklearn():
    y, s = _data(1)
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-6


def test_auprc_matches_sklearn():
    y, s = _data(2)
    assert abs(auprc(y, s) - average_precision_score(y, s)) < 1e-6


def test_brier_matches_sklearn():
    y, s = _data(3)
    p = 1.0 / (1.0 + np.exp(-s))
    assert abs(brier(y, p) - brier_score_loss(y, p)) < 1e-9


def test_regression_scores_known():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = regression_scores(y, y)
    assert out["mae"] == 0.0 and abs(out["r2"] - 1.0) < 1e-9
