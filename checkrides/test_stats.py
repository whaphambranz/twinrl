from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from twinrl_onco.debrief.stats import (
    bca_ci,
    bh_fdr,
    bootstrap_ci,
    delong,
    hosmer_lemeshow,
)


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    sa = 0.8 * y + rng.random(300)
    sb = 0.3 * y + rng.random(300)
    out = delong(y, sa, sb)
    assert abs(out["auc_a"] - roc_auc_score(y, sa)) < 1e-6
    assert abs(out["auc_b"] - roc_auc_score(y, sb)) < 1e-6


def test_delong_agrees_with_bootstrap_sign():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    sa = 0.9 * y + rng.random(400)
    sb = 0.2 * y + rng.random(400)
    out = delong(y, sa, sb)
    paired = np.stack([sa, sb], axis=1)

    def diff(rows):
        return roc_auc_score(y[rows.astype(int)], paired[rows.astype(int), 0]) - roc_auc_score(
            y[rows.astype(int)], paired[rows.astype(int), 1]
        )

    lo, hi = bootstrap_ci(np.arange(y.size).astype(float), diff, n_boot=200, seed=3)
    assert lo > 0 and (out["auc_a"] - out["auc_b"]) > 0


def test_bh_fdr_controls():
    pvals = np.array([0.001, 0.002, 0.5, 0.6, 0.7])
    out = bh_fdr(pvals, q=0.05)
    assert out["n_rejected"] == 2


def test_bca_within_bootstrap_neighbourhood():
    rng = np.random.default_rng(5)
    data = rng.normal(0.9, 0.05, size=120)
    lo_b, hi_b = bootstrap_ci(data, lambda x: float(np.mean(x)), n_boot=300, seed=7)
    lo_c, hi_c = bca_ci(data, lambda x: float(np.mean(x)), n_boot=300, seed=7)
    assert lo_c < hi_c
    assert abs(lo_c - lo_b) < 0.05 and abs(hi_c - hi_b) < 0.05


def test_hosmer_lemeshow_calibrated_is_insignificant():
    rng = np.random.default_rng(9)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    out = hosmer_lemeshow(y, p, groups=10)
    assert out["pvalue"] > 0.05
