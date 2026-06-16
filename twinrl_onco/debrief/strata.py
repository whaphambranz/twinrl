from __future__ import annotations

import numpy as np
import torch

from twinrl_onco.debrief.metrics import auroc
from twinrl_onco.debrief.runner import EvalBundle
from twinrl_onco.groundstation.cohort import Cohort


def _subgroups(cohort: Cohort, index: torch.Tensor) -> dict[str, dict[str, np.ndarray]]:
    sel = {k: v.index_select(0, index).numpy() for k, v in cohort.labels.items()}
    binet = sel["binet"]
    groups: dict[str, dict[str, np.ndarray]] = {
        "IGHV-mutated": {"mask": sel["ighv_unmutated"] == 0},
        "IGHV-unmutated": {"mask": sel["ighv_unmutated"] == 1},
        "del17p-absent": {"mask": sel["del17p"] == 0},
        "del17p-present": {"mask": sel["del17p"] == 1},
        "TP53-wildtype": {"mask": sel["tp53"] == 0},
        "TP53-mutated": {"mask": sel["tp53"] == 1},
        "Binet-A": {"mask": binet == 0},
        "Binet-B": {"mask": binet == 1},
        "Binet-C": {"mask": binet == 2},
        "Age<65": {"mask": sel["age_ge65"] == 0},
        "Age>=65": {"mask": sel["age_ge65"] == 1},
    }
    return groups


def stratified_report(
    bundle: EvalBundle, cohort: Cohort, index: torch.Tensor, horizon: int = 12
) -> dict[str, dict[str, float]]:
    y = bundle.prog_true[horizon]
    s = bundle.prog_score[horizon]
    out: dict[str, dict[str, float]] = {}
    for name, payload in _subgroups(cohort, index).items():
        mask = payload["mask"]
        if mask.sum() < 2:
            continue
        out[name] = {
            "n": float(mask.sum()),
            "auroc": auroc(y[mask], s[mask]),
            "median_ttp": float(np.median(bundle.ttp[mask])),
            "pfs_12m": float(np.mean(bundle.pfs[12][mask]) * 100.0),
            "qalm": float(np.mean(bundle.qalm[mask])),
        }
    return out
