from __future__ import annotations

import numpy as np

from twinrl_onco.debrief.metrics import (
    auprc,
    auroc,
    brier,
    brier_decomposition,
    sensitivity_at_specificity,
)
from twinrl_onco.debrief.runner import EvalBundle


def progression_report(bundle: EvalBundle) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for horizon in (12, 24):
        y = bundle.prog_true[horizon]
        s = bundle.prog_score[horizon]
        decomposition = brier_decomposition(y, s)
        out[f"{horizon}m"] = {
            "auroc": auroc(y, s),
            "auprc": auprc(y, s),
            "sens@90spec": sensitivity_at_specificity(y, s, 0.9),
            "brier": brier(y, s),
            "calibration": decomposition["calibration"],
            "refinement": decomposition["refinement"],
            "n_positive": float(np.sum(y)),
        }
    return out
