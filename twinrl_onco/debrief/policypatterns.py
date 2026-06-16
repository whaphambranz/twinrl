from __future__ import annotations

import numpy as np

from twinrl_onco.debrief.runner import EvalBundle
from twinrl_onco.simenv.actions import ActionGrid


def policy_patterns(bundle: EvalBundle, grid: ActionGrid) -> dict[str, dict[str, float]]:
    phase_seq = bundle.phase_seq
    dose_seq = bundle.dose_seq
    out: dict[str, dict[str, float]] = {}
    for pid, name in enumerate(grid.phases):
        mask = phase_seq == pid
        if mask.sum() == 0:
            out[name] = {"avg_duration": 0.0, "modal_dose": 0.0, "share": 0.0}
            continue
        per_patient = mask.sum(axis=0).astype(np.float64)
        doses = dose_seq[mask]
        values, counts = np.unique(doses, return_counts=True)
        modal = float(values[int(np.argmax(counts))])
        out[name] = {
            "avg_duration": float(per_patient.mean()),
            "modal_dose": modal,
            "share": float(mask.mean()),
        }
    return out
