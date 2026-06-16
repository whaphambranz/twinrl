from __future__ import annotations

import numpy as np

from twinrl_onco.debrief.runner import EvalBundle


def therapy_outcomes(bundle: EvalBundle) -> dict[str, float]:
    return {
        "median_ttp": float(np.median(bundle.ttp)),
        "pfs_12m": float(np.mean(bundle.pfs[12]) * 100.0),
        "pfs_24m": float(np.mean(bundle.pfs[24]) * 100.0),
        "drug_exposure": float(np.mean(bundle.drug_exposure) / max(bundle.dose_seq.shape[0], 1)),
        "qalm": float(np.mean(bundle.qalm)),
        "grade3_ae": float(np.mean(bundle.grade3_ae) * 100.0),
        "cr_rate": float(np.mean(bundle.complete_response) * 100.0),
    }


def therapy_report(bundle: EvalBundle) -> dict[str, float]:
    return therapy_outcomes(bundle)
