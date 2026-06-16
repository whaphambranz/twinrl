from __future__ import annotations

from twinrl_onco.debrief.metrics import regression_scores
from twinrl_onco.debrief.runner import EvalBundle


def world_quality_report(bundle: EvalBundle) -> dict[str, dict[str, float]]:
    labels = {
        "gene": "gene_expression_top100",
        "lymph": "lymphocyte_count",
        "hgb": "hemoglobin",
        "plt": "platelet_count",
    }
    out: dict[str, dict[str, float]] = {}
    for key, name in labels.items():
        payload = bundle.world_quality.get(key)
        if payload is None:
            continue
        out[name] = regression_scores(payload["true"], payload["pred"])
    return out
