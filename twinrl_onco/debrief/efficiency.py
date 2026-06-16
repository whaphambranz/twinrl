from __future__ import annotations

from twinrl_onco.tower import TwinRLOnco

MANUSCRIPT_REFERENCE: dict[str, float] = {
    "train_hours_per_fold": 4.2,
    "inference_ms_per_patient": 15.8,
    "gpu_memory_gb": 14.2,
    "params_millions": 9.6,
}


def parameter_count(model: TwinRLOnco) -> int:
    return sum(p.numel() for p in model.parameters())


def efficiency_report(model: TwinRLOnco) -> dict[str, float]:
    out = dict(MANUSCRIPT_REFERENCE)
    out["params_measured_millions"] = parameter_count(model) / 1e6
    return out
