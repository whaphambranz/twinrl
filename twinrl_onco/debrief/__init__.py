from __future__ import annotations

from twinrl_onco.debrief.efficiency import efficiency_report, parameter_count
from twinrl_onco.debrief.metrics import (
    auprc,
    auroc,
    brier,
    brier_decomposition,
    regression_scores,
    sensitivity_at_specificity,
)
from twinrl_onco.debrief.policypatterns import policy_patterns
from twinrl_onco.debrief.progression import progression_report
from twinrl_onco.debrief.runner import EvalBundle, collect_eval
from twinrl_onco.debrief.stats import (
    bca_ci,
    bh_fdr,
    bonferroni,
    bootstrap_ci,
    cohens_d,
    delong,
    fine_gray_subdistribution,
    hosmer_lemeshow,
    paired_test,
)
from twinrl_onco.debrief.strata import stratified_report
from twinrl_onco.debrief.therapy import therapy_outcomes, therapy_report
from twinrl_onco.debrief.worldquality import world_quality_report

__all__ = [
    "efficiency_report",
    "parameter_count",
    "auprc",
    "auroc",
    "brier",
    "brier_decomposition",
    "regression_scores",
    "sensitivity_at_specificity",
    "policy_patterns",
    "progression_report",
    "EvalBundle",
    "collect_eval",
    "bca_ci",
    "bh_fdr",
    "bonferroni",
    "bootstrap_ci",
    "cohens_d",
    "delong",
    "fine_gray_subdistribution",
    "hosmer_lemeshow",
    "paired_test",
    "stratified_report",
    "therapy_outcomes",
    "therapy_report",
    "world_quality_report",
]
