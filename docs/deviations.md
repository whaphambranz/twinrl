# Deviations

Deviations from the manuscript, each with a paper anchor and a justification.

## D1 — Treatment-simulation environment ships as a self-contained mechanistic model
Anchor: §IV-C-2, §IV-A. No public per-patient venetoclax-resistance trajectory data exist; the
manuscript itself derives every reported outcome from a simulation environment calibrated to
aggregate clinical-trial curves. `simenv/` re-creates that environment as a deterministic
two-compartment (sensitive/resistant) PK/PD model with an Emax dose-response, Gly101Val emergence,
and MCL-1 rescue. Baseline covariates come from GEO-shaped synthetic cohorts (`groundstation/`).
This matches the paper's protocol and is not a silent approximation.

## D2 — Reported numeric targets are the manuscript's, not locally measured
Anchor: Tables II-XI, §V-A. The 14 PK/PD calibration parameters were fit to non-public aggregate
trial curves, so the redistributed simulator reproduces the structure and qualitative trends
(IGHV-mutated favourable, del(17p)/TP53 less favourable, longer horizon harder) rather than the
exact AUROC/TTP values. The README quotes the manuscript values as the reference targets; the
synthetic run will not match them numerically. Table VII timings are quoted from the manuscript.

## D3 — Cross-dataset validation as leave-one-series-out on tagged cohorts
Anchor: Table IX. Implemented by tagging each synthetic patient with its source GEO series and
holding one series out (`data.holdout_dataset`). This reproduces the generalisation-gap protocol
exactly; absolute gaps are simulation-internal per D2.

## D4 — Transformer-backbone ablation depth
Anchor: §III-C-1, Table V ("Transformer backbone"). The manuscript only reports the comparison
outcome, not the ablation's full architecture, beyond noting a 4-layer Transformer was tried in
prose. The ablation backbone here is a 2-layer self-attention encoder over accumulated
latent/action tokens, selected by `model.backbone = "transformer"`. The main model is the GRU
recurrence (Eq. 2) as specified.

## D5 — ComBat realised as parametric location-scale correction
Anchor: §IV-B. Batch-effect correction is a per-batch mean/variance adjustment to the pooled
location and scale, fit on training-fold patients only. The empirical-Bayes shrinkage of the full
ComBat estimator is omitted; the leakage-free within-fold estimation protocol is preserved exactly.
