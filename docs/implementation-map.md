# Implementation Map

Traceability from manuscript items to code. The source tree carries no inline comments or
docstrings; this file is the single place where paper provenance lives.

## Equations and algorithm

| paper item | where | file (twinrl_onco/...) | symbol |
|---|---|---|---|
| Eq. (1) discounted return J(pi) | §III-A | cockpit/actorcritic.py | `discounted_return` |
| reward composition r_t (weighted sum) | §III-A | simenv/reward.py | `compose_reward` |
| response/tox/prog/qol terms | §III-A | simenv/reward.py | `response_term`, `toxicity_term`, `progression_term`, `qol_term` |
| Eq. (2) deterministic state h_t = f_theta(h,z,a) | §III-C-1, Fig.2B | flightmodel/recurrence.py | `GruCore.step` |
| Eq. (3) posterior q_phi(z|h,o) | §III-C-1 | flightmodel/latent.py | `Posterior` |
| Eq. (4) prior p_theta(z|h) | §III-C-1 | flightmodel/latent.py | `Prior` |
| Eq. (5) observation model o_hat = g_theta(h,z) | §III-C-2 | flightmodel/decoder.py | `ObservationDecoder` |
| Eq. (6) reward predictor r_hat | §III-C-3 | flightmodel/predictors.py | `RewardHead` |
| Eq. (7) continuation predictor c_hat | §III-C-3 | flightmodel/predictors.py | `ContinuationHead` |
| Eq. (8) VRSSM ELBO (recon - beta KL + aux) | §III-C-4 | flightmodel/elbo.py | `vrssm_objective` |
| Eq. (9) meta-controller pi_high (softmax over phases) | §III-D-1 | cockpit/metacontroller.py | `MetaController` |
| Eq. (10) low-level pi_low (phase-specific) | §III-D-2 | cockpit/lowlevel.py | `LowLevelController` |
| Eq. (11) high-level value V_high | §III-D-3 | cockpit/values.py | `HighValue` |
| Eq. (12) low-level value V_low | §III-D-3 | cockpit/values.py | `LowValue` |
| Eq. (13) lambda-returns | §III-D-4 | cockpit/actorcritic.py | `lambda_returns` |
| Eq. (14) high actor loss (stop-gradient advantage) | §III-D-4 | cockpit/actorcritic.py | `actor_loss` |
| Eq. (15) progression head p_prog = sigmoid(.) | §III-D-5 | flightmodel/predictors.py | `ProgressionHead` |
| Eq. (16) progression BCE loss L_prog | §III-D-5 | flightmodel/elbo.py | `progression_loss` |
| Algorithm 1 training procedure | §III-E | flightdeck/trainer.py | `Trainer.fit` |
| imagination rollout / MC aggregation (M, H) | Fig.1, Fig.2F-G | imagination/rollout.py, imagination/trajectory.py | `imagine`, `aggregate_progression` |
| std approx 0.28/sqrt(M) | §IV-D | imagination/trajectory.py | `rollout_std` |

## Tables

| table | content | file | symbol |
|---|---|---|---|
| I | dataset summary / baseline covariates | groundstation/cohort.py | `CohortSpec`, `dataset_table` |
| II | progression prediction (AUROC/AUPRC/Sens@90Spec/Brier @12,24mo) | debrief/progression.py | `progression_report` |
| III | adaptive-therapy optimisation (TTP/PFS/QALM/...) | debrief/therapy.py | `therapy_report` |
| IV | stratified analysis by risk group | debrief/strata.py | `stratified_report` |
| V | ablation study | flightplans/experiment/ablation_*.toml + debrief/progression.py | - |
| VI | world-model prediction quality (MAE/RMSE/R2/Corr) | debrief/worldquality.py | `world_quality_report` |
| VII | computational efficiency | debrief/efficiency.py | `efficiency_report` |
| VIII | hyperparameter sensitivity | flightplans/experiment/sensitivity_*.toml | - |
| IX | cross-dataset validation | flightplans/experiment/crossdataset_*.toml + debrief/progression.py | - |
| X | learned treatment-policy patterns | debrief/policypatterns.py | `policy_patterns` |
| XI | statistical significance / CIs | debrief/stats.py | `paired_test`, `bootstrap_ci`, `bca_ci`, `delong` |

## Metrics and statistics

| item | file | symbol |
|---|---|---|
| AUROC, AUPRC, Brier, Sens@90Spec | debrief/metrics.py | `auroc`, `auprc`, `brier`, `sensitivity_at_specificity` |
| Brier decomposition (calibration/refinement) | debrief/metrics.py | `brier_decomposition` |
| Median TTP, 12/24M-PFS, QALM, drug exposure, AE, CR | debrief/therapy.py | `therapy_outcomes` |
| MAE/RMSE/R2/Pearson | debrief/metrics.py | `regression_scores` |
| paired t-test + Benjamini-Hochberg FDR | debrief/stats.py | `paired_test`, `bh_fdr` |
| Bonferroni correction | debrief/stats.py | `bonferroni` |
| bootstrap CI (1000) / BCa CI (2000) | debrief/stats.py | `bootstrap_ci`, `bca_ci` |
| DeLong test | debrief/stats.py | `delong` |
| Hosmer-Lemeshow | debrief/stats.py | `hosmer_lemeshow` |
| Cohen's d | debrief/stats.py | `cohens_d` |
| Fine-Gray competing-risk sensitivity | debrief/stats.py | `fine_gray_subdistribution` |

## Ablation coverage (Table V)

World-model: w/o stochastic z_t; w/o KL; w/o gene attention; w/o multi-head encoder; Transformer
backbone. Hierarchical-policy: flat policy; fixed phase duration; single low-level policy.
Progression-predictor: w/o MC rollouts; single rollout (M=1); w/o trajectory aggregation.
Training: no imagination (model-free); short horizon (H=5); no early stopping. Each maps to a
`flightplans/experiment/ablation_*.toml` toggling one component flag consumed by `tower.assemble`.

## Code-availability statement (verbatim from the manuscript; kept here only, never in README)

"Code and simulation specifications will be made available upon reasonable request."

## Citation (kept here only, never in README)

Z. Wang, C. Ge, Y. Hu, D. Bai, W. Zhang, "TwinRL-Onco: A World Model-Empowered Digital Twin
Framework with Hierarchical Reinforcement Learning for Venetoclax Resistance Trajectory Prediction
and Adaptive Therapy Optimization in Chronic Lymphocytic Leukemia," IEEE Journal of Biomedical and
Health Informatics, 2025.
