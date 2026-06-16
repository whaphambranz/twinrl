# Project Context

    project_name       : twinrl_onco                                   [HIGH]
    domain             : computational oncology - in-silico CLL        [HIGH]
                         venetoclax resistance-trajectory simulation
                         and adaptive-therapy optimisation
                         (model-based RL / digital twin)
    framework          : PyTorch 2.1.0 (+ NumPy 1.24.3, sklearn 1.3.0) [HIGH]
    venue              : IEEE J. Biomedical and Health Informatics      [HIGH]
    primary_datasets   : 3 GEO microarray cohorts (see §6)              [HIGH]
    compute_target     : 1x NVIDIA A100, 4.2 h/fold, 14.2 GB           [HIGH]
    hparams_reference  : Table VIII + Methods prose (see §8)            [HIGH]
    supp_path          : none on disk                                   [HIGH]
    extra_signals      : 1 algorithm box; no released checkpoints;
                         public de-identified data only; SI = training
                         curves over 5 seeds (not on disk)

    NEEDS_USER_DECISION: 0

---

## 1. project_name
`twinrl_onco` — the framework names itself TwinRL-Onco (title; Abstract L26). snake_case package
name keeps the team identity. [HIGH]

## 2. supp_path
`none`. The manuscript references a "supplementary repository" holding per-loss training curves
over 5 seeds (§IV-C-3), but no supplementary file ships alongside the PDF. [HIGH]

## 3. domain
In-silico computational oncology for chronic lymphocytic leukaemia (CLL). The framework predicts
patient-specific venetoclax (BCL-2 inhibitor) resistance trajectories and optimises adaptive
therapy schedules through a world-model-driven digital twin coupled to hierarchical reinforcement
learning. Every reported number is simulation-internal (Abstract; §V-A). [HIGH]

## 4. framework
PyTorch 2.1.0, NumPy 1.24.3, scikit-learn 1.3.0 — stated verbatim in the Reproducibility
Statement. The recurrent latent-dynamics core (GRU), variational latent state, actor-critic policy
optimisation with stop-gradient and lambda-returns are all native `torch.nn` / `torch.autograd`
constructs. [HIGH]

## 5. venue
IEEE Journal of Biomedical and Health Informatics (J-BHI) — running header on every page. [HIGH]

## 6. primary_datasets
Baseline molecular covariates come from three publicly accessible GEO CLL gene-expression cohorts
(§IV-A, Table I). None carry longitudinal venetoclax-resistance trajectories — they seed the
digital-twin baseline only; trajectories are produced by the treatment-simulation environment.

| name | samples | platform | endpoint | IGHV-U | del(17p) | access |
|---|---|---|---|---|---|---|
| GSE39671 | 130 | Affymetrix HG-U133 Plus 2.0 | TTFT | 49.2% | 8.5% | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39671 |
| GSE22762 | 107 | Affymetrix HG-U133 Plus 2.0 | OS | 52.3% | 11.2% | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22762 |
| GSE50006 | 111 | Affymetrix HG-U133 Plus 2.0 | Response | 46.8% | 9.0% | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50006 |
| Combined | 348 | - | Multiple | 49.4% | 9.5% | - |

License/access: GEO is an NCBI public-domain repository; the three series are open-access,
no registration or DUA required. All three accession URLs were checked and return HTTP 200
(verified 2026-06-16). [HIGH]

Treatment-simulation environment: NOT a downloadable dataset. It is a mechanistic two-compartment
(sensitive/resistant) pharmacokinetic-pharmacodynamic model with an Emax dose-response, calibrated
to aggregate clinical-trial curves (Phase-I PK/PD; Gly101Val emergence rate; MURANO 24-month MCL-1
kinetics). 14 free parameters (8 least-squares fit, 4 from PK literature, 2 manual). This release
ships a deterministic, epidemiology-anchored re-implementation of that environment plus GEO-shaped
synthetic baseline cohorts, because no public per-patient venetoclax-resistance trajectory data
exist (§IV-C-2). [HIGH]

## 7. compute_target
1x NVIDIA A100. Training 4.2 h per cross-validation fold; inference 15.8 ms per patient (policy
rollout + progression prediction); 14.2 GB GPU memory during training; 9.6M parameters (Table VII).
Protocol: 5-fold cross-validation x 5 seeds (§IV-C, §IV-D). [HIGH]

## 8. hparams_reference
Table VIII (sensitivity sweep, default in bold) plus Methods prose:

    latent dim dz        : 64                 (Table VIII; Fig.2B)
    hidden dim dh        : 256                (Table VIII; Fig.2B)
    beta (KL weight)     : 0.1                (§III-C-4; Table VIII)
    imagination horizon H: 15                 (§III-E; Fig.2F)
    MC rollouts M        : 50                 (Fig.1; §IV-D)
    lambda (TD)          : 0.95               (§III-D-4, Eq.13)
    learning rate        : 3e-4               (Table VIII default)
    batch size           : 32                 (Table VIII)
    gene-expression dim  : 2247               (Fig.1; Fig.2A)
    flow-cytometry dim   : 256                (Fig.2A)
    clinical-feature dim : 32                 (Fig.2A)
    fused embed dim      : 256                (Fig.2A, e_t = MLPfuse)
    reward weights       : w_resp=1.0, w_tox=0.3, w_prog=0.5, w_qol=0.2   (§III-A)
    response reward      : CR +1.0 / PR +0.5 / SD 0.0 / PD -1.0           (§III-A)
    phases               : induction, consolidation, maintenance, holiday (§III-D-1)
    cv / seeds           : 5-fold / 5 seeds   (§IV-C, §IV-D)
    bootstrap CI         : 1000 (general), BCa 2000 (subgroups)           (§IV-C-4, Repro.)

[HIGH]

## 9. extra_signals
- Algorithm boxes: 1 (Algorithm 1, full training procedure).
- Released checkpoints: none ("available upon reasonable request").
- Proprietary tokenizer: none.
- SI-only experiments: per-loss training curves over 5 seeds (world model / actor / critic).
- Code-availability statement (verbatim, kept only in docs/implementation-map.md, NOT in README):
  "Code and simulation specifications will be made available upon reasonable request."
- Ethics: only publicly available de-identified GEO expression data; no new patient data; no
  human-subject data collection (Ethical Considerations).
- Honesty flags carried from the manuscript: all metrics are simulation-internal and require
  prospective clinical validation; del(17p)/TP53-mutated/Binet-C subgroups are small (wide CIs,
  hypothesis-generating); the holiday phase has no published counterpart and may be a simulation
  artefact.
