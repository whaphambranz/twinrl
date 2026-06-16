# TwinRL-Onco - Pilot's Flight Logbook

A flight logbook for `twinrl_onco`: a world-model-empowered digital twin coupled to hierarchical
reinforcement learning that simulates venetoclax resistance trajectories and optimises adaptive
therapy schedules in chronic lymphocytic leukaemia. Each entry below is one logged operation; the
remarks carry the commands, tables, and reference readings. Every reported number is internal to
the treatment-simulation environment and needs prospective clinical validation before any clinical
reading.

| Logbook field | Reading |
|---|---|
| Airframe | `twinrl_onco` (VRSSM world model + meta/low-level controllers + MC-rollout predictor) |
| Powerplant | PyTorch 2.1.0, NumPy 1.24.3, scikit-learn 1.3.0 |
| Avionics | msgspec config Structs, composable TOML (`extends`), click flight panel |
| Station | IEEE Journal of Biomedical and Health Informatics |
| Endurance | 1x NVIDIA A100, 4.2 h per fold, 14.2 GB |
| Crew of record | five-fold cross-validation x five seeds |

---

## Entry 01 - Pre-flight (installation)

Pip:

    pip install -e .

Conda:

    conda env create -f environment.yml
    conda activate twinrl_onco
    pip install -e .

Docker:

    docker build -t twinrl_onco:0.1.0 .
    docker run --rm twinrl_onco:0.1.0 --help

The flight panel registers as `twinrl` and also runs as `python -m twinrl_onco.panel.console`.

## Entry 02 - Loading the cohort

Baseline molecular covariates come from three public GEO microarray series (Affymetrix HG-U133 Plus
2.0). They seed the digital-twin baseline only; resistance trajectories are produced by the
simulation environment, since no public per-patient venetoclax-resistance trajectory data exist. The
release ships GEO-shaped synthetic baselines (`groundstation/`) and a mechanistic two-compartment
PK/PD environment (`simenv/`), so no download is required to run the panel.

| Series | Samples | Endpoint | Access |
|---|---|---|---|
| GSE39671 | 130 | time-to-first-treatment | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39671 |
| GSE22762 | 107 | overall survival | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22762 |
| GSE50006 | 111 | response | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50006 |
| Combined | 348 | multiple | - |

RMA normalisation, location-scale batch correction, variance-based feature selection, and
standardisation are all fit inside training folds only (`groundstation/preprocess.py`).

## Entry 03 - Engine run-up (smoke)

    make smoke

or directly:

    twinrl train --config flightplans/experiment/_smoke.toml --checkpoint runs/smoke.pt

Two optimisation steps on tiny dimensions; the world-model loss decreases across the run.

## Entry 04 - Departure (training)

Main result:

    twinrl train --config flightplans/experiment/main.toml --checkpoint runs/twinrl_main.pt

One ablation (toggle a single component, Table V):

    twinrl train --config flightplans/experiment/ablation_no_stochastic.toml

Leave-one-series-out cross-dataset run (Table IX):

    twinrl train --config flightplans/experiment/crossdataset_hold_gse22762.toml

A hyperparameter override on any plan:

    twinrl train --config flightplans/experiment/main.toml --set model.beta=0.2 --set imagination.horizon=20

## Entry 05 - Instrument readback (evaluation)

    twinrl evaluate --config flightplans/experiment/main.toml --checkpoint runs/twinrl_main.pt
    twinrl forecast --config flightplans/experiment/main.toml --checkpoint runs/twinrl_main.pt --limit 12

Reference readings are the manuscript targets. The redistributed simulator reproduces the structure
and direction of these readings rather than the exact values, because the simulation calibration
(fourteen PK/PD parameters fit to non-public aggregate-trial curves) is not redistributable; see
`docs/deviations.md`.

| Endpoint | Command verb | Reference reading |
|---|---|---|
| 12-month progression AUROC | `evaluate` | 0.917 |
| 24-month progression AUROC | `evaluate` | 0.896 |
| Sensitivity at 90% specificity (12-month) | `evaluate` | 0.687 |
| Brier score (12-month) | `evaluate` | 0.119 |
| Median time-to-progression | `evaluate` | 26.5 months |
| 12-month progression-free survival | `evaluate` | 84.7% |
| Quality-adjusted life months | `evaluate` | 21.6 |
| Cross-dataset average AUROC | `evaluate` (crossdataset plan) | 0.885 |
| World-model 1-step gene R2 | `evaluate` | 0.871 |
| Per-patient resistance forecast | `forecast` | P(prog) at 12 and 24 months |

## Entry 06 - Endurance and fuel (compute)

| Quantity | Reading |
|---|---|
| Accelerator | 1x NVIDIA A100 |
| Training | 4.2 hours per cross-validation fold |
| Inference | 15.8 ms per patient (rollout plus progression head) |
| Memory | 14.2 GB during training |
| Parameters | 9.6 M |
| Protocol | five folds x five seeds |

## Entry 07 - Navigation log (package map)

| Stage | Module | Carries |
|---|---|---|
| Ground station | `groundstation/` | GEO-shaped cohort, within-fold preprocessing, folds |
| Flight model | `flightmodel/` | VRSSM encoders, GRU recurrence, latent, decoder, heads, ELBO |
| Simulation | `simenv/` | PK/PD dynamics, reward composition, POMDP step, action grid |
| Cockpit | `cockpit/` | meta-controller, low-level controller, hierarchical values, actor-critic |
| Imagination | `imagination/` | latent rollouts and Monte-Carlo resistance aggregation |
| Flight deck | `flightdeck/` | training loop, optimisers, replay buffer |
| Debrief | `debrief/` | metrics, statistics, per-table reports |
| Panel | `panel/` | click commands: train, evaluate, forecast, simulate, export |

Configuration plans live in `flightplans/` and compose through an `extends` chain
(`train/base` -> `data/base` -> `model/base` -> `experiment/main`). Checkrides (the test suite) live
in `checkrides/`.

## Entry 08 - Airmanship (limitations and ethics)

This research uses only publicly available, de-identified GEO gene-expression data; no new patient
data were collected and no human-subject data were obtained. All outcomes are simulation-internal
and are stated for hypothesis generation, not clinical prediction. The del(17p), TP53-mutated, and
Binet-C subgroups are small, so their intervals are wide and their readings are hypothesis-
generating. The learned holiday phase has no published counterpart and may be a simulation artefact.
External validation against per-patient longitudinal data is required before any clinical reading is
trusted.
