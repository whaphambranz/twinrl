# Repository Plan

The package is laid out as a flight-simulator training facility: a simulated world that a pilot
(the hierarchical policy) learns to fly inside through imagined sorties.

## Tree

    twinrl_onco/
      telemetry.py            logging, seeding, atomic checkpoint i/o, device selection
      flightplan.py           msgspec Struct config schema + TOML(extends) loader + override merge
      tower.py                assembles the full TwinRL-Onco model from a resolved config
      groundstation/          data: GEO-shaped synthetic cohorts, RMA/ComBat within-fold prep,
                              stratified k-fold split, gene/feature selection
      flightmodel/            VRSSM world model
        encoders.py           multi-head gene-attention encoder + flow + clinical + MLP fuse
        recurrence.py         GRU deterministic core (Eq.2)
        latent.py             Gaussian posterior/prior + KL (Eq.3-4)
        decoder.py            observation model (Eq.5)
        predictors.py         reward/continuation/progression heads (Eq.6,7,15)
        elbo.py               ELBO + progression BCE (Eq.8,16)
      simenv/                 treatment simulation environment
        pkpd.py               two-compartment sensitive/resistant PK/PD + Emax dose-response
        reward.py             reward composition + response/tox/prog/qol terms (§III-A)
        environment.py        POMDP step / reset / observation function
      cockpit/                hierarchical policy
        metacontroller.py     pi_high over phases (Eq.9)
        lowlevel.py           pi_low phase-specific dosing (Eq.10)
        values.py             V_high / V_low (Eq.11-12)
        actorcritic.py        lambda-returns + actor/critic losses (Eq.1,13,14)
      imagination/            resistance-trajectory predictor
        rollout.py            latent imagination rollout (H steps)
        trajectory.py         M-rollout Monte-Carlo aggregation -> P(prog by t+H)
      flightdeck/             training
        trainer.py            Algorithm 1 loop (world model + imagination policy opt + collection)
        optim.py              optimiser/scheduler bundles, grad clipping, AMP, early stopping
        replay.py             experience replay buffer seeded from baseline covariates
      debrief/                evaluation + statistics
        metrics.py            AUROC/AUPRC/Brier/Sens@90Spec/regression scores
        stats.py              t-test, BH-FDR, Bonferroni, bootstrap/BCa, DeLong, HL, Cohen d, Fine-Gray
        progression.py        Table II / V / IX reports
        therapy.py            Table III therapy outcomes
        strata.py             Table IV stratified analysis
        worldquality.py       Table VI world-model quality
        efficiency.py         Table VII compute efficiency
        policypatterns.py     Table X policy-pattern extraction
      panel/                  click command group: train/evaluate/forecast/simulate/export

    flightplans/              TOML configs (model/ data/ train/ experiment/)
    checkrides/               pytest suite (varied test kinds)
    dispatch/                 shell launch scripts (single-GPU + torchrun examples)
    docs/                     project-context, implementation-map, deviations
    assets/                   placeholder for figures

## Config / CLI stack
msgspec frozen `Struct` schema, composable TOML via an `extends` chain (stdlib `tomllib`), dotted
`key=value` CLI overrides, dispatched by a `click` command group. No argparse, no OmegaConf,
no Hydra, no Pydantic-Typer.

## Dependencies (pinned)
torch==2.1.0, numpy==1.24.3, scikit-learn==1.3.0, scipy>=1.10, msgspec>=0.18, click>=8.1.
tomllib is stdlib (Python >=3.11). Dev: ruff, black, isort, mypy, pytest, pre-commit.

## Test plan (checkrides/)
shape; ELBO/KL correctness; posterior-collapse guard; GRU determinism; overfit-single-batch
(world model and policy); lambda-return hand-check; hierarchical-value consistency; MC-rollout
std ~ 0.28/sqrt(M) scaling; metric-vs-sklearn (AUROC/AUPRC/Brier); DeLong-vs-bootstrap agreement;
reward-composition correctness; PK/PD dose-monotonicity; gradient-flow incl. stop-gradient;
determinism/seed; config-extends merge; style-guard (ast+tokenize: no comments/docstrings/
forbidden phrases/no emoji over package + checkrides + README + Makefile); end-to-end training
smoke (2 steps, loss decreases).
