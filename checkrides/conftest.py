from __future__ import annotations

from pathlib import Path

import pytest

from twinrl_onco.flightplan import load_config
from twinrl_onco.groundstation.cohort import build_cohort
from twinrl_onco.simenv import behavior_rollout
from twinrl_onco.simenv.environment import TreatmentEnvironment
from twinrl_onco.telemetry import set_seed
from twinrl_onco.tower import build_model

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "flightplans" / "experiment" / "_smoke.toml"


@pytest.fixture
def config():
    return load_config(str(SMOKE))


@pytest.fixture
def cohort(config):
    set_seed(0)
    return build_cohort(config.data, config.model, seed=0)


@pytest.fixture
def model(config):
    set_seed(0)
    return build_model(config)


@pytest.fixture
def environment(model, cohort):
    return TreatmentEnvironment(
        cohort.biology,
        cohort.sim_params,
        cohort.clinical,
        model.config.model,
        model.config.sim,
        model.grid,
        seed=1,
    )


@pytest.fixture
def trajectory(environment):
    return behavior_rollout(environment, environment.sim.episode_length, seed=2)


@pytest.fixture
def features(model, trajectory):
    dose_mg, combo_idx = model.grid.decode(trajectory.action)
    return model.grid.features(dose_mg, combo_idx, trajectory.phase)
