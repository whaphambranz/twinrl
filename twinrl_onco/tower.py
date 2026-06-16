from __future__ import annotations

import torch
from torch import nn

from twinrl_onco.cockpit.policy import HierarchicalPolicy
from twinrl_onco.flightmodel.vrssm import VRSSM, WorldState
from twinrl_onco.flightplan import Config
from twinrl_onco.groundstation.cohort import Cohort
from twinrl_onco.simenv.actions import ActionGrid, grid_from_plan
from twinrl_onco.simenv.environment import TreatmentEnvironment


class TwinRLOnco(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.grid: ActionGrid = grid_from_plan(config.policy)
        self.world = VRSSM(config.model, self.grid)
        self.policy = HierarchicalPolicy(config.model, config.policy, self.grid)

    def encode(
        self,
        gene: torch.Tensor,
        flow: torch.Tensor,
        clinical: torch.Tensor,
        action_features: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> WorldState:
        return self.world.observe(gene, flow, clinical, action_features, generator)


def build_model(config: Config) -> TwinRLOnco:
    return TwinRLOnco(config)


def make_environment(
    model: TwinRLOnco, cohort: Cohort, index: torch.Tensor, seed: int
) -> TreatmentEnvironment:
    return TreatmentEnvironment(
        biology=cohort.biology.index_select(0, index),
        sim_params={k: v.index_select(0, index) for k, v in cohort.sim_params.items()},
        clinical_static=cohort.clinical.index_select(0, index),
        model=model.config.model,
        sim=model.config.sim,
        grid=model.grid,
        seed=seed,
    )
