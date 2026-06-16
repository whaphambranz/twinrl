from __future__ import annotations

from dataclasses import dataclass

import torch

from twinrl_onco.flightplan import ModelPlan, SimPlan
from twinrl_onco.simenv.actions import ActionGrid
from twinrl_onco.simenv.pkpd import response_category, step_populations
from twinrl_onco.simenv.reward import compose_reward
from twinrl_onco.telemetry import generator_for

PATHWAY = 64
DYN_BASE = 10


@dataclass
class Trajectory:
    gene: torch.Tensor
    flow: torch.Tensor
    clinical: torch.Tensor
    action: torch.Tensor
    phase: torch.Tensor
    reward: torch.Tensor
    continuation: torch.Tensor
    progression: torch.Tensor


class TreatmentEnvironment:
    def __init__(
        self,
        biology: torch.Tensor,
        sim_params: dict[str, torch.Tensor],
        clinical_static: torch.Tensor,
        model: ModelPlan,
        sim: SimPlan,
        grid: ActionGrid,
        seed: int = 0,
    ) -> None:
        self.biology = biology
        self.params = sim_params
        self.clinical_static = clinical_static.clone()
        self.model = model
        self.sim = sim
        self.grid = grid
        self.n = biology.shape[0]
        gen = generator_for(seed)
        bdim = biology.shape[1]
        self.gene_loadings = torch.randn(bdim, model.gene_dim, generator=gen) * 0.5
        self.flow_loadings = torch.randn(bdim, model.flow_dim, generator=gen) * 0.5
        self.sensitive = torch.zeros(self.n)
        self.resistant = torch.zeros(self.n)
        self.baseline = torch.ones(self.n)

    def reset(self) -> dict[str, torch.Tensor]:
        self.sensitive = torch.full((self.n,), self.sim.baseline_burden)
        self.resistant = 0.02 * torch.sigmoid(self.biology[:, 5])
        self.baseline = (self.sensitive + self.resistant).clamp_min(1e-3)
        return self._observe()

    def _observe(self) -> dict[str, torch.Tensor]:
        burden = self.sensitive + self.resistant
        ratio = burden / self.baseline
        resist_frac = self.resistant / burden.clamp_min(1e-6)
        gene = torch.tanh(self.biology @ self.gene_loadings)
        bump = (0.8 * resist_frac + 0.4 * (ratio - 1.0)).unsqueeze(-1)
        gene[:, :PATHWAY] = gene[:, :PATHWAY] + bump
        flow = torch.tanh(self.biology @ self.flow_loadings)
        flow[:, :32] = flow[:, :32] + (0.5 * burden).unsqueeze(-1)
        clinical = self.clinical_static.clone()
        if clinical.shape[1] > DYN_BASE + 3:
            clinical[:, DYN_BASE] = burden
            clinical[:, DYN_BASE + 1] = 1.0 / (1.0 + 0.5 * burden)
            clinical[:, DYN_BASE + 2] = 1.0 / (1.0 + 0.3 * burden)
            clinical[:, DYN_BASE + 3] = resist_frac
        return {"gene": gene, "flow": flow, "clinical": clinical}

    def step(
        self, dose_mg: torch.Tensor, combo_idx: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        self.sensitive, self.resistant = step_populations(
            self.sensitive,
            self.resistant,
            dose_mg,
            combo_idx,
            self.params["growth_sensitive"],
            self.params["growth_resistant"],
            self.params["ec50_mg"],
            self.params["mcl1"],
            self.params["resist_prop"],
            self.sim,
        )
        burden = self.sensitive + self.resistant
        ratio = burden / self.baseline
        category = response_category(ratio, self.sim)
        top = float(max(self.grid.dose_levels_mg))
        reward = compose_reward(category, ratio, dose_mg, combo_idx, top, self.sim)
        progression = (ratio > self.sim.pd_threshold).float()
        continuation = 1.0 - progression
        return self._observe(), reward, continuation, progression


def clinical_standard_schedule(length: int, grid: ActionGrid) -> torch.Tensor:
    phases = []
    for cycle in range(length):
        if cycle < 4:
            phases.append(grid.phases.index("induction"))
        else:
            phases.append(grid.phases.index("maintenance"))
    return torch.tensor(phases, dtype=torch.long)


def behavior_rollout(env: TreatmentEnvironment, length: int, seed: int) -> Trajectory:
    gen = generator_for(seed)
    grid = env.grid
    masks = grid.all_phase_masks()
    schedule = clinical_standard_schedule(length, grid)
    genes, flows, clinicals = [], [], []
    actions, phases, rewards, conts, progs = [], [], [], [], []
    obs = env.reset()
    for t in range(length):
        phase_idx = schedule[t].expand(env.n)
        valid = masks[phase_idx].float()
        probs = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        action = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        dose_mg, combo_idx = grid.decode(action)
        genes.append(obs["gene"])
        flows.append(obs["flow"])
        clinicals.append(obs["clinical"])
        obs, reward, cont, prog = env.step(dose_mg, combo_idx)
        actions.append(action)
        phases.append(phase_idx)
        rewards.append(reward)
        conts.append(cont)
        progs.append(prog)
    return Trajectory(
        gene=torch.stack(genes),
        flow=torch.stack(flows),
        clinical=torch.stack(clinicals),
        action=torch.stack(actions),
        phase=torch.stack(phases),
        reward=torch.stack(rewards),
        continuation=torch.stack(conts),
        progression=torch.stack(progs),
    )
