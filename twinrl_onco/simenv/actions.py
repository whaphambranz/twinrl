from __future__ import annotations

from dataclasses import dataclass

import torch

from twinrl_onco.flightplan import PolicyPlan

PHASE_DOSE_RULES: dict[str, tuple[float, ...]] = {
    "induction": (200.0, 400.0),
    "consolidation": (100.0, 200.0, 400.0),
    "maintenance": (0.0, 100.0, 200.0),
    "holiday": (0.0,),
}


@dataclass(frozen=True)
class ActionGrid:
    dose_levels_mg: tuple[float, ...]
    combos: tuple[str, ...]
    phases: tuple[str, ...]

    @property
    def n_doses(self) -> int:
        return len(self.dose_levels_mg)

    @property
    def n_combos(self) -> int:
        return len(self.combos)

    @property
    def n_actions(self) -> int:
        return self.n_doses * self.n_combos

    @property
    def feature_dim(self) -> int:
        return 1 + self.n_combos + len(self.phases)

    def decode(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dose_idx = action % self.n_doses
        combo_idx = action // self.n_doses
        dose_mg = torch.tensor(self.dose_levels_mg, device=action.device)[dose_idx]
        return dose_mg, combo_idx

    def phase_mask(self, phase_idx: int) -> torch.Tensor:
        name = self.phases[phase_idx]
        allowed = PHASE_DOSE_RULES.get(name, self.dose_levels_mg)
        mask = torch.zeros(self.n_actions, dtype=torch.bool)
        for combo in range(self.n_combos):
            for di, dose in enumerate(self.dose_levels_mg):
                if dose in allowed:
                    mask[combo * self.n_doses + di] = True
        return mask

    def all_phase_masks(self) -> torch.Tensor:
        return torch.stack([self.phase_mask(i) for i in range(len(self.phases))])

    def features(
        self, dose_mg: torch.Tensor, combo_idx: torch.Tensor, phase_idx: torch.Tensor
    ) -> torch.Tensor:
        top = float(max(self.dose_levels_mg))
        dose_feat = (dose_mg / top).unsqueeze(-1)
        combo_oh = torch.nn.functional.one_hot(combo_idx, self.n_combos).float()
        phase_oh = torch.nn.functional.one_hot(phase_idx, len(self.phases)).float()
        return torch.cat([dose_feat, combo_oh, phase_oh], dim=-1)


def grid_from_plan(plan: PolicyPlan) -> ActionGrid:
    return ActionGrid(
        dose_levels_mg=plan.dose_levels_mg,
        combos=plan.combos,
        phases=plan.phases,
    )
