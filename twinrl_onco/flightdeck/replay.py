from __future__ import annotations

import torch

from twinrl_onco.simenv.environment import Trajectory


class ReplayBuffer:
    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self._fields: dict[str, torch.Tensor] | None = None

    def add(self, traj: Trajectory) -> None:
        block = {
            "gene": traj.gene,
            "flow": traj.flow,
            "clinical": traj.clinical,
            "action": traj.action,
            "phase": traj.phase,
            "reward": traj.reward,
            "continuation": traj.continuation,
            "progression": traj.progression,
        }
        if self._fields is None:
            self._fields = block
            return
        merged = {key: torch.cat([self._fields[key], block[key]], dim=1) for key in block}
        limit = self.capacity
        self._fields = {key: value[:, -limit * 4 :] for key, value in merged.items()}

    @property
    def populated(self) -> bool:
        return self._fields is not None

    def sample(
        self, n_patients: int, seq_len: int, generator: torch.Generator
    ) -> dict[str, torch.Tensor]:
        if self._fields is None:
            raise RuntimeError("buffer is empty")
        total_t = self._fields["gene"].shape[0]
        total_b = self._fields["gene"].shape[1]
        seq_len = min(seq_len, total_t)
        start = int(torch.randint(0, total_t - seq_len + 1, (1,), generator=generator))
        cols = torch.randint(0, total_b, (n_patients,), generator=generator)
        out: dict[str, torch.Tensor] = {}
        for key, value in self._fields.items():
            out[key] = value[start : start + seq_len].index_select(1, cols)
        return out
