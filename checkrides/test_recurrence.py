from __future__ import annotations

import torch

from twinrl_onco.flightmodel.recurrence import GruCore, TransformerCore


def test_gru_deterministic():
    core = GruCore(4, 3, 8)
    h = core.reset(2, torch.device("cpu"))
    z = torch.randn(2, 4)
    a = torch.randn(2, 3)
    out_a = core.step(h, z, a)
    out_b = core.step(h, z, a)
    assert torch.allclose(out_a, out_b)


def test_gru_state_changes():
    core = GruCore(4, 3, 8)
    h = core.reset(2, torch.device("cpu"))
    z = torch.randn(2, 4)
    a = torch.randn(2, 3)
    assert not torch.allclose(core.step(h, z, a), h)


def test_transformer_core_runs_and_resets():
    core = TransformerCore(4, 3, 8, 2)
    h = core.reset(2, torch.device("cpu"))
    z = torch.randn(2, 4)
    a = torch.randn(2, 3)
    first = core.step(h, z, a)
    assert first.shape == (2, 8)
    core.reset(2, torch.device("cpu"))
    assert len(core._history) == 0
