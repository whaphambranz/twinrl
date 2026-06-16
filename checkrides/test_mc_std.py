from __future__ import annotations

from twinrl_onco.imagination.trajectory import rollout_std


def test_rollout_std_matches_manuscript():
    targets = {10: 0.087, 50: 0.041, 200: 0.020, 500: 0.013}
    for m, expected in targets.items():
        assert abs(rollout_std(m) - expected) < 0.005


def test_rollout_std_monotone():
    assert rollout_std(10) > rollout_std(50) > rollout_std(200) > rollout_std(500)
