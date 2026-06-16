from __future__ import annotations

import torch

from twinrl_onco.flightdeck.trainer import Trainer


def _batch(trainer, trajectory):
    return {
        "gene": trajectory.gene,
        "flow": trajectory.flow,
        "clinical": trajectory.clinical,
        "action": trajectory.action,
        "phase": trajectory.phase,
        "reward": trajectory.reward,
        "continuation": trajectory.continuation,
        "progression": trajectory.progression,
    }


def test_world_step_populates_all_world_grads(config, trajectory):
    trainer = Trainer(config, torch.device("cpu"))
    trainer._world_step(_batch(trainer, trajectory))
    missing = [
        n for n, p in trainer.model.world.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert missing == []


def test_policy_step_populates_actor_and_critic_grads(config, trajectory):
    trainer = Trainer(config, torch.device("cpu"))
    trainer._policy_step(_batch(trainer, trajectory))
    actor = list(trainer.model.policy.low.parameters())
    critic = list(trainer.model.policy.low_value.parameters())
    assert all(p.grad is not None for p in actor)
    assert all(p.grad is not None for p in critic)
