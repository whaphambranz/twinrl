from __future__ import annotations


def test_world_state_shapes(model, trajectory, features):
    state = model.world.observe(trajectory.gene, trajectory.flow, trajectory.clinical, features)
    steps, batch = trajectory.gene.shape[0], trajectory.gene.shape[1]
    assert state.hidden.shape == (steps, batch, model.config.model.hidden_dim)
    assert state.latent.shape == (steps, batch, model.config.model.latent_dim)
    assert state.post_mean.shape == state.prior_mean.shape


def test_policy_action_shapes(model, trajectory, features):
    state = model.world.observe(trajectory.gene, trajectory.flow, trajectory.clinical, features)
    decision = model.policy.act(state.hidden[0], state.latent[0])
    batch = trajectory.gene.shape[1]
    assert decision["action"].shape == (batch,)
    assert decision["phase"].shape == (batch,)
    assert decision["features"].shape == (batch, model.grid.feature_dim)


def test_progression_logits_shape(model, trajectory, features):
    state = model.world.observe(trajectory.gene, trajectory.flow, trajectory.clinical, features)
    logits = model.world.progression_logits(state)
    assert logits.shape == trajectory.progression.shape
