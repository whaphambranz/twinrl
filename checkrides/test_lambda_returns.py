from __future__ import annotations

import torch

from twinrl_onco.cockpit.actorcritic import discounted_return, lambda_returns


def test_lambda_one_equals_monte_carlo():
    reward = torch.tensor([[1.0], [2.0], [3.0]])
    cont = torch.ones(3, 1)
    values = torch.zeros(4, 1)
    gamma = 0.9
    out = lambda_returns(reward, cont, values, gamma, 1.0)
    expected_first = 1.0 + gamma * (2.0 + gamma * 3.0)
    assert abs(float(out[0, 0]) - expected_first) < 1e-5


def test_lambda_zero_is_one_step_td():
    reward = torch.tensor([[1.0], [1.0]])
    cont = torch.ones(2, 1)
    values = torch.tensor([[5.0], [7.0], [9.0]])
    gamma = 0.5
    out = lambda_returns(reward, cont, values, gamma, 0.0)
    assert abs(float(out[0, 0]) - (1.0 + gamma * 7.0)) < 1e-5


def test_continuation_zero_truncates():
    reward = torch.tensor([[1.0], [1.0]])
    cont = torch.tensor([[0.0], [1.0]])
    values = torch.ones(3, 1)
    out = lambda_returns(reward, cont, values, 0.9, 0.95)
    assert abs(float(out[0, 0]) - 1.0) < 1e-6


def test_discounted_return():
    reward = torch.tensor([[2.0], [2.0]])
    out = discounted_return(reward, 0.5)
    assert abs(float(out[0]) - (2.0 + 0.5 * 2.0)) < 1e-6
