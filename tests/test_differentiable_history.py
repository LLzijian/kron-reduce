import pytest
import torch

from kron_reduce.models.differentiable_history import (
    UnresolvedHistoryError,
    predict_from_history,
    solve_hidden,
)


def _ledger():
    entries = []

    def spend(operation, amount=1):
        entries.append((operation, amount))

    return spend, entries


def test_solve_hidden_exact_recovery_qr_closure_and_budget():
    design = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, -1.0]],
                          dtype=torch.float64)
    expected = torch.tensor([[0.3, -0.7], [1.2, 0.2]], dtype=torch.float64)
    rhs = expected @ design.T
    spend, calls = _ledger()

    result = solve_hidden(design, rhs, spend=spend)

    assert torch.allclose(result["hidden_start"], expected, atol=1e-13, rtol=1e-13)
    assert torch.allclose(result["qr_q"].T @ result["qr_q"], torch.eye(2, dtype=torch.float64),
                          atol=1e-13, rtol=1e-13)
    assert torch.allclose(result["qr_q"] @ result["qr_r"], design, atol=1e-13, rtol=1e-13)
    assert result["rank"] == 2
    assert calls == [("qr", 1), ("svd", 1), ("matrix_products", 1),
                     ("triangular_solves", 1)]


def test_solve_hidden_is_differentiable_in_design_and_rhs():
    design = torch.tensor([[1.2, 0.1], [0.2, 1.4], [-0.7, 0.3]],
                          dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor([[0.4, -0.2, 0.8]], dtype=torch.float64, requires_grad=True)

    def solve(d, y):
        return solve_hidden(d, y, spend=lambda *_: None)["hidden_start"]

    assert torch.autograd.gradcheck(solve, (design, rhs), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_rank_deficient_hidden_design_is_rejected_before_solves():
    design = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=torch.float64)
    rhs = torch.ones((2, 3), dtype=torch.float64)
    spend, calls = _ledger()

    with pytest.raises(UnresolvedHistoryError, match="rank/conditioning failed"):
        solve_hidden(design, rhs, spend=spend)

    assert calls == [("qr", 1), ("svd", 1)]


@pytest.mark.parametrize("case", ["design_shape", "rhs_shape", "empty", "float32", "design_nan", "rhs_inf"])
def test_solve_hidden_rejects_invalid_matrices(case):
    design = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    rhs = torch.ones((2, 3), dtype=torch.float64)
    if case == "design_shape":
        design = torch.ones((3,))
    elif case == "rhs_shape":
        rhs = torch.ones((2, 4), dtype=torch.float64)
    elif case == "empty":
        rhs = torch.empty((0, 3), dtype=torch.float64)
    elif case == "float32":
        design = design.float()
    elif case == "design_nan":
        design[0, 0] = torch.nan
    elif case == "rhs_inf":
        rhs[0, 0] = torch.inf

    with pytest.raises(ValueError, match="finite CPU float64 observation matrices required"):
        solve_hidden(design, rhs, spend=lambda *_: None)


class _TinyLinearModel:
    backbone_kind = "linear"
    n_nodes = 2
    latent_dimension = 2

    def __init__(self):
        self.calls = []

    @staticmethod
    def _tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    def rollout(self, initial, power, equilibrium, count, *, max_step, spend):
        spend("rollouts")
        self.calls.append((initial.detach().clone(), power.detach().clone(), count, max_step))
        t = torch.linspace(0.0, 1.0, count+1, dtype=torch.float64)[None, :, None]
        q0 = initial[:, None, :2]
        v0 = initial[:, None, 2:4]
        hidden = initial[:, None, 4:]
        q = q0+t*v0+.5*t.square()*hidden
        v = v0+t*hidden
        return torch.cat((q, v, hidden.expand(-1, count+1, -1)), dim=2)


def _history_problem():
    model = _TinyLinearModel()
    q0 = torch.tensor([[0.1, -0.2], [-0.05, 0.08]], dtype=torch.float64)
    v0 = torch.tensor([[0.03, -0.04], [0.02, 0.01]], dtype=torch.float64)
    hidden = torch.tensor([[0.4, -0.25], [-0.2, 0.35]], dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)[None, :, None]
    q = q0[:, None]+t*v0[:, None]+.5*t.square()*hidden[:, None]
    v = v0[:, None]+t*hidden[:, None]
    power = torch.zeros((2, 2), dtype=torch.float64)
    equilibrium = torch.zeros_like(power)
    scale = torch.ones(4, dtype=torch.float64)
    return model, q, v, power, equilibrium, scale, hidden


def test_predict_from_history_recovers_clean_state_and_resets_visible_origin():
    model, q, v, power, equilibrium, scale, expected_hidden = _history_problem()
    spend, calls = _ledger()

    result = predict_from_history(model, q, v, power, equilibrium, scale, 4,
                                  max_step=0.01, spend=spend)

    assert result["hidden_start"].shape == (2, 2)
    assert torch.allclose(result["hidden_start"], expected_hidden, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result["history"][:, 0, :2], q[:, 0], atol=0, rtol=0)
    assert torch.allclose(result["history"][:, 0, 2:4], v[:, 0], atol=0, rtol=0)
    assert torch.allclose(result["history"][:, 0, 4:], expected_hidden, atol=1e-12, rtol=1e-12)
    forecast_origin = model.calls[-1][0]
    assert torch.allclose(forecast_origin[:, :2], q[:, -1], atol=0, rtol=0)
    assert torch.allclose(forecast_origin[:, 2:4], v[:, -1], atol=0, rtol=0)
    assert torch.allclose(forecast_origin[:, 4:], result["history"][:, -1, 4:], atol=0, rtol=0)
    assert result["forecast"].shape == (2, 5, 6)
    assert [call[2:] for call in model.calls] == [(10, 0.01), (4, 0.01)]
    assert calls == [("rollouts", 1), ("qr", 1), ("svd", 1), ("matrix_products", 1),
                     ("triangular_solves", 1), ("matrix_products", 11), ("rollouts", 1)]
