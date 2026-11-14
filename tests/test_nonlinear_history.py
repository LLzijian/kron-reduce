import numpy as np
import pytest
import torch

from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.models.differentiable_history import UnresolvedHistoryError
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_history import (
    history_with_tangent,
    predict_from_history,
    tangent_rhs,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


def _project(value, mass):
    return value - (value @ mass)[:, None] / mass.sum()


def _model(*, backbone="sine", port=(0.3, 0.4, 0.5), tau=(0.08, 0.2, 0.7),
           train_stiffness=False):
    mass = np.array([1.0, 1.2, 0.8], dtype=np.float64)
    closure = NonlinearPassiveClosure(
        mass=mass,
        edge_stiffness=np.array([1.5, 1.2, 1.8]),
        damping=0.02*np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]),
        coupling=np.eye(3)-np.ones((3, 3))/3,
        latent_skew=np.zeros((3, 3)),
        latent_dissipation=np.eye(3),
        backbone=backbone,
    )
    return DiagonalMemoryModel(
        closure,
        port_kind="local",
        port=np.asarray(port, dtype=np.float64),
        tau=np.asarray(tau, dtype=np.float64),
        encoder=np.zeros((3, 24), dtype=np.float64),
        feature_scale=np.ones(24, dtype=np.float64),
        train_stiffness=train_stiffness,
    )


def _problem(model, *, hidden=(0.08, -0.05, 0.03)):
    mass = model.mass
    q0 = _project(torch.tensor([[0.04, -0.02, 0.01]], dtype=torch.float64), mass)
    v0 = _project(torch.tensor([[0.03, -0.01, 0.02]], dtype=torch.float64), mass)
    z0 = torch.tensor([hidden], dtype=torch.float64)
    initial = torch.cat((q0, v0, z0), dim=1)
    raw_power = torch.tensor([[0.12, -0.04, -0.08]], dtype=torch.float64)
    power = raw_power-raw_power.mean(dim=1, keepdim=True)
    equilibrium = _project(torch.tensor([[0.06, -0.025, -0.01]], dtype=torch.float64), mass)
    path = model.rollout(initial, power, equilibrium, 10, max_step=0.005, linear_fast=False)
    return path[:, :, :3], path[:, :, 3:6], power, equilibrium, initial


def _spend(*_args):
    return None


def test_tangent_rhs_matches_directional_finite_difference_and_history_matches_rk4():
    model = _model()
    q, v, power, equilibrium, initial = _problem(model)
    state = torch.cat((q[:, 3], v[:, 3], initial[:, 6:]), dim=1)
    tangent = torch.tensor(
        [[[0.1, -0.2, 0.05], [0.2, 0.1, -0.1], [-0.1, 0.05, 0.2],
          [0.3, 0.1, -0.2], [-0.2, 0.25, 0.05], [0.1, -0.15, 0.3],
          [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )

    _, tangent_rhs_value = tangent_rhs(model, state, tangent, power, equilibrium, spend=_spend)
    epsilon = 1e-6
    for column in range(model.latent_dimension):
        direction = tangent[:, :, column]
        plus = model.rhs(state + epsilon*direction, power, equilibrium)
        minus = model.rhs(state - epsilon*direction, power, equilibrium)
        finite_difference = (plus-minus)/(2*epsilon)
        torch.testing.assert_close(tangent_rhs_value[:, :, column], finite_difference,
                                   rtol=2e-7, atol=2e-9)

    integrated, response = history_with_tangent(model, initial, power, equilibrium,
                                                 max_step=0.005, spend=_spend)
    ordinary = model.rollout(initial, power, equilibrium, 10,
                             max_step=0.005, linear_fast=False, spend=_spend)
    torch.testing.assert_close(integrated, ordinary, rtol=0, atol=2e-14)
    epsilon = 1e-6
    for column in range(model.latent_dimension):
        direction = torch.zeros_like(initial)
        direction[:, 6+column] = 1
        plus = model.rollout(initial+epsilon*direction, power, equilibrium, 10,
                             max_step=0.005, linear_fast=False, spend=_spend)
        minus = model.rollout(initial-epsilon*direction, power, equilibrium, 10,
                              max_step=0.005, linear_fast=False, spend=_spend)
        finite_difference = (plus-minus)/(2*epsilon)
        torch.testing.assert_close(response[:, :, :, column], finite_difference,
                                   rtol=2e-7, atol=2e-9)


def test_full_gauss_newton_recovers_hidden_start_and_resets_visible_origin():
    model = _model()
    q, v, power, equilibrium, initial = _problem(model)
    result = predict_from_history(model, q, v, power, equilibrium, torch.ones(6, dtype=torch.float64), 5,
                                  max_step=0.005, spend=_spend)

    torch.testing.assert_close(result["hidden_start"], initial[:, 6:], rtol=1e-7, atol=2e-9)
    torch.testing.assert_close(result["history"][:, 0, :3], q[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(result["history"][:, 0, 3:6], v[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(result["history"][:, :, :6], torch.cat((q, v), dim=2),
                               rtol=2e-7, atol=2e-9)
    torch.testing.assert_close(result["forecast"][:, 0, :3], q[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(result["forecast"][:, 0, 3:6], v[:, -1], rtol=0, atol=0)
    assert result["iterates"].shape == (4, 1, 3)
    assert result["gn_qr_q"].shape[0] == 3


def test_gradients_pass_through_all_gauss_newton_steps_for_port_and_rate():
    candidate = _model()
    target_model = _model(port=(0.36, 0.34, 0.57), tau=(0.11, 0.16, 0.9))
    q, v, power, equilibrium, _ = _problem(candidate)
    q, v = q.detach(), v.detach()
    _, _, _, _, target_initial = _problem(target_model)
    target_path = target_model.rollout(target_initial, power, equilibrium, 5,
                                       max_step=0.005, linear_fast=False)[:, 1:, :6].detach()
    scale = torch.tensor([0.4, 0.5, 0.6, 0.8, 0.9, 1.0], dtype=torch.float64)

    def objective():
        result = predict_from_history(candidate, q, v, power, equilibrium, scale, 5,
                                      max_step=0.005, spend=_spend)
        difference = (result["forecast"][:, 1:, :6]-target_path)/scale
        return difference.square().mean()

    parameters = ((candidate.port_parameter, (0,)), (candidate.rate_parameter, (1,)))
    gradients = torch.autograd.grad(objective(), tuple(parameter for parameter, _ in parameters))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    for (parameter, index), gradient in zip(parameters, gradients, strict=True):
        original = parameter.detach()[index].clone()
        epsilon = 2e-5
        with torch.no_grad():
            parameter[index] = original + epsilon
        plus = float(objective().detach())
        with torch.no_grad():
            parameter[index] = original - epsilon
        minus = float(objective().detach())
        with torch.no_grad():
            parameter[index] = original
        finite_difference = (plus-minus)/(2*epsilon)
        analytic = float(gradient[index])
        assert abs(analytic) > 1e-9
        assert finite_difference == pytest.approx(analytic, rel=3e-3, abs=3e-7)


@pytest.mark.parametrize("case", ["linear", "wrong_history", "step", "negative_count", "scale", "power"])
def test_invalid_history_contract_is_rejected(case):
    model = _model(backbone="linear" if case == "linear" else "sine")
    q, v, power, equilibrium, _ = _problem(model)
    scale = torch.ones(6, dtype=torch.float64)
    count, max_step = 3, 0.005
    if case == "wrong_history":
        q, v = q[:, :-1], v[:, :-1]
    elif case == "step":
        max_step = 0.01
    elif case == "negative_count":
        count = -1
    elif case == "scale":
        scale[0] = 0
    elif case == "power":
        power[0, 0] += 0.1

    with pytest.raises(ValueError):
        predict_from_history(model, q, v, power, equilibrium, scale, count,
                             max_step=max_step, spend=_spend)


def test_rank_deficient_history_is_rejected():
    model = _model(port=(0.3, 0.4, 1e-20))
    q = torch.zeros((1, 11, 3), dtype=torch.float64)
    v = torch.zeros_like(q)
    power = torch.zeros((1, 3), dtype=torch.float64)
    equilibrium = torch.zeros_like(power)

    with pytest.raises(UnresolvedHistoryError, match="rank/conditioning failed"):
        predict_from_history(model, q, v, power, equilibrium, torch.ones(6, dtype=torch.float64), 2,
                             max_step=0.005, spend=_spend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_history_prediction_and_gradients_match_cpu():
    cpu = _model(train_stiffness=True)
    histories = [_problem(cpu, hidden=hidden) for hidden in ((.08, -.05, .03), (.03, .07, -.02))]
    q, v, power, equilibrium, initial = (torch.cat(rows).detach() for rows in zip(*histories, strict=True))
    scale = torch.tensor([0.4, 0.5, 0.6, 0.8, 0.9, 1.0], dtype=torch.float64)
    target = _model(port=(0.36, 0.34, 0.57), tau=(0.11, 0.16, 0.9))
    target_path = target.rollout(initial, power, equilibrium, 4,
                                 max_step=0.005, linear_fast=False)[:, 1:, :6].detach()

    cpu_spend, cuda_spend = [], []
    cpu_result = predict_from_history(cpu, q, v, power, equilibrium, scale, 4,
                                      max_step=0.005,
                                      spend=lambda *item: cpu_spend.append(item))
    gpu = _model(train_stiffness=True).to("cuda")
    gpu_result = predict_from_history(gpu, q, v, power, equilibrium, scale, 4,
                                      max_step=0.005,
                                      spend=lambda *item: cuda_spend.append(item))
    for key in ("hidden_start", "history", "forecast", "iterates", "iteration_residuals",
                "gn_hidden_start"):
        torch.testing.assert_close(gpu_result[key].cpu(), cpu_result[key], rtol=2e-7, atol=2e-9)
    assert all(value.device.type == "cuda" for value in gpu_result.values())
    assert sorted(cpu_spend) == sorted(cuda_spend)
    with pytest.raises(ValueError):
        predict_from_history(gpu, q, v, power, equilibrium, scale, -1,
                             max_step=0.005, spend=_spend)

    def gradients(model, device):
        result = predict_from_history(model, q, v, power, equilibrium, scale, 4,
                                      max_step=0.005, spend=_spend)
        truth = torch.cat((torch.cat((q[:, -1:], v[:, -1:]), dim=-1), target_path), dim=1)
        value = loss(result["forecast"], truth.to(device), [.5, .5], scale.numpy(), [4, 2])
        params = (model.port_parameter, model.rate_parameter, model.log_stiffness_factor)
        return torch.autograd.grad(value, params)

    cpu_grad = gradients(cpu, "cpu")
    gpu_grad = gradients(gpu, "cuda")
    for actual, expected in zip(gpu_grad, cpu_grad, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("max_step", [0.0025, 0.00125])
def test_cuda_public_forecaster_predict_and_checkpoint_round_trip(tmp_path, max_step):
    from kron_reduce.data.history_contract import CausalStateHistory
    from kron_reduce.models.machine_local_forecast import FrozenMachineLocalForecaster

    model = _model()
    q, v, power, equilibrium, _ = _problem(model)
    history = CausalStateHistory(-np.arange(10, -1, -1)*0.01, q[0].detach().numpy(), v[0].detach().numpy(),
                                 power[0].numpy())
    forecaster = FrozenMachineLocalForecaster(model.to("cuda"), np.ones(6), max_step=max_step)
    spent = {}

    def spend(key, amount=1):
        spent[key] = spent.get(key, 0) + amount

    # A non-multiple of ten exercises the final eager frames after CUDA replay.
    gpu = forecaster.predict_batch([history], equilibrium.numpy(), 103, spend=spend)
    assert all(value.device.type == "cuda" for value in gpu.values())
    assert spent["torch_rhs"] == 4*round(.01/max_step)*(3*10+10+103)
    path = tmp_path / "cuda-forecaster.pt"
    forecaster.save_checkpoint(path)
    restored = FrozenMachineLocalForecaster.load_checkpoint(path)
    cpu = restored.predict_batch([history], equilibrium.numpy(), 103)
    for key in ("hidden_start", "history", "forecast", "iterates", "gn_hidden_start"):
        torch.testing.assert_close(gpu[key].cpu(), cpu[key], rtol=2e-7, atol=2e-9)
