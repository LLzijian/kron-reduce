"""Engineering checks for trainable passive spatial relaxation rates."""
import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.machine_local_topology import predict_from_topology_history

DEVICES = [pytest.param("cpu", id="cpu")]
if torch.cuda.is_available():
    DEVICES.append(pytest.param("cuda", id="cuda"))
else:
    DEVICES.append(pytest.param("cuda", id="cuda", marks=pytest.mark.skip(reason="CUDA unavailable")))


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def _model(*, device="cpu", trainable=True, rates=None, scalar_rate=None,
           stiffness=2.4, limit=0.31, train_scales=False):
    params = parameters()
    model = make_model(params, "T0M1", device, train_scales=train_scales).eval()
    if trainable:
        model.set_trainable_spatial_coupling(stiffness=stiffness, limit=limit)
        if rates is not None:
            with torch.no_grad():
                model.spatial_relaxation_rates.copy_(torch.as_tensor(
                    rates, dtype=torch.float64, device=model.mass.device
                ))
    elif scalar_rate is not None:
        model.set_dynamic_coupling(
            scalar_rate/stiffness, stiffness=stiffness, limit=limit, spatial=True
        )
    return model, params


def _initial(model, *, nonzero_xi=False):
    mass = model.mass.detach().cpu().numpy()
    n = len(mass)
    projection = np.eye(n)-np.outer(mass, np.ones(n))/mass.sum()
    q = (0.11*np.sin(.62*np.arange(n)+.15))@projection
    v = (0.075*np.cos(.47*np.arange(n)+.28))@projection
    z = 0.021*np.sin(.33*np.arange(n)+.2)
    power = .035*np.sin(.39*np.arange(n)+.1)
    power -= power.mean()
    equilibrium = np.linspace(-.035, .041, n)
    xi = np.linspace(-.035, .044, n-1) if nonzero_xi else np.zeros(n-1)
    state = model._tensor(np.concatenate((q, v, z, xi))[None, :])
    return (q[None, :], v[None, :], z[None, :], power[None, :], equilibrium,
            state, model._tensor(power[None, :]), model._tensor(equilibrium[None, :]))


def _no_spend(*_args):
    pass


def _true_path(rate, *, device="cpu", count=14, max_step=0.0025):
    model, _params = _model(device=device, trainable=False, scalar_rate=rate)
    q, v, _z, _power, _eq, state, power, equilibrium = _initial(model)
    path = model.rollout(state, power, equilibrium, count, max_step=max_step, spend=_no_spend)
    return model, (q, v, power, equilibrium), path


def _predict(model, inputs, path, count=4):
    _q, _v, power, equilibrium = inputs
    scale = torch.ones(2*model.n_nodes, dtype=torch.float64, device=model.mass.device)
    return predict_from_topology_history(
        model,
        path[:, :11, :model.n_nodes],
        path[:, :11, model.n_nodes:2*model.n_nodes],
        power, equilibrium, scale, count, max_step=0.0025, spend=_no_spend,
    )


def test_equal_machine_rates_match_fixed_scalar_spatial_coupling_through_rollout_and_observer(device):
    rate, stiffness, limit = 0.86, 2.4, 0.31
    trainable, _ = _model(device=device, trainable=True, rates=np.full(10, rate),
                          stiffness=stiffness, limit=limit)
    fixed, _ = _model(device=device, trainable=False, scalar_rate=rate,
                      stiffness=stiffness, limit=limit)
    _q, _v, _z, _power, _eq, initial, power, equilibrium = _initial(trainable)

    trainable_path = trainable.rollout(initial, power, equilibrium, 8, max_step=0.0025, spend=_no_spend)
    fixed_path = fixed.rollout(initial, power, equilibrium, 8, max_step=0.0025, spend=_no_spend)
    torch.testing.assert_close(trainable_path, fixed_path, rtol=3e-11, atol=3e-12)

    observed = fixed.rollout(initial, power, equilibrium, 14, max_step=0.0025, spend=_no_spend)
    inputs = (None, None, power, equilibrium)
    trainable_result = _predict(trainable, inputs, observed)
    fixed_result = _predict(fixed, inputs, observed)
    torch.testing.assert_close(trainable_result["hidden_start"], fixed_result["hidden_start"],
                               rtol=4e-10, atol=4e-12)
    torch.testing.assert_close(trainable_result["history"], fixed_result["history"],
                               rtol=4e-10, atol=4e-12)
    torch.testing.assert_close(trainable_result["forecast"], fixed_result["forecast"],
                               rtol=4e-10, atol=4e-12)


def test_implicit_history_and_future_gradients_match_centered_differences(device):
    true_rate = 0.94
    _truth_model, inputs, truth = _true_path(true_rate, device=device)
    model, _ = _model(device=device, trainable=True, rates=np.linspace(.52, .83, 10))
    rates = model.spatial_relaxation_rates
    result = _predict(model, inputs, truth)
    assert rates.device.type == device and rates.dtype == torch.float64
    assert result["hidden_start"].requires_grad
    assert result["history"].requires_grad and result["forecast"].requires_grad

    probe = torch.linspace(-.7, .8, model.memory_dimension, dtype=torch.float64, device=device)
    history_objective = (result["hidden_start"]*probe).sum()
    target = truth[:, 11:15, :2*model.n_nodes]
    future_residual = result["forecast"][:, 1:5, :2*model.n_nodes]-target
    future_probe = future_residual.detach()
    future_scale = 1e6
    future_objective = future_scale*(future_residual*future_probe).sum()
    history_gradient = torch.autograd.grad(history_objective, rates, retain_graph=True)[0]
    future_gradient = torch.autograd.grad(future_objective, rates)[0]
    assert torch.isfinite(history_gradient).all() and history_gradient.norm() > 1e-11
    assert torch.isfinite(future_gradient).all() and future_gradient.norm() > 1e-14

    direction = torch.linspace(-.8, .9, len(rates), dtype=torch.float64, device=device)
    direction = direction/direction.norm()
    original = rates.detach().clone()
    epsilon = 2e-4

    def values_at(candidate):
        with torch.no_grad():
            rates.copy_(candidate)
            output = _predict(model, inputs, truth)
            hvalue = float((output["hidden_start"]*probe).sum())
            residual = output["forecast"][:, 1:5, :2*model.n_nodes]-target
            fvalue = float(future_scale*(residual*future_probe).sum())
        return hvalue, fvalue

    plus_h, plus_f = values_at(original+epsilon*direction)
    minus_h, minus_f = values_at(original-epsilon*direction)
    with torch.no_grad():
        rates.copy_(original)
    finite_difference_h = (plus_h-minus_h)/(2*epsilon)
    finite_difference_f = (plus_f-minus_f)/(2*epsilon)
    autograd_h = float(history_gradient@direction)
    autograd_f = float(future_gradient@direction)
    np.testing.assert_allclose(autograd_h, finite_difference_h, rtol=3e-3, atol=2e-7)
    np.testing.assert_allclose(autograd_f, finite_difference_f, rtol=3e-3, atol=2e-9)


def test_zero_rates_recover_plain_visible_path_and_keep_a_nonzero_training_gradient(device):
    _truth_model, inputs, truth = _true_path(.82, device=device)
    trainable, _ = _model(device=device, trainable=True)
    plain, _ = _model(device=device, trainable=False)
    rates = trainable.spatial_relaxation_rates
    assert torch.count_nonzero(rates) == 0
    q, v, z, _p_np, _eq_np, _state, power, equilibrium = _initial(trainable)
    base = torch.cat((trainable._tensor(q), trainable._tensor(v), trainable._tensor(z)), dim=1)
    initial_trainable = torch.cat((base, torch.zeros((1, 9), dtype=torch.float64, device=device)), dim=1)
    plain_path = plain.rollout(base, power, equilibrium, 8, max_step=.0025, spend=_no_spend)
    zero_rate_path = trainable.rollout(initial_trainable, power, equilibrium, 8,
                                       max_step=.0025, spend=_no_spend)
    torch.testing.assert_close(zero_rate_path[..., :30], plain_path, rtol=0.0, atol=2e-14)
    torch.testing.assert_close(zero_rate_path[..., 30:], torch.zeros_like(zero_rate_path[..., 30:]),
                               rtol=0.0, atol=0.0)

    result = _predict(trainable, inputs, truth)
    target = truth[:, 11:15, :2*trainable.n_nodes]
    loss = ((result["forecast"][:, 1:5, :2*trainable.n_nodes]-target)**2).mean()
    gradient = torch.autograd.grad(loss, rates)[0]
    # This is the unscaled closed-loop MSE gradient at the zero-rate boundary.
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-16


def test_trainable_spatial_relaxation_obeys_energy_dissipation_identity(device):
    model, _ = _model(device=device, trainable=True, rates=np.linspace(.25, 1.7, 10))
    _q, _v, _z, _power, _eq, state, power, equilibrium = _initial(model, nonzero_xi=True)
    state = state.detach().requires_grad_(True)
    field = model.rhs(state, power, equilibrium)
    storage = model.storage(state, equilibrium).sum()
    storage_gradient = torch.autograd.grad(storage, state)[0]
    energy_rate = (storage_gradient*field).sum(dim=1)

    n = model.n_nodes
    velocity = state[:, n:2*n]
    memory = state[:, 2*n:3*n]
    xi_gradient = storage_gradient[:, 3*n:]
    basis = model.basis[:, :n-1]
    machine_gradient = xi_gradient @ basis.T
    coupling_loss = (model.spatial_relaxation_rates.clamp_min(0.)*machine_gradient.square()).sum()/2.4
    input_work = (velocity*power).sum(dim=1)
    damping_loss = torch.einsum("bi,ij,bj->b", velocity, model.effective_damping, velocity)
    memory_loss = (memory.square()/model.tau).sum(dim=1)
    expected_rate = input_work-damping_loss-memory_loss-coupling_loss
    torch.testing.assert_close(energy_rate, expected_rate, rtol=4e-10, atol=5e-11)
    assert torch.all(input_work.abs() > 1e-6)


def test_zero_rate_gradient_matches_with_trainable_scales_from_same_initial_model(device):
    _truth_model, inputs, truth = _true_path(.82, device=device)
    fixed_scales, _ = _model(device=device, trainable=True, train_scales=False)
    learned_scales, _ = _model(device=device, trainable=True, train_scales=True)
    torch.testing.assert_close(fixed_scales.effective_stiffness, learned_scales.effective_stiffness,
                               rtol=0.0, atol=0.0)
    torch.testing.assert_close(fixed_scales.effective_damping, learned_scales.effective_damping,
                               rtol=0.0, atol=0.0)

    gradients = []
    target = truth[:, 11:15, :2*fixed_scales.n_nodes]
    for model in (fixed_scales, learned_scales):
        assert torch.count_nonzero(model.spatial_relaxation_rates) == 0
        result = _predict(model, inputs, truth)
        loss = ((result["forecast"][:, 1:5, :2*model.n_nodes]-target)**2).mean()
        gradient = torch.autograd.grad(loss, model.spatial_relaxation_rates)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 1e-16
        gradients.append(gradient)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-10, atol=2e-12)


def test_zero_rate_boundary_has_one_sided_derivative_and_paired_shared_gradients(device):
    _, inputs, truth = _true_path(.82, device=device)
    candidate, _ = _model(device=device, train_scales=True)
    control, _ = _model(device=device, trainable=False, train_scales=True)
    named = dict(candidate.named_parameters())
    shared = dict(control.named_parameters())
    rates = candidate.spatial_relaxation_rates
    weights = candidate._tensor(np.linspace(.1, 1., 20))

    def observable(model):
        return (_predict(model, inputs, truth, count=20)["forecast"][:, -1, :20]*weights).sum()

    origin = observable(candidate)
    calibrated = observable(control)
    torch.testing.assert_close(origin, calibrated, rtol=0., atol=1e-14)
    derivatives = torch.autograd.grad(origin, (rates, *(named[key] for key in shared)))
    control_derivatives = torch.autograd.grad(calibrated, tuple(shared.values()))
    for actual, expected in zip(derivatives[1:], control_derivatives, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-11)
    directional = derivatives[0].sum()
    assert directional.abs() > 1e-8
    with torch.no_grad():
        for epsilon in (.001, .0005):
            rates.fill_(epsilon)
            slope = (observable(candidate)-origin)/epsilon
            torch.testing.assert_close(slope, directional, rtol=.003, atol=2e-10)

