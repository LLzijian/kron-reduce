import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.first_swing_fit import (
    enable_first_swing_rho,
    load_first_swing_scoring_view,
    predict_first_swing_fit,
)
from kron_reduce.models.first_swing_gn_grid import predict_gn_grid
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _unmetered(*_):
    pass


def _model():
    return make_model(parameters(), "T0M1", "cpu", train_scales=True)


@pytest.fixture
@torch.no_grad()
def problem():
    reference = _model().requires_grad_(False)
    reference.set_angle_damping(2./3., spatial_mix=1.)
    n = reference.n_nodes
    state = reference.mass.new_zeros((2, 3*n))
    state[:, :3] = state.new_tensor([[1.2, -.7, .42], [-.8, .4, .1]])
    state[:, n:n+3] = state.new_tensor([[.65, -.3, -.15], [-.4, .15, .2]])
    state[:, 2*n:2*n+2] = state.new_tensor([[.13, -.1], [.45, -.2]])
    for visible in (state[:, :n], state[:, n:2*n]):
        visible -= ((visible@reference.mass)/reference.mass.sum())[:, None]
    power = state.new_zeros((len(state), n))
    power[:, :2] = state.new_tensor([[.07, -.07], [-.04, .04]])
    equilibrium = torch.zeros_like(power)
    scale = state.new_full((2*n,), .03)
    settings = {"max_step": .0025, "spend": _unmetered, "memory_radius": .25, "prior_weight": 1e-6}
    history = reference.rollout(state, power, equilibrium, 10, max_step=settings["max_step"], spend=_unmetered)
    return reference, (history[:, :, :n], history[:, :, n:2*n], power, equilibrium, scale), settings


@torch.no_grad()
def test_v4_start_reproduces_frozen_hard_and_fixed_prior_soft_forecasts(problem):
    reference, inputs, settings = problem
    model = enable_first_swing_rho(_model())
    assert model.angle_damping_rho.item() == pytest.approx(2./3.)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 23
    assert "parametrizations.angle_damping_rho.original" in dict(model.named_parameters())

    hard = predict_first_swing_fit(model, *inputs, 3, **settings, soft_radius=False)
    old_hard = predict_from_topology_history(reference, *inputs, 3, **settings)
    soft = predict_first_swing_fit(model, *inputs, 3, **settings, soft_radius=True)
    old_soft = predict_gn_grid(reference, *inputs, **settings, count=3, primary_r0=None,
                               modes=[{"name": "fixed_soft", "prior": "fixed", "radius_multiplier": 2., "reference": .1}])
    for key in ("forecast", "history", "hidden_start", "singular"):
        torch.testing.assert_close(hard[key], old_hard[key], atol=1e-11, rtol=1e-10)
    for key in ("history", "hidden_start", "singular"):
        torch.testing.assert_close(soft[key], old_soft[key][0], atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(soft["forecast"][..., :2*model.n_nodes], old_soft["forecast"][0], atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("soft_radius", [False, True])
def test_raw_rho_gradient_covers_memory_fit_and_future(problem, soft_radius):
    _, inputs, settings = problem
    model = enable_first_swing_rho(_model())
    n = model.n_nodes
    raw = model.parametrizations.angle_damping_rho.original

    def observables():
        result = predict_first_swing_fit(model, *inputs, 3, **settings, soft_radius=soft_radius)
        hidden = result["hidden_start"].square().mean()
        future_speed = result["forecast"][:, 1:, n:2*n].square().mean()
        return hidden, future_speed

    hidden, future_speed = observables()
    analytic = torch.stack((torch.autograd.grad(hidden, raw, retain_graph=True)[0],
                            torch.autograd.grad(future_speed, raw)[0]))
    epsilon = 2e-5
    with torch.no_grad():
        raw.add_(epsilon)
        plus = torch.stack(observables())
        raw.sub_(2.*epsilon)
        minus = torch.stack(observables())
        raw.add_(epsilon)
    assert torch.all(analytic.abs() > 1e-6)
    torch.testing.assert_close(analytic, (plus-minus)/(2.*epsilon), atol=1e-8, rtol=2e-5)


def test_trained_rho_state_reloads_and_dissipates_with_nonzero_input(problem):
    _, inputs, settings = problem
    model = enable_first_swing_rho(_model())
    restored = enable_first_swing_rho(_model())
    n = model.n_nodes
    with torch.no_grad():
        model.parametrizations.angle_damping_rho.original.add_(1.)
        assert 2./3. < model.angle_damping_rho.item() < .95
        checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
        restored.load_state_dict(checkpoint, strict=True)
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in checkpoint.items())
        expected = predict_first_swing_fit(model, *inputs, 1, **settings, soft_radius=True)["forecast"]
        predicted = predict_first_swing_fit(restored, *inputs, 1, **settings, soft_radius=True)["forecast"]
    torch.testing.assert_close(predicted, expected, atol=0., rtol=0.)

    state = predicted[:, -1].requires_grad_(True)
    _, _, power, equilibrium, _ = inputs
    gradient, = torch.autograd.grad(restored.storage(state, equilibrium).sum(), state)
    storage_rate = (gradient*restored.rhs(state, power, equilibrium)).sum(dim=1)
    supplied = (state[:, n:2*n]*power).sum(dim=1)
    assert torch.all(storage_rate < supplied)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
@torch.no_grad()
def test_frozen_scoring_view_matches_fitted_eager_forecast_in_cuda_graph(problem, monkeypatch):
    _, inputs, settings = problem
    inputs = tuple(value.cuda() for value in inputs)
    fitted = enable_first_swing_rho(_model().cuda())
    fitted.parametrizations.angle_damping_rho.original.add_(1.)
    fitted.log_stiffness_factor.add_(.04)
    view = load_first_swing_scoring_view(fitted, _model().cuda())
    assert isinstance(view.angle_damping_rho, float)
    assert 2./3. < view.angle_damping_rho == float(fitted.angle_damping_rho) < .95
    assert not view.training and all(not parameter.requires_grad for parameter in view.parameters())

    first = predict_first_swing_fit(fitted, *inputs, 50, **settings, soft_radius=True)["forecast"]
    _, _, power, equilibrium, _ = inputs
    second = fitted.rollout(first[:, -1], power, equilibrium, 50, max_step=settings["max_step"], spend=_unmetered)
    expected = torch.cat((first, second[:, 1:]), dim=1)

    capture, calls = view._cuda_rollout, []

    def track(initial, count, frame, spend, calls_per_frame):
        calls.append(count)
        return capture(initial, count, frame, spend, calls_per_frame)

    monkeypatch.setattr(view, "_cuda_rollout", track)
    actual = predict_first_swing_fit(view, *inputs, 100, **settings, soft_radius=True)["forecast"]
    assert calls == [100]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
