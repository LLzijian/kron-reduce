import numpy as np
import pytest
import torch

from kron_reduce.evaluation.kiki_machine_local_topology import scales_from_train, selection_rank
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model, predict
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


def backbone():
    n = 10
    mass = np.ones(n)
    projection = np.eye(n) - np.ones((n, n))/n
    return NonlinearPassiveClosure(mass, np.ones(45), projection, projection,
                                   np.zeros((n, n)), np.eye(n))


@pytest.mark.parametrize("support", ["T0M1", "T1M1"])
def test_task3_predict_shape_and_memory_only_diagnostics(support):
    model = make_model(backbone(), support)
    event = {"x": np.zeros((11, 20)), "p0": np.zeros(10), "equilibrium": np.zeros(10)}
    result = predict(model, [event], np.ones(20), .005, lambda *_: None, count=1,
                     prior_weight=1e-6)
    assert result["forecast"].shape == (1, 2, 20)
    assert result["history"].shape == (1, 11, 30)
    assert result["hidden_start"].shape == (1, 10)
    assert result["gn_singular"].shape == (1, 1, model.latent_dimension)
    assert all(torch.isfinite(tensor).all() for tensor in result.values())


def test_snapshot_rejects_changed_backbone_and_freezes_weights(tmp_path):
    model = make_model(backbone(), "T1M1", train_scales=True)
    state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    path = tmp_path / "weights.npz"
    np.savez_compressed(path, **state)
    loaded = load_snapshot(make_model(backbone(), "T1M1", train_scales=True), path)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(ValueError, match="snapshot tensor"):
        load_snapshot(make_model(backbone(), "T0M1", train_scales=True), path)
    altered = dict(state)
    altered["stiffness"] = state["stiffness"] + 1
    np.savez_compressed(path, **altered)
    with pytest.raises(ValueError, match="backbone"):
        load_snapshot(make_model(backbone(), "T1M1", train_scales=True), path)


def test_selection_rank_requires_coverage_before_error_count():
    def result(*, resolved, tp, tn, score):
        return {"stability": {"allocated": 107, "resolved": resolved, "tp": tp, "tn": tn},
                "task3_six_term_score": score}

    covered = selection_rank(result(resolved=107, tp=45, tn=50, score=.2), .3, 8)
    fewer_errors_but_missing = selection_rank(result(resolved=106, tp=45, tn=55, score=.2), .3, 4)
    assert covered < fewer_errors_but_missing
    assert selection_rank(result(resolved=107, tp=50, tn=50, score=.4), .3, 2)[0] == 1
    assert selection_rank(result(resolved=107, tp=50, tn=50, score=float("nan")), .3, 2)[0] == 2


def test_train_scale_memory_radius_uses_mass_weighted_history_rms():
    a, b = np.zeros((311, 20)), np.zeros((311, 20))
    a[:11, 10:13] = [2., 0., 3.]
    b[:11, 10:13] = [4., 0., 1.]
    mass = np.arange(1., 11.)
    scale = scales_from_train([{"x": a}, {"x": b}], mass)
    mean_v_squared = np.r_[10., 1e-10, 5., np.full(7, 1e-10)]
    assert scale["memory_radius"] == pytest.approx(3*np.sqrt(mass @ mean_v_squared))
    assert scale["history"][11] == pytest.approx(1e-5)


def test_physical_units_forecast_with_prior_keeps_parameter_gradient_and_reload(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(backbone(), "T1M1", train_scales=True, device=device)
    q = np.linspace(-.002, .002, 10)
    v = np.linspace(.001, -.001, 10)
    power = np.linspace(-.05, .05, 10)
    equilibrium = np.linspace(-.2, .2, 10)
    hidden = np.r_[.01*np.linspace(-1., 1., 10), .01*np.linspace(-1., 1., 9)]
    start = model._tensor(np.r_[q, v, hidden])[None]
    with torch.no_grad():
        history = model.rollout(start, model._tensor(power)[None], model._tensor(equilibrium)[None],
                                10, max_step=.005, spend=lambda *_: None)
    event = {"x": history[0, :, :20].cpu().numpy(), "p0": power, "equilibrium": equilibrium}
    result = predict(model, [event], np.ones(20), .005, lambda *_: None, count=1,
                     prior_weight=1e-6)
    assert torch.isfinite(result["forecast"]).all()
    params = tuple(model.parameters())
    objective = lambda: predict(model, [event], np.ones(20), .005, lambda *_: None,
                                count=1, prior_weight=1e-6)["forecast"][0, -1, 10].square()
    gradients = torch.autograd.grad(objective(), params)
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in gradients)
    directions = [torch.ones_like(parameter)/parameter.numel() for parameter in params]
    epsilon = 1e-5
    with torch.no_grad():
        for parameter, direction in zip(params, directions, strict=True):
            parameter.add_(epsilon*direction)
    plus = float(objective().detach())
    with torch.no_grad():
        for parameter, direction in zip(params, directions, strict=True):
            parameter.add_(-2*epsilon*direction)
    minus = float(objective().detach())
    with torch.no_grad():
        for parameter, direction in zip(params, directions, strict=True):
            parameter.add_(epsilon*direction)
    analytic = sum(float((gradient*direction).sum()) for gradient, direction in zip(gradients, directions, strict=True))
    assert analytic == pytest.approx((plus-minus)/(2*epsilon), rel=2e-3, abs=1e-11)

    path = tmp_path / "frozen.npz"
    np.savez_compressed(path, **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    reloaded = load_snapshot(make_model(backbone(), "T1M1", train_scales=True, device=device), path)
    with torch.no_grad():
        frozen_reference = predict(model, [event], np.ones(20), .005, lambda *_: None,
                                   count=1, prior_weight=1e-6)
        reproduced = predict(reloaded, [event], np.ones(20), .005, lambda *_: None,
                             count=1, prior_weight=1e-6)
    assert all(torch.equal(value, reloaded.state_dict()[key]) for key, value in model.state_dict().items())
    torch.testing.assert_close(frozen_reference["forecast"], reproduced["forecast"], rtol=0, atol=1e-14)
