from collections import Counter
from itertools import combinations
from math import log

import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.unified_discovery_m3 import MODULES, UnifiedDiscoveryModel, predict_m3
from kron_reduce.physics.coordinates import coi_basis

STRENGTHS = {"G1": {"a": .08}, "G2": {"rho": .58}, "G3": {"mu_c": .23},
             "G4": {"mu_xi": .33, "kappa_xi": .10}, "G5": {"gamma_s": .45}}
SUPPORTS = tuple(group for size in range(len(MODULES)+1) for group in combinations(MODULES, size))


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _unmetered(*_):
    pass


def _model(support=MODULES, *, device="cpu", zero=False):
    mass, _, _, port, tau, _ = parameters()
    n = len(mass)
    modes = coi_basis(np.ones(n))
    damping = (modes*np.linspace(.35, 1.15, n-1)) @ modes.T
    stiffness = np.linspace(.09, .17, n*(n-1)//2)
    empty = np.empty((0, 0))
    backbone = NonlinearPassiveClosure(mass, stiffness, damping, np.empty((n, 0)), empty, empty)
    values = {key: 0. if zero and key != "kappa_xi" else value for group in support
              for key, value in STRENGTHS[group].items()}
    return UnifiedDiscoveryModel(backbone, port=port, tau=tau, support=support, module_values=values).to(device)


def _cases(model, amplitudes):
    generator = torch.Generator().manual_seed(8265)
    n = model.n_nodes
    samples = len(amplitudes)
    state = model._tensor(torch.randn((samples, 2*n+model.latent_dimension), generator=generator, dtype=torch.float64))
    state *= model.mass.new_tensor(amplitudes)[:, None]
    state[:, n:2*n] *= .45
    state[:, 2*n:] *= .3
    for visible in (state[:, :n], state[:, n:2*n]):
        visible -= ((visible@model.mass)/model.mass.sum())[:, None]
    power = .14*model._tensor(torch.randn((samples, n), generator=generator, dtype=torch.float64))
    power = power @ model.power_projection.T
    equilibrium = .09*model._tensor(torch.randn((samples, n), generator=generator, dtype=torch.float64))
    return state, power, equilibrium


@pytest.mark.parametrize("support", SUPPORTS, ids=["+".join(group) or "backbone" for group in SUPPORTS])
def test_all_supports_full_power_balance_and_state_tangent(support):
    model = _model(support)
    state, power, equilibrium = _cases(model, (.07, .35, .9, 1.6))
    n, width = model.n_nodes, state.shape[-1]
    assert model.latent_dimension == 10+int("G3" in support)+9*int("G4" in support)
    with torch.enable_grad():
        state.requires_grad_(True)
        gradient, = torch.autograd.grad(model.storage(state, equilibrium).sum(), state)
    state = state.detach()
    with torch.no_grad():
        eye = torch.eye(width, dtype=state.dtype).expand(len(state), -1, -1)
        field, jacobian = model._rhs(power, equilibrium, tangent=True)(state, eye)
        torch.testing.assert_close(field, model.rhs(state, power, equilibrium), atol=1e-13, rtol=1e-13)
        batch_rhs = model._rhs(power.repeat_interleave(width, dim=0), equilibrium.repeat_interleave(width, dim=0))
        perturbation = 8e-7*eye
        plus = batch_rhs((state[:, None]+perturbation).reshape(-1, width))
        minus = batch_rhs((state[:, None]-perturbation).reshape(-1, width))
        finite_difference = ((plus-minus)/1.6e-6).reshape(len(state), width, width).transpose(1, 2)
        torch.testing.assert_close(jacobian, finite_difference, atol=2e-8, rtol=2e-7)

        q, v, z = (state[:, part*n:(part+1)*n] for part in range(3))
        root = model._damping_root()
        strength = model.rho if "G2" in support else 0.
        local_s = (q[..., :, None]-q[..., None, :]).square().sum(dim=-1)/(n-1)
        diagonal = 1.-strength*local_s/(1.+local_s)
        damping = (root*diagonal[:, None, :]) @ root
        torch.testing.assert_close(damping, model.damping_matrix(q).expand_as(damping), atol=2e-13, rtol=2e-13)
        assert torch.linalg.eigvalsh(damping).min() >= -1e-12
        damping_loss = ((v@root).square()*diagonal).sum(dim=-1)
        memory_loss = (z.square()/model.tau).sum(dim=-1)
        common_loss = model.mu_c*gradient[:, 3*n].square() if "G3" in support else 0.
        spatial_loss = model.mu_xi*gradient[:, -9:].square().sum(dim=-1) if "G4" in support else 0.
        dissipated = damping_loss+memory_loss+common_loss+spatial_loss
        torch.testing.assert_close(model.dissipation(state, equilibrium), dissipated, atol=2e-13, rtol=2e-13)
        supplied = (v*power).sum(dim=-1)
        terms = gradient*field
        residual = (terms.sum(dim=-1)-(supplied-dissipated)).abs()/terms.abs().sum(dim=-1).clamp_min(1.)
        assert residual.max() < 1e-12
        assert torch.all(dissipated > 0.)

        extreme = state.clone()
        extreme[:, 3*n:] *= 1000.
        multiplier = model.edge_stiffness(extreme)/model.effective_stiffness
        assert torch.all((multiplier >= .25-1e-12) & (multiplier <= 2.+1e-12))
        physics = model.physical_parameters()
        assert len(physics["edge_stiffness"]) == 45 and min(physics["edge_stiffness"]) > 0.
        assert len(physics["damping_coefficients"]) == 9 and min(physics["damping_coefficients"]) > 0.
        assert len(physics["port_gain"]) == len(physics["tau"]) == 10
        assert min(physics["port_gain"]) >= .01 and .02 <= min(physics["tau"]) <= max(physics["tau"]) <= 5.


@torch.no_grad()
def test_optimizer_projection_enforces_physical_bounds():
    model = _model()
    reference = model.physical_parameters()
    model.edge_log_factor[::2], model.edge_log_factor[1::2] = 3., -3.
    model.damping_log_factor[::2], model.damping_log_factor[1::2] = -3., 3.
    model.port_parameter.fill_(log(.001))
    model.rate_parameter[::2], model.rate_parameter[1::2] = -80., 80.
    for name, value in {"a": 9., "rho": 9., "mu_c": -1., "mu_xi": 9., "kappa_xi": -1., "gamma_s": 9.}.items():
        getattr(model, name).fill_(value)
    model.project_parameters_()
    after = model.physical_parameters()
    for key, limits in (("edge_stiffness", (2., .5)), ("damping_coefficients", (.5, 2.))):
        ratios = np.asarray(after[key])/reference[key]
        np.testing.assert_allclose(ratios[::2], limits[0], rtol=1e-13, atol=0.)
        np.testing.assert_allclose(ratios[1::2], limits[1], rtol=1e-13, atol=0.)
    np.testing.assert_allclose(after["port_gain"], .01, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(after["tau"][::2], .02, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(after["tau"][1::2], 5., rtol=1e-13, atol=0.)
    actual = {name: value for values in after["modules"].values() for name, value in values.items()}
    assert actual == pytest.approx({"a": .6, "rho": .85, "mu_c": 0., "mu_xi": .5, "kappa_xi": .05, "gamma_s": 8.})


@torch.no_grad()
def test_zero_strength_blocks_reduce_exactly_to_backbone():
    reference = _model(MODULES).with_support(())
    state, power, equilibrium = _cases(reference, (.35, .9, 1.6))
    for support in (*((name,) for name in MODULES), MODULES):
        candidate = _model(support, zero=True)
        dummy = state.new_zeros((len(state), candidate.latent_dimension-reference.latent_dimension))
        padded = torch.cat((state, dummy), dim=-1)
        expected = reference.rhs(state, power, equilibrium)
        torch.testing.assert_close(candidate.rhs(padded, power, equilibrium), torch.cat((expected, dummy), dim=-1),
                                   atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(candidate.storage(padded, equilibrium), reference.storage(state, equilibrium),
                                   atol=1e-13, rtol=1e-13)


def _observations(model):
    n = model.n_nodes
    with torch.no_grad():
        state, power, equilibrium = _cases(model, (.37, .83))
        history = model.rollout(state, power, equilibrium, 10, max_step=.005, spend=_unmetered)
    return history[:, :, :n], history[:, :, n:2*n], power, equilibrium, model.mass.new_full((2*n,), .08)


def _shared_parameters(model):
    params = tuple(p for p in model.parameters() if p.numel() > 1 and p.requires_grad)
    assert sorted(p.numel() for p in params) == [9, 10, 10, 45]
    return params


def _settings(spend=_unmetered, step=.005):
    return {"max_step": step, "spend": spend, "memory_radius": .45, "prior_weight": 1e-5}


def _observable(result):
    values = torch.cat((result["hidden_start"], result["forecast"][:, -1, :20]), dim=-1)
    rng = torch.Generator().manual_seed(13)
    weights = torch.randn(values.shape, generator=rng, dtype=values.dtype).to(values.device)
    return (values*weights).sum()


@torch.no_grad()
def _difference(parameter, index, objective):
    flat = parameter.view(-1)
    initial = flat[index].clone()
    epsilon = 2e-5
    flat[index] = initial+epsilon
    try:
        plus = objective()
        flat[index] = initial-epsilon
        minus = objective()
        return (plus-minus)/(2.*epsilon)
    finally:
        flat[index] = initial


def test_all_74_shared_coordinates_differentiate_through_soft_gn_and_forecast():
    model = _model()
    inputs = _observations(model)
    params = _shared_parameters(model)
    count = 2

    def objective():
        return _observable(predict_m3(model, *inputs, count, **_settings()))

    ledger = Counter()

    def spend(operation, amount=1):
        ledger[operation] += amount

    result = predict_m3(model, *inputs, count, **_settings(spend))
    assert result["hidden_start"].shape == result["singular"].shape == (2, 10)
    assert torch.all(result["hidden_start"].norm(dim=-1) < .90)
    assert torch.count_nonzero(result["history"][:, 0, 30:]) == 0
    origin = torch.cat((inputs[0][:, -1], inputs[1][:, -1], result["history"][:, -1, 20:]), dim=-1)
    torch.testing.assert_close(result["forecast"][:, 0], origin, atol=0., rtol=0.)
    assert ledger == {"rollouts": 7, "torch_rhs": 496, "tangent_rhs": 400, "qr": 10, "svd": 10,
                      "matrix_products": 10, "triangular_solves": 10}
    analytic = torch.cat([gradient.flatten() for gradient in torch.autograd.grad(_observable(result), params)])
    numeric = torch.stack([_difference(p, i, objective) for p in params for i in range(p.numel())])
    assert analytic.numel() == numeric.numel() == 74
    assert torch.all(numeric.abs() > 1e-9)
    torch.testing.assert_close(analytic, numeric, atol=4e-7, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture and gradients")
def test_cuda_causal_two_step_capture_and_gradient(monkeypatch):
    model = _model(device="cuda")
    inputs = _observations(model)
    capture, size = model._cuda_rollout, []

    def track(initial, count, frame, spend, calls_per_frame):
        size.append(count)
        return capture(initial, count, frame, spend, calls_per_frame)

    monkeypatch.setattr(model, "_cuda_rollout", track)
    with torch.no_grad():
        dual = []
        for step in (.0025, .00125):
            settings = _settings(step=step)
            captured = predict_m3(model, *inputs, 100, **settings)["forecast"]
            first = predict_m3(model, *inputs, 50, **settings)["forecast"]
            _, _, power, equilibrium, _ = inputs
            second = model.rollout(first[:, -1], power, equilibrium, 50, max_step=step, spend=_unmetered)
            torch.testing.assert_close(captured, torch.cat((first, second[:, 1:]), dim=1), atol=1e-11, rtol=1e-10)
            dual.append(captured)
        assert size == [100, 100]
        torch.testing.assert_close(*dual, atol=4e-7, rtol=3e-6)

    def objective():
        return _observable(predict_m3(model, *inputs, 2, **_settings()))

    params = _shared_parameters(model)
    gradients = torch.autograd.grad(objective(), params)
    assert all(torch.isfinite(g).all() for g in gradients)
    chosen = int(gradients[0].abs().argmax())
    assert gradients[0][chosen].abs() > 1e-7
    finite_difference = _difference(params[0], chosen, objective)
    torch.testing.assert_close(gradients[0][chosen], finite_difference, atol=4e-7, rtol=4e-4)
