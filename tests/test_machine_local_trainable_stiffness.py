import numpy as np
import torch

from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.machine_local_forecast import FrozenMachineLocalForecaster
from kron_reduce.models.nonlinear_history import predict_from_history, tangent_rhs
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence
from kron_reduce.physics.coordinates import coi_project, coi_project_power


def _model(*, train_stiffness=False):
    n = 3
    mass = np.array([1.2, 1.6, 2.1])
    incidence = complete_incidence(n)
    stiffness = np.array([0.31, 0.24, 0.19])
    damping = (incidence * np.array([0.07, 0.09, 0.08])) @ incidence.T
    backbone = NonlinearPassiveClosure(
        mass, stiffness, damping, np.zeros((n, 0)), np.zeros((0, 0)),
        np.zeros((0, 0)), "sine",
    )
    return DiagonalMemoryModel(
        backbone, port_kind="local", port=np.array([0.22, 0.29, 0.34]),
        tau=np.array([0.21, 0.38, 0.64]),
        encoder=np.linspace(-0.01, 0.015, n * 8 * n).reshape(n, 8 * n),
        feature_scale=np.linspace(0.8, 1.2, 8 * n), train_stiffness=train_stiffness,
    )


def _history_inputs(model):
    n = model.n_nodes
    mass = model.mass.detach().numpy()
    q = np.stack([coi_project(np.array([.012*np.sin(t/3), -.009*np.cos(t/4), .006*t/10]), mass)
                  for t in range(11)])[None]
    v = np.stack([coi_project(np.array([.008*np.cos(t/5), .006*np.sin(t/4), -.004]), mass)
                  for t in range(11)])[None]
    power = coi_project_power(np.array([.012, -.005, -.007]), mass)[None]
    equilibrium = coi_project(np.array([.045, -.021, .013]), mass)[None]
    features = torch.linspace(-.03, .04, 8*n, dtype=torch.float64)[None]
    return (torch.tensor(q), torch.tensor(v), torch.tensor(power),
            torch.tensor(equilibrium), features)


def test_zero_factor_matches_frozen_model_and_numpy_physics(tmp_path):
    baseline, trainable = _model(), _model(train_stiffness=True)
    assert "log_stiffness_factor" not in baseline.state_dict()
    assert torch.equal(trainable.log_stiffness_factor.detach(), torch.zeros(()))
    torch.testing.assert_close(trainable.effective_stiffness, trainable.stiffness, rtol=0, atol=0)

    n, r = trainable.n_nodes, trainable.latent_dimension
    mass = trainable.mass.detach().numpy()
    q = coi_project(np.array([.035, -.018, .011]), mass)
    v = coi_project(np.array([.014, -.009, .004]), mass)
    z = np.array([.02, -.013, .008])
    equilibrium = coi_project(np.array([.06, -.025, .012]), mass)
    power = coi_project_power(np.array([.013, -.006, -.007]), mass)
    state = np.r_[q, v, z]
    expected = trainable.closure()
    torch.testing.assert_close(trainable.rhs(torch.tensor(state), torch.tensor(power),
                                             torch.tensor(equilibrium)),
                               torch.tensor(expected.rhs(state, power, equilibrium)),
                               rtol=2e-13, atol=2e-13)
    assert abs(expected.energy_identity(state, power, equilibrium)["identity_error"]) < 1e-12
    tangent = torch.linspace(-.3, .4, (2*n+r), dtype=torch.float64)[None, :, None]
    _, derivative = tangent_rhs(trainable, torch.tensor(state[None]), tangent,
                                torch.tensor(power[None]), torch.tensor(equilibrium[None]),
                                spend=lambda *_: None)
    epsilon = 1e-6
    plus = trainable.rhs(torch.tensor(state[None])+epsilon*tangent[:, :, 0],
                         torch.tensor(power[None]), torch.tensor(equilibrium[None]))
    minus = trainable.rhs(torch.tensor(state[None])-epsilon*tangent[:, :, 0],
                          torch.tensor(power[None]), torch.tensor(equilibrium[None]))
    torch.testing.assert_close(derivative[:, :, 0], (plus-minus)/(2*epsilon), rtol=2e-7, atol=2e-9)

    initial = torch.tensor(np.r_[q, v, np.zeros(r)][None])
    p, e = torch.tensor(power[None]), torch.tensor(equilibrium[None])
    torch.testing.assert_close(trainable.rollout(initial, p, e, 3, linear_fast=False),
                               baseline.rollout(initial, p, e, 3, linear_fast=False),
                               rtol=0, atol=0)

    with torch.no_grad():
        trainable.log_stiffness_factor.fill_(np.log(1.37))
    frozen = trainable.freeze_effective_stiffness()
    expected_k = trainable.stiffness.detach().numpy()*1.37
    np.testing.assert_allclose(frozen.effective_stiffness.detach().numpy(), expected_k,
                               rtol=2e-15, atol=2e-15)
    assert "log_stiffness_factor" not in frozen.state_dict()
    assert frozen.closure().energy_identity(state, power, equilibrium)["identity_error"] < 1e-12
    wrapper = FrozenMachineLocalForecaster(frozen, np.ones(2*n), max_step=.0025)
    checkpoint = tmp_path / "frozen.pt"
    wrapper.save_checkpoint(checkpoint)
    restored = FrozenMachineLocalForecaster.load_checkpoint(checkpoint)
    torch.testing.assert_close(restored.model.effective_stiffness, frozen.effective_stiffness,
                               rtol=0, atol=0)


def test_three_step_gauss_newton_reset_future_loss_gradients_match_differences():
    model = _model(train_stiffness=True)
    q, v, power, equilibrium, _features = _history_inputs(model)
    scale = torch.tensor([.03, .03, .03, .02, .02, .02], dtype=torch.float64)
    weights = torch.linspace(.2, 1., 4*(2*model.n_nodes+model.latent_dimension),
                             dtype=torch.float64).reshape(1, 4, -1)

    def objective():
        result = predict_from_history(model, q, v, power, equilibrium, scale, 3,
                                      max_step=.005, spend=lambda *_: None)
        return (result["forecast"]*weights).sum()

    parameters = (model.log_stiffness_factor, model.port_parameter, model.rate_parameter)
    value = objective()
    gradients = torch.autograd.grad(value, parameters)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        direction = torch.ones_like(parameter)
        direction /= torch.linalg.vector_norm(direction)
        original = parameter.detach().clone()
        epsilon = 1e-6
        with torch.no_grad():
            parameter.copy_(original+epsilon*direction)
        plus = objective().detach()
        with torch.no_grad():
            parameter.copy_(original-epsilon*direction)
        minus = objective().detach()
        with torch.no_grad():
            parameter.copy_(original)
        analytic = (gradient*direction).sum()
        finite_difference = (plus-minus)/(2*epsilon)
        torch.testing.assert_close(analytic, finite_difference, rtol=3e-3, atol=2e-8)
