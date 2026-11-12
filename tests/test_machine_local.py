import numpy as np
import pytest
import torch
from scipy.linalg import helmert

from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence
from kron_reduce.physics.coordinates import coi_project


def _backbone(kind="sine"):
    mass = np.array([1.5, 2.0, 2.5])
    incidence = complete_incidence(3)
    stiffness = np.array([0.35, 0.28, 0.22])
    edge_damping = np.array([0.08, 0.11, 0.09])
    damping = (incidence * edge_damping) @ incidence.T
    return NonlinearPassiveClosure(
        mass,
        stiffness,
        damping,
        np.zeros((3, 0)),
        np.zeros((0, 0)),
        np.zeros((0, 0)),
        kind,
    )


def _model(*, kind="local", backbone=None, encoder=None):
    backbone = _backbone() if backbone is None else backbone
    n = backbone.n_nodes
    encoder = (
        np.arange(1, n * 8 * n + 1, dtype=float).reshape(n, 8 * n) / 1000
        if encoder is None
        else np.asarray(encoder, dtype=float)
    )
    r = encoder.shape[0]
    tau = np.array([0.19, 0.37, 0.68])[:r]
    scale = np.linspace(0.8, 1.2, 8 * n)
    local_port = np.array([0.24, 0.31, 0.42])
    if kind == "local":
        port = local_port
    else:
        projection = np.eye(n) - np.outer(backbone.mass, np.ones(n)) / backbone.mass.sum()
        local_coupling = (projection * local_port[None, :])[:, :r]
        basis = helmert(n, full=False).T
        port = basis.T @ local_coupling
    return DiagonalMemoryModel(
        backbone,
        port_kind=kind,
        port=port,
        tau=tau,
        encoder=encoder,
        feature_scale=scale,
    )


def _state_and_inputs():
    mass = _backbone().mass
    q = coi_project(np.array([0.025, -0.01, 0.018]), mass)
    v = coi_project(np.array([0.012, -0.017, 0.006]), mass)
    z = np.array([0.014, -0.008, 0.011])
    state = np.r_[q, v, z]
    power = np.array([0.012, -0.004, -0.008])
    equilibrium = coi_project(np.array([0.08, -0.035, 0.02]), mass)
    return state, power, equilibrium


def test_local_and_global_ports_share_physical_coupling_and_rhs():
    local = _model(kind="local")
    global_model = _model(kind="global")
    state, power, equilibrium = _state_and_inputs()

    torch.testing.assert_close(local.coupling, global_model.coupling, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        local.rhs(torch.tensor(state), torch.tensor(power), torch.tensor(equilibrium)),
        global_model.rhs(torch.tensor(state), torch.tensor(power), torch.tensor(equilibrium)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        local.rhs(torch.tensor(state), torch.tensor(power), torch.tensor(equilibrium)).detach(),
        local.closure().rhs(state, power, equilibrium),
        rtol=1e-12,
        atol=1e-12,
    )


def test_torch_rhs_supports_single_and_batch_shapes():
    model = _model()
    state, power, equilibrium = _state_and_inputs()
    states = np.stack((state, 0.7 * state))
    powers = np.stack((power, 0.5 * power))
    equilibria = np.stack((equilibrium, 0.8 * equilibrium))

    single = torch.stack(
        [
            model.rhs(torch.tensor(s), torch.tensor(p), torch.tensor(e))
            for s, p, e in zip(states, powers, equilibria, strict=True)
        ]
    )
    batch = model.rhs(torch.tensor(states), torch.tensor(powers), torch.tensor(equilibria))
    torch.testing.assert_close(batch, single, rtol=1e-13, atol=1e-13)


def test_linear_fast_transition_matches_stepwise_rk4():
    backbone = _backbone(kind="linear")
    model = _model(backbone=backbone)
    state, power, _ = _state_and_inputs()
    initial = torch.tensor(state[None, :])
    powers = torch.tensor(power[None, :])
    equilibrium = torch.zeros((1, 3), dtype=torch.float64)

    fast = model.rollout(initial, powers, equilibrium, 4, max_step=0.005, linear_fast=True)
    rk4 = model.rollout(initial, powers, equilibrium, 4, max_step=0.005, linear_fast=False)
    torch.testing.assert_close(fast, rk4, rtol=2e-12, atol=2e-12)


def test_full_causal_encoder_port_and_rate_gradients_match_differences():
    model = _model()
    mass = model.mass.detach().numpy()
    times = np.arange(11)
    q = np.stack(
        [coi_project(np.array([0.002 * t, -0.001 * t, 0.0004 * t]), mass) for t in times]
    )[None]
    v = np.stack(
        [coi_project(np.array([0.003 * np.sin(t / 4), 0.002 * np.cos(t / 5), -0.001]), mass)
         for t in times]
    )[None]
    power = torch.tensor([[0.006, -0.002, -0.004]], dtype=torch.float64)
    equilibrium = torch.tensor([[0.035, -0.012, 0.007]], dtype=torch.float64)
    features = torch.linspace(-0.02, 0.03, 24, dtype=torch.float64)[None]
    q_t, v_t = torch.tensor(q), torch.tensor(v)
    output_weights = torch.linspace(0.2, 1.0, 27, dtype=torch.float64).reshape(1, 3, 9)

    def objective():
        result = model.causal_rollout(
            q_t, v_t, power, equilibrium, features, 2, max_step=0.01, linear_fast=False
        )
        return (result * output_weights).sum()

    parameters = (model.encoder, model.port_parameter, model.rate_parameter)
    value = objective()
    gradients = torch.autograd.grad(value, parameters)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        direction = torch.linspace(0.3, 1.1, parameter.numel(), dtype=torch.float64).reshape_as(parameter)
        direction = direction / torch.linalg.vector_norm(direction)
        analytic = (gradient * direction).sum()
        original = parameter.detach().clone()
        epsilon = 1e-6
        with torch.no_grad():
            parameter.copy_(original + epsilon * direction)
        plus = objective().detach()
        with torch.no_grad():
            parameter.copy_(original - epsilon * direction)
        minus = objective().detach()
        with torch.no_grad():
            parameter.copy_(original)
        finite_difference = (plus - minus) / (2 * epsilon)
        torch.testing.assert_close(analytic, finite_difference, rtol=2e-3, atol=2e-8)


def test_encoder_regularization_uses_fixed_two_row_reference_denominator():
    rows = np.arange(1, 4 * 24 + 1, dtype=float).reshape(4, 24) / 100
    model_three = _model(encoder=rows[:3])
    model_two = _model(kind="global", encoder=rows[:2])
    denominator = 2 * 24

    assert model_three.regularization().item() == pytest.approx(
        1e-4 * np.square(rows[:3]).sum() / denominator
    )
    assert model_two.regularization().item() == pytest.approx(
        1e-4 * np.square(rows[:2]).sum() / denominator
    )
    assert model_three.regularization().item() - model_two.regularization().item() == pytest.approx(
        1e-4 * np.square(rows[2]).sum() / denominator
    )


def test_invalid_shapes_dtype_and_non_coi_rollout_are_rejected():
    model = _model()
    state, power, equilibrium = _state_and_inputs()
    with pytest.raises(ValueError, match="dimensions differ"):
        model.rhs(torch.tensor(state[:-1]), torch.tensor(power), torch.tensor(equilibrium))
    with pytest.raises(ValueError, match="CPU float64"):
        model.rhs(torch.tensor(state, dtype=torch.float32), torch.tensor(power), torch.tensor(equilibrium))

    initial = torch.tensor(state[None, :])
    powers = torch.tensor(power[None, :])
    equilibria = torch.tensor(equilibrium[None, :])
    non_coi = initial.clone()
    non_coi[:, 0] += 0.1
    with pytest.raises(ValueError, match="balanced power and COI visible state"):
        model.rollout(non_coi, powers, equilibria, 1)
    with pytest.raises(ValueError, match="balanced power and COI visible state"):
        model.rollout(initial, powers + 0.1, equilibria, 1)
