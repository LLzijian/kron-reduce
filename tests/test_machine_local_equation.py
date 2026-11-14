import numpy as np
import pytest
import torch

from kron_reduce.models.machine_local_equation import JointMachineLocalModel
from kron_reduce.models.nonlinear_history import predict_from_history
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence


def _backbone():
    n = 10
    incidence = complete_incidence(n)
    stiffness = np.linspace(0.2, 0.5, incidence.shape[1])
    edge_damping = np.full(incidence.shape[1], 0.03)
    damping = (incidence * edge_damping) @ incidence.T
    return NonlinearPassiveClosure(
        np.linspace(1.0, 2.0, n), stiffness, damping,
        np.zeros((n, 0)), np.zeros((0, 0)), np.zeros((0, 0)), "sine",
    )


def _model(*, harmonic=False, modulated_damping=False):
    return JointMachineLocalModel(
        _backbone(), port=np.linspace(0.2, 0.4, 10), tau=np.linspace(0.1, 0.5, 10),
        harmonic=harmonic, modulated_damping=modulated_damping,
    )


def _project(x, mass):
    return x - (x @ mass)[:, None] / mass.sum()


def test_base_matches_frozen_backbone_and_common_start():
    model = _model()
    q = _project(torch.linspace(-0.04, 0.05, 10, dtype=torch.float64)[None], model.mass)
    v = _project(torch.linspace(0.03, -0.02, 10, dtype=torch.float64)[None], model.mass)
    z = torch.linspace(-0.02, 0.02, 10, dtype=torch.float64)[None]
    state = torch.cat((q, v, z), dim=1).requires_grad_()
    power = torch.zeros_like(q)
    equilibrium = _project(torch.linspace(-0.03, 0.04, 10, dtype=torch.float64)[None], model.mass)
    base = _model()
    torch.testing.assert_close(model.rhs(state, power, equilibrium), base.rhs(state, power, equilibrium), rtol=0, atol=0)
    path = model.rollout(state, power, equilibrium, 3, max_step=0.0025, linear_fast=False)
    torch.testing.assert_close(path[:, 0, :20], state[:, :20], rtol=0, atol=0)


def test_potential_force_hessian_and_damping_are_passive():
    model = _model(harmonic=True, modulated_damping=True)
    with torch.no_grad():
        model.harmonic_strength.fill_(0.6)
        model.modulated_damping_strength.fill_(1.0)
    q = _project(torch.linspace(-0.11, 0.13, 10, dtype=torch.float64)[None], model.mass).requires_grad_()
    equilibrium = _project(torch.linspace(0.08, -0.07, 10, dtype=torch.float64)[None], model.mass)
    potential = model.potential(q, equilibrium)
    force = torch.autograd.grad(potential.sum(), q, create_graph=True)[0]
    torch.testing.assert_close(force.sum(dim=1), torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0)
    hessian = torch.autograd.functional.hessian(lambda x: model.potential(x[None], equilibrium).sum(), q[0])
    torch.testing.assert_close(hessian, hessian.T, rtol=1e-11, atol=1e-11)
    assert torch.linalg.eigvalsh(hessian).min() >= -1e-10
    damping = model.damping_matrix(q)
    torch.testing.assert_close(damping, damping.transpose(-1, -2), rtol=1e-12, atol=1e-12)
    assert torch.linalg.eigvalsh(damping).min() >= -1e-10
    assert torch.all(torch.einsum("bi,bij,bj->b", q, damping, q) >= -1e-10)
    edge = q @ model.incidence
    expected_modulation = (model.incidence * ((8./model.incidence.shape[1])
                           * model.modulated_damping_strength * edge.sin().square())[..., None, :]) @ model.incidence.T
    torch.testing.assert_close(damping-model.effective_damping, expected_modulation,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("switch", ["harmonic_strength", "modulated_damping_strength"])
def test_enabled_scalar_switch_has_finite_gradient_at_exact_zero(switch):
    model = _model(harmonic=True, modulated_damping=True)
    parameter = getattr(model, switch)
    with torch.no_grad():
        parameter.zero_()
    q = _project(torch.linspace(-0.09, 0.12, 10, dtype=torch.float64)[None], model.mass)
    equilibrium = _project(torch.linspace(0.03, -0.06, 10, dtype=torch.float64)[None], model.mass)
    objective = model.potential(q, equilibrium).sum() if switch == "harmonic_strength" else model.damping_matrix(q).square().sum()
    gradient, = torch.autograd.grad(objective, parameter)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().item() > 1e-12
    disabled = _model()
    torch.testing.assert_close(model.effective_stiffness, disabled.effective_stiffness, rtol=0, atol=0)
    torch.testing.assert_close(model.effective_damping, disabled.effective_damping, rtol=0, atol=0)
    torch.testing.assert_close(model.potential(q, equilibrium), disabled.potential(q, equilibrium),
                               rtol=0, atol=0)
    torch.testing.assert_close(model.damping_matrix(q), disabled.damping_matrix(q)[None], rtol=0, atol=0)


def test_rhs_jacobian_matches_directional_difference_with_active_terms():
    model = _model(harmonic=True, modulated_damping=True)
    with torch.no_grad():
        model.harmonic_strength.fill_(0.6)
        model.modulated_damping_strength.fill_(1.0)
    q = _project(torch.linspace(-0.04, 0.07, 10, dtype=torch.float64)[None], model.mass)
    v = _project(torch.linspace(0.02, -0.03, 10, dtype=torch.float64)[None], model.mass)
    state = torch.cat((q, v, torch.linspace(-0.01, 0.02, 10, dtype=torch.float64)[None]), dim=1)
    power = torch.zeros_like(q)
    equilibrium = torch.zeros_like(q)
    direction = torch.linspace(-0.3, 0.4, state.shape[1], dtype=torch.float64)[None]
    _, jacobian_direction = torch.autograd.functional.jvp(
        lambda value: model.rhs(value, power, equilibrium), state, direction,
    )
    tangent = direction[:, :, None]
    field = model._rhs(power, equilibrium, tangent=True)
    field_value, tangent_value = field(state, tangent)
    torch.testing.assert_close(field_value, model.rhs(state, power, equilibrium), rtol=0, atol=0)
    torch.testing.assert_close(tangent_value[:, :, 0], jacobian_direction, rtol=2e-12, atol=2e-12)
    eps = 1e-6
    finite = (model.rhs(state + eps * direction, power, equilibrium) -
              model.rhs(state - eps * direction, power, equilibrium)) / (2 * eps)
    torch.testing.assert_close(tangent_value[:, :, 0], finite, rtol=2e-6, atol=2e-8)


def test_storage_energy_rate_matches_power_minus_dissipation():
    model = _model(harmonic=True, modulated_damping=True)
    with torch.no_grad():
        model.harmonic_strength.fill_(0.6)
        model.modulated_damping_strength.fill_(1.0)
    q = _project(torch.linspace(-0.06, 0.08, 10, dtype=torch.float64)[None], model.mass)
    v = _project(torch.linspace(0.04, -0.03, 10, dtype=torch.float64)[None], model.mass)
    z = torch.linspace(-0.02, 0.025, 10, dtype=torch.float64)[None]
    state = torch.cat((q, v, z), dim=1).requires_grad_()
    power = torch.linspace(0.01, -0.008, 10, dtype=torch.float64)[None]
    power = power - power.mean(dim=1, keepdim=True)
    equilibrium = _project(torch.linspace(0.02, -0.015, 10, dtype=torch.float64)[None], model.mass)

    def storage(value):
        q_value, v_value, z_value = value[:, :10], value[:, 10:20], value[:, 20:]
        return (0.5*(model.mass*v_value.square()).sum(dim=1)
                + model.potential(q_value, equilibrium) + 0.5*z_value.square().sum(dim=1))

    gradient = torch.autograd.grad(storage(state).sum(), state, create_graph=True)[0]
    rate = (gradient * model.rhs(state, power, equilibrium)).sum(dim=1)
    damping = model.damping_matrix(q)
    expected = ((v*power).sum(dim=1)
                - torch.einsum("bi,bij,bj->b", v, damping, v)
                - (z.square() / model.tau).sum(dim=1))
    torch.testing.assert_close(rate, expected, rtol=2e-10, atol=2e-12)


def test_three_step_gauss_newton_chain_has_parameter_gradient():
    candidate = _model(harmonic=True, modulated_damping=True)
    target = _model(harmonic=True, modulated_damping=True)
    with torch.no_grad():
        candidate.harmonic_strength.fill_(0.35)
        candidate.modulated_damping_strength.fill_(0.4)
        target.harmonic_strength.fill_(0.6)
        target.modulated_damping_strength.fill_(0.8)
        target.port_parameter.add_(torch.linspace(-0.3, 0.35, 10, dtype=torch.float64))
        target.rate_parameter.add_(torch.linspace(-0.35, 0.3, 10, dtype=torch.float64))
        target.edge_log_factor.copy_(torch.linspace(-0.3, 0.25, target.edge_log_factor.numel()))
        target.damping_log_factor.copy_(torch.linspace(0.25, -0.3, target.damping_log_factor.numel()))
    q0 = _project(torch.linspace(-0.08, 0.1, 10, dtype=torch.float64)[None], target.mass)
    v0 = _project(torch.linspace(0.05, -0.07, 10, dtype=torch.float64)[None], target.mass)
    hidden = torch.linspace(-0.08, 0.1, 10, dtype=torch.float64)[None]
    initial = torch.cat((q0, v0, hidden), dim=1)
    raw_power = torch.linspace(0.08, -0.07, 10, dtype=torch.float64)[None]
    power = raw_power - raw_power.mean(dim=1, keepdim=True)
    equilibrium = _project(torch.linspace(0.04, -0.03, 10, dtype=torch.float64)[None], target.mass)
    spend = lambda *_: None
    target_path = target.rollout(initial, power, equilibrium, 13, max_step=0.005,
                                 linear_fast=False, spend=spend).detach()
    q, v = target_path[:, :11, :10], target_path[:, :11, 10:20]
    target_forecast = target.rollout(target_path[:, 10], power, equilibrium, 3,
                                     max_step=0.005, linear_fast=False, spend=spend)[:, 1:, :20]
    scale = torch.ones(20, dtype=torch.float64)

    def objective():
        result = predict_from_history(candidate, q, v, power, equilibrium, scale, 3,
                                      max_step=0.005, spend=spend)
        return 1e6*(result["forecast"][:, 1:, :20]-target_forecast).square().mean()

    parameters = (
        (candidate.port_parameter, (3,)),
        (candidate.rate_parameter, (4,)),
        (candidate.edge_log_factor, (3,)),
        (candidate.damping_log_factor, (3,)),
        (candidate.harmonic_strength, ()),
        (candidate.modulated_damping_strength, ()),
    )
    gradients = torch.autograd.grad(objective(), tuple(parameter for parameter, _ in parameters))
    for (parameter, _), gradient in zip(parameters, gradients, strict=True):
        assert torch.isfinite(gradient).all()
        direction = torch.linspace(0.5, 1.5, parameter.numel(), dtype=torch.float64).reshape_as(parameter)
        original = parameter.detach().clone()
        epsilon = 2e-5
        with torch.no_grad():
            parameter.copy_(original + epsilon*direction)
        plus = float(objective().detach())
        with torch.no_grad():
            parameter.copy_(original - epsilon*direction)
        minus = float(objective().detach())
        with torch.no_grad():
            parameter.copy_(original)
        finite_difference = (plus-minus)/(2*epsilon)
        analytic = float((gradient*direction).sum())
        assert abs(analytic) > 1e-5
        assert finite_difference == pytest.approx(analytic, rel=2e-2, abs=5e-7)


def test_state_dict_restores_frozen_parameter_row_exactly():
    model = _model(harmonic=True, modulated_damping=True)
    with torch.no_grad():
        model.harmonic_strength.fill_(0.17)
        model.modulated_damping_strength.fill_(0.11)
        model.edge_log_factor.copy_(torch.linspace(-0.1, 0.1, model.edge_log_factor.numel()))
    q = _project(torch.linspace(-0.04, 0.05, 10, dtype=torch.float64)[None], model.mass)
    v = _project(torch.linspace(0.03, -0.02, 10, dtype=torch.float64)[None], model.mass)
    state = torch.cat((q, v, torch.linspace(-0.02, 0.02, 10, dtype=torch.float64)[None]), dim=1)
    power = torch.zeros_like(q)
    equilibrium = torch.zeros_like(q)
    before = model.rollout(state, power, equilibrium, 3, max_step=0.0025, linear_fast=False)
    restored = _model(harmonic=True, modulated_damping=True)
    restored.load_state_dict(model.state_dict())
    after = restored.rollout(state, power, equilibrium, 3, max_step=0.0025, linear_fast=False)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_causal_prediction_matches_cpu():
    cpu = _model(harmonic=True, modulated_damping=True)
    with torch.no_grad():
        cpu.harmonic_strength.fill_(0.6)
        cpu.modulated_damping_strength.fill_(1.0)
    q0 = _project(torch.linspace(-0.04, 0.05, 10, dtype=torch.float64)[None], cpu.mass)
    v0 = _project(torch.linspace(0.03, -0.02, 10, dtype=torch.float64)[None], cpu.mass)
    initial = torch.cat((q0, v0, torch.linspace(-0.02, 0.02, 10, dtype=torch.float64)[None]), dim=1)
    power = torch.linspace(0.02, -0.015, 10, dtype=torch.float64)[None]
    power = power - power.mean(dim=1, keepdim=True)
    equilibrium = _project(torch.linspace(-0.03, 0.04, 10, dtype=torch.float64)[None], cpu.mass)
    spend = lambda *_: None
    observed = cpu.rollout(initial, power, equilibrium, 10, max_step=0.005,
                           linear_fast=False, spend=spend).detach()
    q, v = observed[:, :, :10], observed[:, :, 10:20]
    gpu = _model(harmonic=True, modulated_damping=True).to("cuda")
    gpu.load_state_dict(cpu.state_dict())
    with torch.no_grad():
        expected = predict_from_history(cpu, q, v, power, equilibrium,
                                       torch.ones(20, dtype=torch.float64), 100,
                                       max_step=0.005, spend=spend)
        actual = predict_from_history(gpu, q.cuda(), v.cuda(), power.cuda(), equilibrium.cuda(),
                                     torch.ones(20, dtype=torch.float64, device="cuda"), 100,
                                     max_step=0.005, spend=spend)
    assert actual["iteration_histories"].shape[0] == 3
    assert actual["iteration_responses"].shape[0] == 3
    assert actual["forecast"].shape[1] == 101
    for key in ("iteration_histories", "iteration_responses", "history", "forecast"):
        torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=2e-8, atol=2e-10)
