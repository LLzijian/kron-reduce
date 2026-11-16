import numpy as np
import pytest
import torch

import kron_reduce.models.machine_local_topology as topology_history
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.evaluation.topology_memory_rom_truth import truth_rollout
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.topology_memory_rom import TopologyMemoryROM


def _model(*, common=False, spatial=False):
    mass, stiffness, damping, port, tau, _ = parameters()
    empty = np.zeros((0, 0), dtype=np.float64)
    backbone = NonlinearPassiveClosure(mass, stiffness, damping, np.zeros((10, 0)), empty, empty)
    return TopologyMemoryROM(backbone, port=port, tau=tau, common=common, spatial=spatial,
                             saturation=True)


def _balanced(x, mass):
    return x-(x@mass)[:, None]/mass.sum()


@pytest.mark.parametrize(("common", "spatial", "expected"),
                         [(False, False, 0), (True, False, 1), (False, True, 9), (True, True, 10)])
def test_topology_memory_rom_layout_and_analytic_tangent(common, spatial, expected):
    model = _model(common=common, spatial=spatial)
    assert model.dynamic_dimension == expected
    assert model.latent_dimension == model.memory_dimension+expected
    assert model.topology_stiffness == (1.,)*(int(common)+int(spatial))

    generator = torch.Generator().manual_seed(23)
    n, width = model.n_nodes, 3*model.n_nodes+expected
    state = torch.randn((2, width), generator=generator, dtype=torch.float64)*.025
    state[:, :n] = _balanced(state[:, :n], model.mass)
    state[:, n:2*n] = _balanced(state[:, n:2*n], model.mass)
    state[:, 2*n:3*n] += .015
    if expected:
        state[:, 3*n:] += .02
    power = _balanced(torch.randn((2, n), generator=generator, dtype=torch.float64)*.01,
                      model.mass)
    equilibrium = torch.randn((2, n), generator=generator, dtype=torch.float64)*.04
    with torch.no_grad():
        model.topology_rates.fill_(.1)
        model.saturation_strength.fill_(.35)

    direction = torch.randn(state.shape, generator=generator, dtype=state.dtype)
    field, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])
    torch.testing.assert_close(field, model.rhs(state, power, equilibrium))
    epsilon = 1e-6
    numerical = (model.rhs(state+epsilon*direction, power, equilibrium)
                 -model.rhs(state-epsilon*direction, power, equilibrium))/(2*epsilon)
    torch.testing.assert_close(tangent[..., 0], numerical, atol=2e-8, rtol=2e-7)


def _history(model):
    n, r, d = model.n_nodes, model.memory_dimension, model.dynamic_dimension
    generator = torch.Generator().manual_seed(7)
    q = _balanced(torch.randn((1, n), generator=generator, dtype=torch.float64)*.06, model.mass)
    v = _balanced(torch.randn((1, n), generator=generator, dtype=torch.float64)*.025, model.mass)
    hidden = torch.full((1, r), .012, dtype=torch.float64)
    dynamic = torch.full((1, d), .025, dtype=torch.float64)
    initial = torch.cat((q, v, hidden, dynamic), dim=1)
    power = torch.randn((1, n), generator=generator, dtype=torch.float64)*.005
    power = power@model.power_projection.T
    equilibrium = torch.full((1, n), .03, dtype=torch.float64)
    with torch.no_grad():
        model.topology_rates.fill_(.1)
        model.saturation_strength.fill_(.15)
    observed = model.rollout(initial, power, equilibrium, 10, max_step=.005, spend=lambda *_: None)
    return observed[:, :, :n], observed[:, :, n:2*n], power, equilibrium


def test_history_fit_dynamic_flag_bounds_and_optional_data_jacobian(monkeypatch):
    model = _model(common=True)
    q, v, power, equilibrium = _history(model)
    widths = []

    def unit_update(design, rhs, *, spend):
        widths.append(design.shape[-1])
        return {"hidden_start":torch.ones((len(design), design.shape[-1]), dtype=design.dtype),
                "singular":torch.ones((len(design), design.shape[-1]), dtype=design.dtype)}

    monkeypatch.setattr(topology_history, "solve_hidden_batch", unit_update)
    legacy = predict_from_topology_history(model, q, v, power, equilibrium, np.ones(20), 0,
                                           max_step=.005, spend=lambda *_: None)
    assert "observation_jacobian" not in legacy
    assert legacy["hidden_start"].shape == (1, model.memory_dimension)
    torch.testing.assert_close(legacy["history"][:, 0, -model.dynamic_dimension:],
                               torch.zeros((1, model.dynamic_dimension), dtype=torch.float64))
    assert widths[-5:] == [model.memory_dimension]*5

    widths.clear()
    fitted = predict_from_topology_history(
        model, q, v, power, equilibrium, np.ones(20), 0, max_step=.005, spend=lambda *_: None,
        fit_dynamic_coupling=True, memory_radius=.1, dynamic_radius=.04,
        return_observation_jacobian=True)
    assert widths == [model.memory_dimension+model.dynamic_dimension]*5
    assert torch.linalg.vector_norm(fitted["hidden_start"][:, :model.memory_dimension]) <= .1+1e-12
    assert torch.linalg.vector_norm(fitted["hidden_start"][:, model.memory_dimension:]) <= .04+1e-12
    assert fitted["observation_jacobian"].shape == (1, 20*model.n_nodes, model.latent_dimension)
    assert fitted["observation_jacobian_iterate"] == 4


def test_optional_data_jacobian_preserves_a_fully_flat_saturated_map(monkeypatch):
    model = _model(common=True)
    q, v, power, equilibrium = _history(model)

    def flat_history(model, initial, power, equilibrium, *, max_step, spend):
        path = model.rollout(initial, power, equilibrium, 10, max_step=max_step, spend=spend)
        tangent = torch.zeros((len(initial), 11, initial.shape[1], model.latent_dimension),
                              dtype=initial.dtype, device=initial.device)
        return path, tangent

    def no_update(design, rhs, *, spend):
        return {"hidden_start":torch.zeros((len(design), design.shape[-1]), dtype=design.dtype),
                "singular":torch.zeros((len(design), design.shape[-1]), dtype=design.dtype)}

    monkeypatch.setattr(topology_history, "history_with_tangent", flat_history)
    monkeypatch.setattr(topology_history, "solve_hidden_batch", no_update)
    result = predict_from_topology_history(
        model, q, v, power, equilibrium, np.ones(20), 0, max_step=.005, spend=lambda *_: None,
        fit_dynamic_coupling=True, return_observation_jacobian=True)
    assert result["observation_jacobian_iterate"] == 4
    assert torch.count_nonzero(result["observation_jacobian"]) == 0


def test_combined_topology_and_memory_agree_with_independent_truth_and_power_balance():
    model = _model(common=True, spatial=True)
    model.set_angle_damping(2/3, spatial_mix=1.)
    generator = torch.Generator().manual_seed(951)
    state = .15*torch.randn((3, 40), generator=generator, dtype=model.mass.dtype)
    q, v, z, c, xi = state.split((10, 10, 10, 1, 9), dim=1)
    q, v = _balanced(q, model.mass), _balanced(v, model.mass)
    state = torch.cat((q, v, z, c, xi), dim=1)
    power = torch.randn((3, 10), generator=generator, dtype=model.mass.dtype) @ model.power_projection.T
    equilibrium = torch.zeros_like(power)
    with torch.no_grad():
        model.topology_rates.copy_(model.mass.new_tensor((.23, .45)))
        model.saturation_strength.fill_(.6)
        path = model.rollout(state, power, equilibrium, 20, max_step=.0025, spend=lambda *_: None)
    independent = truth_rollout(parameters(), q.numpy(), v.numpy(), power.numpy(), z.numpy(),
                                count=20, max_step=.0025, spend=lambda *_: None, topology="both",
                                c=c[:, 0].numpy(), xi=xi.numpy(), gamma_s=.6,
                                topology_dynamics=((.23, .45), (1., 1.)))
    torch.testing.assert_close(path, torch.from_numpy(independent), atol=2e-11, rtol=2e-11)

    state.requires_grad_()
    energy_gradient, = torch.autograd.grad(model.storage(state, equilibrium).sum(), state)
    derivative = model.rhs(state, power, equilibrium)
    incident_squares = (q @ model.incidence).square() @ model.incident_weights.T
    weighted_damping = (v @ model._damping_root()).square()*(1-(2/3)*incident_squares/(1+incident_squares))
    mobility = torch.repeat_interleave(model.topology_rates, torch.tensor([1, 9]))
    dissipation = weighted_damping.sum(dim=-1)+(z.square()/model.tau).sum(dim=-1)
    dissipation += (mobility*energy_gradient[:, 30:].square()).sum(dim=-1)
    torch.testing.assert_close((energy_gradient*derivative).sum(dim=-1), (v*power).sum(dim=-1)-dissipation,
                               atol=2e-13, rtol=2e-13)


def test_shared_parameter_derivative_includes_causal_history_solve():
    model = _model(common=True, spatial=True)
    q, v, power, equilibrium = (x.detach() for x in _history(model))

    def objective(candidate):
        result = predict_from_topology_history(
            candidate, q, v, power, equilibrium, np.ones(20), 5, max_step=.005,
            spend=lambda *_: None, prior_weight=1e-5, fit_dynamic_coupling=True,
        )
        return result["forecast"][:, -1, 10:20].square().sum()

    parameters = (model.topology_rates, model.saturation_strength)
    directions = (model.mass.new_tensor((.33, -.27)), model.mass.new_tensor(.56))
    derivative = sum((x*y).sum() for x, y in zip(torch.autograd.grad(objective(model), parameters), directions))
    restored = _model(common=True, spatial=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(objective(restored), objective(model), atol=0., rtol=0.)
        other = (restored.topology_rates, restored.saturation_strength)
        readings = []
        delta = 2e-5
        for sign in (-1., 1.):
            for target, center, direction in zip(other, parameters, directions):
                target.copy_(center+sign*delta*direction)
            readings.append(objective(restored))
    assert derivative.abs() > 1e-9
    torch.testing.assert_close(derivative, (readings[1]-readings[0])/(2*delta), atol=3e-9, rtol=3e-5)
