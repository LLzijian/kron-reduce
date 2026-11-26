"""Known-truth ES-ROM checks; no trajectory files or rosters are opened."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from kron_reduce.data.task3_measurement_noise import NoiseDraw, inject_raw_history
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.noise_robustness_mhe import (
    coi_lift,
    joint_history_with_tangent,
    predict_mhe,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.unified_discovery_m3 import UnifiedDiscoveryModel, predict_m3
from kron_reduce.physics.coordinates import coi_project, coi_project_power


def _spend(*_):
    pass


@pytest.fixture(scope="module")
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    mass, stiffness, damping, port, tau, _ = parameters()
    empty = np.empty((0, 0))
    backbone = NonlinearPassiveClosure(mass, stiffness, damping, np.empty((10, 0)), empty, empty)
    result = UnifiedDiscoveryModel(backbone, port=port, tau=tau, support=("G2",),
                                   module_values={"rho": .61}).requires_grad_(False).eval()
    yield result
    torch.set_num_threads(previous)


def _truth(model, *, step, zero_memory=False, batch=1):
    rng = np.random.default_rng(20261126)
    mass = model.mass.cpu().numpy()
    angle = coi_project(rng.normal(0., .15, (batch, 10)), mass)
    speed = coi_project(rng.normal(0., .3, (batch, 10)), mass)
    memory = np.zeros_like(speed) if zero_memory else rng.normal(0., .03, (batch, 10))
    initial = model._tensor(np.concatenate((angle, speed, memory), axis=1))
    power = model._tensor(coi_project_power(rng.normal(0., .12, (batch, 10)), mass))
    equilibrium = model._tensor(rng.normal(0., .06, (batch, 10)))
    scale = model._tensor(np.repeat([.2, .4], 10))
    with torch.no_grad():
        history = model.rollout(initial, power, equilibrium, 10, max_step=step, spend=_spend)
    return history, power, equilibrium, scale


def _inputs(history, power, equilibrium, scale):
    return history[:, :, :10], history[:, :, 10:20], power, equilibrium, scale


def test_raw_noise_projection_preserves_coi_power_balance_and_future():
    rng = np.random.default_rng(20261126)
    mass = 1.+.1*np.arange(10)
    raw = rng.normal(size=(15, 20))
    raw_power = rng.normal(size=10)
    angle, speed, power = (rng.normal(0., s, shape) for s, shape in (
        (.01, (11, 10)), (.05, (11, 10)), (.01, (10,))))
    noisy, noisy_power = inject_raw_history(raw, raw_power, mass, NoiseDraw(angle, speed, power))
    for block, noise in ((slice(0, 10), angle), (slice(10, 20), speed)):
        projected = coi_project(noisy[:11, block], mass)
        np.testing.assert_allclose(projected @ mass, 0., atol=1e-13)
        np.testing.assert_allclose(projected-coi_project(raw[:11, block], mass),
                                   coi_project(noise, mass), atol=1e-14)
    projected_power = coi_project_power(noisy_power, mass)
    np.testing.assert_allclose(projected_power.sum(), 0., atol=1e-13)
    np.testing.assert_allclose(projected_power-coi_project_power(raw_power, mass),
                               coi_project_power(power, mass), atol=1e-14)
    np.testing.assert_array_equal(noisy[11:], raw[11:])
    assert not np.shares_memory(raw, noisy)


@pytest.mark.parametrize("step", (.0025, .00125))
def test_projected_joint_rk4_tangent_matches_all_28_central_differences(model, step):
    truth, power, equilibrium, _ = _truth(model, step=step)
    state = truth[:, 0]
    actual, tangent = joint_history_with_tangent(model, state, power, equilibrium, max_step=step, spend=_spend)
    lift = coi_lift(model.mass)
    torch.testing.assert_close(lift.T @ lift, torch.eye(28, dtype=lift.dtype), atol=1e-14, rtol=0.)
    torch.testing.assert_close(model.mass @ lift[:10], torch.zeros(28, dtype=lift.dtype), atol=1e-14, rtol=0.)
    torch.testing.assert_close(actual, truth, atol=1e-14, rtol=1e-14)
    epsilon = 2e-5
    starts = torch.cat((state+epsilon*lift.T, state-epsilon*lift.T))
    with torch.no_grad():
        paths = model.rollout(starts, power.expand(56, -1), equilibrium.expand(56, -1), 10,
                              max_step=step, spend=_spend)
    numeric = ((paths[:28]-paths[28:])/(2.*epsilon)).permute(1, 2, 0)
    torch.testing.assert_close(tangent[0], numeric, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize("step", (.0025, .00125))
def test_clean_observations_exactly_route_to_original_five_gn(model, step):
    history, power, equilibrium, scale = _truth(model, step=step)
    inputs = _inputs(history, power, equilibrium, scale)
    options = {"max_step": step, "spend": _spend, "memory_radius": .65}
    with torch.no_grad():
        original = predict_m3(model, *inputs, 3, **options, prior_weight=1e-6)
    new = predict_mhe(model, *inputs, 3, **options)
    assert not new["joint_active"].item() and new["map_lambda"].item() == 1e-6
    for key in ("history", "forecast", "hidden_start"):
        torch.testing.assert_close(new[key], original[key], atol=1e-10, rtol=0.)


def test_joint_solver_recovers_clean_nonstationary_start_below_1e_minus_10(model):
    step = .0025
    truth, power, equilibrium, scale = _truth(model, step=step, zero_memory=True)
    # The memory prior is centered on this known truth. Gate zero exposes the
    # actual 28-coordinate solve on the small nonzero Simpson truncation error.
    result = predict_mhe(model, *_inputs(truth, power, equilibrium, scale), 0,
                         max_step=step, spend=_spend, memory_radius=.65, gate_rad=0.)
    assert result["joint_active"].item()
    assert float((result["history"][:, 0]-truth[:, 0]).abs().max()) < 1e-10


def test_ten_sigma_first_and_last_frame_outliers_and_l2_ablation(model):
    step = .0025
    truth, power, equilibrium, scale = _truth(model, step=step, zero_memory=True, batch=2)
    observed = truth.clone()
    spike = np.zeros(10)
    spike[3] = 1.
    projected = model._tensor(coi_project(spike, model.mass.numpy()))
    for row, frame in ((0, 0), (1, 10)):
        observed[row, frame, :10] += .1*projected
        observed[row, frame, 10:20] -= .5*projected
    options = {"max_step": step, "spend": _spend, "memory_radius": .65}
    inputs = _inputs(observed, power, equilibrium, scale)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    outputs = {method: predict_mhe(model, *inputs, 1, method=method, **options) for method in ("gn", "l2", "huber")}
    assert not outputs["gn"]["joint_active"].any()
    assert outputs["l2"]["joint_active"].all() and outputs["huber"]["joint_active"].all()

    def endpoint_errors(result):
        start = (result["history"][0, 0, :20]-truth[0, 0, :20])/scale
        end = (result["origin"][1, :20]-truth[1, -1, :20])/scale
        return torch.stack((torch.linalg.vector_norm(start), torch.linalg.vector_norm(end)))

    errors = {method: endpoint_errors(result) for method, result in outputs.items()}
    assert torch.all(errors["huber"] < .25*errors["gn"])
    assert torch.all(errors["huber"] < errors["l2"])
    torch.testing.assert_close(outputs["l2"]["map_lambda"], outputs["huber"]["map_lambda"])
    torch.testing.assert_close(outputs["huber"]["origin"], outputs["huber"]["history"][:, -1], rtol=0., atol=0.)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0., rtol=0.)


def test_frozen_energy_balance_at_filtered_origin_below_1e_minus_12(model):
    step = .0025
    truth, power, equilibrium, scale = _truth(model, step=step)
    observed = truth.clone()
    perturbation = model._tensor(coi_project(np.arange(10)*.01, model.mass.numpy()))
    observed[:, -1, :10] += perturbation
    result = predict_mhe(model, *_inputs(observed, power, equilibrium, scale), 0,
                         max_step=step, spend=_spend, memory_radius=.65)
    assert result["joint_active"].item()
    origin = result["origin"].detach().requires_grad_(True)
    gradient, = torch.autograd.grad(model.storage(origin, equilibrium).sum(), origin)
    rate = (gradient*model.rhs(origin, power, equilibrium)).sum(dim=-1)
    supplied = (origin[:, 10:20]*power).sum(dim=-1)
    assert (rate-supplied+model.dissipation(origin, equilibrium)).abs().max() < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
def test_cuda_mixed_clean_and_noisy_batch_matches_cpu_reference(model):
    gpu = deepcopy(model).cuda()
    step = .00125
    truth, power, equilibrium, scale = _truth(model, step=step, zero_memory=True, batch=2)
    truth[1, -1, :10] += model._tensor(coi_project(np.arange(10)*.02, model.mass.numpy()))
    inputs = _inputs(truth, power, equilibrium, scale)
    settings = {"max_step": step, "spend": _spend, "memory_radius": .65}
    reference = predict_mhe(model, *inputs, 2, **settings)
    actual = predict_mhe(gpu, *inputs, 2, **settings)
    assert actual["joint_active"].tolist() == [False, True]
    assert actual["forecast"].is_cuda and actual["forecast"].dtype == torch.float64
    for key in ("history", "forecast", "map_lambda", "sigma_hat"):
        torch.testing.assert_close(actual[key].cpu(), reference[key], rtol=1e-9, atol=1e-11)
