"""Known-truth synthetic numerical gates for the robust ES-ROM initializer V2."""

import json
import tomllib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.noise_robustness_mhe import predict_mhe
from kron_reduce.models.noise_robustness_mhe_v2 import (
    TUKEY_CUTOFF,
    _tukey,
    kinematic_cleaner,
    predict_mhe_v2,
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
    result = UnifiedDiscoveryModel(
        backbone, port=port, tau=tau, support=("G2",), module_values={"rho": .61}
    ).requires_grad_(False).eval()
    yield result
    torch.set_num_threads(previous)


def _truth(model, *, step, batch=1, memory=True):
    """Fixed-seed known-truth rollout; fixture is independent of any real-data score."""
    rng = np.random.default_rng(20261127)
    mass = model.mass.cpu().numpy()
    angle = coi_project(rng.normal(0., .34, (batch, 10)), mass)
    speed = coi_project(rng.normal(0., .72, (batch, 10)), mass)
    latent = rng.normal(0., .16, (batch, 10)) if memory else np.zeros((batch, 10))
    initial = model._tensor(np.concatenate((angle, speed, latent), axis=1))
    power = model._tensor(coi_project_power(rng.normal(0., .18, (batch, 10)), mass))
    equilibrium = model._tensor(coi_project(rng.normal(0., .04, (batch, 10)), mass))
    scale = model._tensor(np.repeat([.2, .4], 10))
    with torch.no_grad():
        history = model.rollout(initial, power, equilibrium, 10, max_step=step, spend=_spend)
    return history, power, equilibrium, scale


def _inputs(history, power, equilibrium, scale):
    return history[:, :, :10], history[:, :, 10:20], power, equilibrium, scale


def _project_visible(history, mass):
    visible = history[:, :, :20].clone()
    for block in (slice(0, 10), slice(10, 20)):
        visible[:, :, block] = torch.as_tensor(
            coi_project(visible[:, :, block].cpu().numpy(), mass.cpu().numpy()),
            dtype=visible.dtype, device=visible.device,
        )
    return visible


def _assert_energy_identity(model, path, power, equilibrium):
    states = path[0].detach().requires_grad_(True)
    held_power = power.expand(len(states), -1)
    anchor = equilibrium.expand(len(states), -1)
    gradient, = torch.autograd.grad(model.storage(states, anchor).sum(), states)
    rate = (gradient * model.rhs(states, held_power, anchor)).sum(dim=-1)
    supplied = (states[:, 10:20] * held_power).sum(dim=-1)
    residual = rate - supplied + model.dissipation(states, anchor)
    assert residual.abs().max() < 1e-12


@pytest.mark.parametrize("step", (.0025, .00125))
def test_ten_machine_coi_energy_identity_holds_through_eight_seconds(model, step):
    truth, power, equilibrium, scale = _truth(model, step=step)
    with torch.no_grad():
        path = model.rollout(truth[:, 0], power, equilibrium, 800, max_step=step, spend=_spend)
    assert path.shape == (1, 801, 30)
    assert (path[:, :, :10] @ model.mass).abs().max() < 1e-12
    assert (path[:, :, 10:20] @ model.mass).abs().max() < 1e-12
    _assert_energy_identity(model, path, power, equilibrium)

    # Check the actual eight-second forecast issued by V2 after endpoint noise.
    observed = truth[:, :, :20].clone()
    rng = np.random.default_rng(20261130)
    noise = torch.as_tensor(rng.normal(size=observed.shape), dtype=observed.dtype)
    noise[:, :, :10] *= .01
    noise[:, :, 10:20] *= .05
    noise = _project_visible(noise, model.mass)
    observed += noise
    direction = model._tensor(coi_project(np.eye(10)[3], model.mass.numpy()))
    sigma = noise.std(dim=1, unbiased=True).mean(dim=0)
    observed[:, -1, :10] += (10. * sigma[3] / direction[3]) * direction
    observed[:, -1, 10:20] -= (10. * sigma[13] / direction[3]) * direction
    issued = predict_mhe_v2(
        model, observed[:, :, :10], observed[:, :, 10:20], power, equilibrium, scale, 800,
        max_step=step, spend=_spend, memory_radius=.65,
    )
    _assert_energy_identity(model, issued["forecast"], power, equilibrium)


@pytest.mark.parametrize("step", (.0025, .00125))
def test_v2_tangent_matches_centered_differences_in_all_28_directions(model, step):
    # Imported lazily so the shared public tangent implementation remains the one under test.
    from kron_reduce.models.noise_robustness_mhe import coi_lift, joint_history_with_tangent

    truth, power, equilibrium, _ = _truth(model, step=step)
    state = truth[:, 0]
    actual, tangent = joint_history_with_tangent(
        model, state, power, equilibrium, max_step=step, spend=_spend
    )
    lift = coi_lift(model.mass)
    torch.testing.assert_close(lift.T @ lift, torch.eye(28, dtype=lift.dtype), atol=1e-14, rtol=0.)
    torch.testing.assert_close(model.mass @ lift[:10], torch.zeros(28, dtype=lift.dtype), atol=1e-14, rtol=0.)
    torch.testing.assert_close(actual, truth, atol=1e-14, rtol=1e-14)
    epsilon = 2e-5
    starts = torch.cat((state + epsilon * lift.T, state - epsilon * lift.T))
    with torch.no_grad():
        paths = model.rollout(
            starts, power.expand(56, -1), equilibrium.expand(56, -1), 10,
            max_step=step, spend=_spend,
        )
    numeric = ((paths[:28] - paths[28:]) / (2. * epsilon)).permute(1, 2, 0)
    torch.testing.assert_close(tangent[0], numeric, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize("step", (.0025, .00125))
def test_clean_reconstruction_is_exact_and_preserves_legacy_bypass(model, step):
    truth, power, equilibrium, scale = _truth(model, step=step, memory=False)
    inputs = _inputs(truth, power, equilibrium, scale)
    options = {"max_step": step, "spend": _spend, "memory_radius": .65}
    with torch.no_grad():
        legacy = predict_m3(model, *inputs, 3, **options, prior_weight=1e-6)
        v1 = predict_mhe(model, *inputs, 3, **options)
        v2 = predict_mhe_v2(model, *inputs, 3, **options)
    assert not v2["joint_active"].item()
    for candidate in (v1, v2):
        assert not candidate["joint_active"].item()
        for key in ("history", "forecast", "hidden_start", "origin"):
            torch.testing.assert_close(candidate[key], legacy.get(key, legacy["forecast"][:, 0]), atol=1e-10, rtol=0.)
    assert float((v2["history"][:, 0, :20] - truth[:, 0, :20]).abs().max()) < 1e-10
    assert float((v2["origin"][:, :20] - truth[:, -1, :20]).abs().max()) < 1e-10


@pytest.mark.parametrize("step", (.0025, .00125))
def test_v2_rejects_only_the_pulsed_channel_at_each_frame(model, step):
    truth, _, _, scale = _truth(model, step=step, memory=False)
    rng = np.random.default_rng(20261128)
    observed = truth[:, :, :20].clone()
    channel_sigma = .001 * scale
    noise = torch.as_tensor(rng.normal(size=observed.shape), dtype=observed.dtype) * channel_sigma
    observed += noise
    before = kinematic_cleaner(observed, scale)["weight"]
    observed[0, 0, 3] += 10. * channel_sigma[3]
    diagnostics = kinematic_cleaner(observed, scale)
    weights = diagnostics["weight"]
    assert weights.shape == (1, 11, 20)
    assert weights[0, 0, 3] == 0.
    unrelated = torch.arange(20) != 3
    torch.testing.assert_close(weights[:, :, unrelated], before[:, :, unrelated], atol=0., rtol=0.)
    assert diagnostics["clean"][0, 0, 3].isfinite()


@pytest.mark.parametrize("step", (.0025, .00125))
def test_lm_acceptance_is_monotone_and_near_threshold_outliers_leak_smoothly(model, step):
    truth, power, equilibrium, scale = _truth(model, step=step, memory=False)
    clean_observed = truth[:, :, :20].clone()
    baseline_weight = kinematic_cleaner(clean_observed, scale)["weight"]
    one_channel_pulse = clean_observed.clone()
    one_channel_pulse[0, -1, 3] += .003
    pulse_weight = kinematic_cleaner(one_channel_pulse, scale)["weight"]
    assert pulse_weight[0, -1, 3] < baseline_weight[0, -1, 3]
    unrelated = torch.arange(20) != 3
    torch.testing.assert_close(pulse_weight[:, :, unrelated], baseline_weight[:, :, unrelated],
                               atol=0., rtol=0.)

    # A balanced perturbation activates LM; accepted objective values must not rise.
    observed = clean_observed.clone()
    direction = model._tensor(coi_project(np.eye(10)[3], model.mass.numpy()))
    observed[:, -1, :10] += .003 * direction
    observed = _project_visible(observed, model.mass)
    assert torch.isfinite(pulse_weight).all() and ((pulse_weight >= 0.) & (pulse_weight <= 1.)).all()
    result = predict_mhe_v2(
        model, observed[:, :, :10], observed[:, :, 10:20], power, equilibrium, scale, 0,
        max_step=step, spend=_spend, memory_radius=.65, gate_rad=0.,
    )
    objectives = result["lm_objectives"]
    accepted = objectives[torch.isfinite(objectives)]
    assert len(accepted) >= 2
    assert torch.all(accepted[1:] <= accepted[:-1] + 1e-14)


def test_tukey_weight_has_bounded_near_threshold_leakage():
    standardized = torch.tensor([TUKEY_CUTOFF - 1e-3, TUKEY_CUTOFF, TUKEY_CUTOFF + 1e-3],
                                dtype=torch.float64)
    weights = _tukey(standardized)
    assert 0. < weights[0] < 1e-5
    assert weights[1] == 0.
    assert weights[2] == 0.


def synthetic_endpoint_battery(model, step, *, device=None):
    """Return JSON-ready scores for the fixed 40-case paired synthetic battery."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not model.mass.is_cuda:
        model = deepcopy(model).cuda()
    elif device == "cpu" and model.mass.is_cuda:
        model = model.cpu()
    truth, power, equilibrium, scale = _truth(model, step=step, batch=1)
    rows = [(machine, endpoint, sign)
            for machine in range(10)
            for endpoint in (0, 10)
            for sign in (1., -1.)]
    truth = truth.repeat(len(rows), 1, 1)
    power = power.repeat(len(rows), 1)
    equilibrium = equilibrium.repeat(len(rows), 1)
    observed = truth[:, :, :20].clone()
    rng = np.random.default_rng(20261129)
    background = torch.as_tensor(rng.normal(size=(1, 11, 20)), dtype=observed.dtype,
                                 device=observed.device)
    background[:, :, :10] *= .0033
    background[:, :, 10:20] *= .010
    observed += _project_visible(background, model.mass).repeat(len(rows), 1, 1)
    mass = model.mass.cpu().numpy()
    for row, (machine, endpoint, sign) in enumerate(rows):
        direction = model._tensor(coi_project(np.eye(10)[machine], mass))
        observed[row, endpoint, :10] += sign * .100 * direction
        observed[row, endpoint, 10:20] -= sign * .500 * direction

    inputs = (observed[:, :, :10], observed[:, :, 10:20], power, equilibrium, scale)
    settings = {"max_step": step, "spend": _spend, "memory_radius": .65}
    config = Path(__file__).resolve().parents[1] / "configs/experiments/noise_robustness_esrom_v2.toml"
    with config.open("rb") as stream:
        plan = tomllib.load(stream)
    common = {key: plan[key] for key in ("gate_rad", "huber_delta")}
    candidates = {row["name"]: {key: row[key] for key in ("alpha", "tau_mis", "inverse_snr")}
                  for row in plan["candidates"]}
    outputs = {
        "orig": predict_m3(model, *inputs, 800, **settings, prior_weight=1e-6),
        "v1": predict_mhe(model, *inputs, 800, **settings, gate_rad=.0003, huber_delta=1.5, alpha=.1),
    }
    for name, options in candidates.items():
        outputs[name] = predict_mhe_v2(model, *inputs, 800, **settings, **common, **options)
    with torch.no_grad():
        reference = model.rollout(truth[:, -1], power, equilibrium, 800,
                                  max_step=step, spend=_spend)

    scores = {}
    for name, result in outputs.items():
        predicted = result["forecast"]
        assert predicted.shape == (40, 801, 30)
        assert torch.isfinite(predicted).all()
        assert (predicted[:, :, :10] @ model.mass).abs().max() < 1e-12
        assert (predicted[:, :, 10:20] @ model.mass).abs().max() < 1e-12
        target_peak_gap = torch.empty(40, dtype=predicted.dtype, device=predicted.device)
        global_peak_error = (predicted[:, 1:51, 10:20]
                             - reference[:, 1:51, 10:20]).abs().amax(dim=(1, 2))
        nrmse_scale = predicted.new_tensor([.476753] * 10 + [1.003367] * 10)
        normalized_square = ((predicted[:, 1:801, :20] - reference[:, 1:801, :20])
                             / nrmse_scale) ** 2
        application_peak_gap = (predicted[:, 1:51, 10:20].abs().amax(dim=(1, 2))
                                - reference[:, 1:51, 10:20].abs().amax(dim=(1, 2))).abs()
        for row, (machine, _, _) in enumerate(rows):
            actual_peak = reference[row, 1:51, 10 + machine].abs().amax()
            forecast_peak = predicted[row, 1:51, 10 + machine].abs().amax()
            target_peak_gap[row] = (forecast_peak - actual_peak).abs()
        scores[name] = {
            "target_peak_gap": target_peak_gap,
            "global_peak_error": global_peak_error,
            "application_peak_gap": application_peak_gap,
            "nrmse": torch.sqrt(normalized_square.mean(dim=(1, 2))),
            "aggregate_nrmse": torch.sqrt(normalized_square.mean()),
        }

    endpoint_scores = {}
    per_machine = {}
    for endpoint in (0, 10):
        indices = [i for i, (_, case_endpoint, _) in enumerate(rows)
                   if case_endpoint == endpoint]
        endpoint_scores[str(endpoint)] = {
            name: float(scores[name]["target_peak_gap"][indices].mean().cpu())
            for name in outputs
        }
    for machine in range(10):
        per_machine[str(machine)] = {}
        for endpoint in (0, 10):
            indices = [i for i, (m, ep, _) in enumerate(rows)
                       if m == machine and ep == endpoint]
            per_machine[str(machine)][str(endpoint)] = {
                name: float(scores[name]["target_peak_gap"][indices].mean().cpu())
                for name in outputs
            }
    model_scores = {}
    for name in outputs:
        model_scores[name] = {
            "target_peak_gap": [float(x) for x in scores[name]["target_peak_gap"].cpu()],
            "global_peak_error": [float(x) for x in scores[name]["global_peak_error"].cpu()],
            "application_peak_gap": [float(x) for x in scores[name]["application_peak_gap"].cpu()],
            "nrmse": [float(x) for x in scores[name]["nrmse"].cpu()],
            "aggregate_nrmse": float(scores[name]["aggregate_nrmse"].cpu()),
        }
    gates = {
        name: {
            baseline: {
                "first_endpoint_peak": endpoint_scores["0"][name] <= .9 * endpoint_scores["0"][baseline],
                "last_endpoint_peak": endpoint_scores["10"][name] <= .9 * endpoint_scores["10"][baseline],
                "all_case_nrmse": model_scores[name]["aggregate_nrmse"] < model_scores[baseline]["aggregate_nrmse"],
            }
            for baseline in ("orig", "v1")
        }
        for name in candidates
    }
    passing = [name for name, comparisons in gates.items() if all(all(row.values()) for row in comparisons.values())]
    return {
        "status": "PASS" if passing else "FAIL",
        "step_s": step,
        "device": str(model.mass.device),
        "cases": [{"machine": m, "endpoint": endpoint, "sign": sign} for m, endpoint, sign in rows],
        "endpoint_target_peak_means": endpoint_scores,
        "per_machine_target_peak_means": per_machine,
        "models": model_scores,
        "gates": gates,
        "passing_candidates": passing,
    }


def test_matched_endpoint_signed_pulse_battery_across_all_machines(model, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        summaries = [synthetic_endpoint_battery(model, step) for step in (.0025, .00125)]
    finally:
        torch.use_deterministic_algorithms(previous)
    for summary in summaries:
        print("V2_PULSE_DIAGNOSTICS " + json.dumps(
            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ))
    passing = set(summaries[0]["passing_candidates"]) & set(summaries[1]["passing_candidates"])
    assert passing, "No registered V2 candidate passed the 40-case endpoint battery at both RK4 steps."


@pytest.mark.parametrize("step", (.0025, .00125))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
def test_cuda_float64_deterministic_v2_matches_cpu(model, step, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    gpu = deepcopy(model).cuda()
    truth, power, equilibrium, scale = _truth(model, step=step, memory=False)
    observed = truth.to(gpu.device).clone()
    observed[:, -1, :10] += gpu._tensor(coi_project(np.arange(10) * .01, model.mass.cpu().numpy()))
    inputs = _inputs(observed, power, equilibrium, scale)
    settings = {"max_step": step, "spend": _spend, "memory_radius": .65}
    reference = predict_mhe_v2(model, *inputs, 2, **settings)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual = predict_mhe_v2(gpu, *inputs, 2, **settings)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert actual["forecast"].is_cuda and actual["forecast"].dtype == torch.float64
    assert torch.equal(actual["joint_active"].cpu(), reference["joint_active"].cpu())
    for key in ("history", "forecast", "origin", "map_lambda", "sigma_hat", "lm_objectives"):
        torch.testing.assert_close(actual[key].cpu(), reference[key], rtol=1e-9, atol=1e-11, equal_nan=True)
