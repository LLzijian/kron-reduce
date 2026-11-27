"""Numerical invariants and paired, known-truth endpoint comparison for ES-ROM V3."""

import json
import math
import os
import tomllib
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_noise_robustness_mhe_v2 import (
    _assert_energy_identity,
    _inputs,
    _project_visible,
    _spend,
    _truth,
    model,
)
from test_noise_robustness_mhe_v2 import (
    test_v2_tangent_matches_centered_differences_in_all_28_directions as _check_tangent,
)

from kron_reduce.models.noise_robustness_mhe import predict_mhe
from kron_reduce.models.noise_robustness_mhe_v3 import predict_mhe_v3
from kron_reduce.models.unified_discovery_m3 import predict_m3
from kron_reduce.physics.coordinates import coi_project

STEPS = (.0025, .00125)
TRACK_B_FACTOR = 1.08
TRACK_A_MINIMUM_REDUCTION = .10
CASES = [(machine, endpoint, sign) for machine in range(10)
         for endpoint in (0, 10) for sign in (1., -1.)]
PLAN = Path(__file__).resolve().parents[1] / "configs/experiments/noise_robustness_esrom_v3.toml"


def _settings(step):
    return {"max_step": step, "spend": _spend, "memory_radius": .65}


def _truth_model(predictor, track):
    if track == "A":
        return predictor
    if track != "B":
        raise ValueError(f"unknown synthetic track {track!r}")
    truth = deepcopy(predictor)
    with torch.no_grad():
        truth.edge_log_factor.add_(math.log(TRACK_B_FACTOR))
        truth.damping_log_factor.add_(math.log(TRACK_B_FACTOR))
    return truth


def _endpoint_fixture(truth_model, step):
    # Match the V2 battery's iteration order, seed, background and actual COI pulse.
    truth, power, equilibrium, scale = _truth(truth_model, step=step, batch=1)
    truth = truth.repeat(len(CASES), 1, 1)
    power = power.repeat(len(CASES), 1)
    equilibrium = equilibrium.repeat(len(CASES), 1)
    observed = truth[:, :, :20].clone()
    rng = np.random.default_rng(20261129)
    background = torch.as_tensor(rng.normal(size=(1, 11, 20)), dtype=observed.dtype,
                                 device=observed.device)
    background[:, :, :10] *= .0033
    background[:, :, 10:20] *= .010
    observed += _project_visible(background, truth_model.mass).repeat(len(CASES), 1, 1)
    mass = truth_model.mass.cpu().numpy()
    for row, (machine, endpoint, sign) in enumerate(CASES):
        direction = truth_model._tensor(coi_project(np.eye(10)[machine], mass))
        observed[row, endpoint, :10] += sign * .100 * direction
        observed[row, endpoint, 10:20] -= sign * .500 * direction
    return truth, observed, power, equilibrium, scale


def _coi_residual(path, mass):
    return torch.maximum((path[:, :, :10] @ mass).abs().max(),
                         (path[:, :, 10:20] @ mass).abs().max())


def _pulse_weights(result, cases):
    weights = result["outlier_weight"]
    row = torch.arange(len(cases), device=weights.device)
    machine, endpoint = torch.tensor([case[:2] for case in cases], device=weights.device).T
    return torch.stack((weights[row, endpoint, machine], weights[row, endpoint, machine + 10]), 1)


def _assert_finite_adaptation(result):
    for key in ("mismatch_variance", "endpoint_gain", "outlier_weight", "channel_noise",
                "map_lambda", "sigma_hat", "condition"):
        assert torch.isfinite(result[key]).all(), key
        assert (result[key] >= 0.).all(), key
    for key in ("endpoint_gain", "outlier_weight"):
        assert (result[key] <= 1.).all(), key
    objectives = result["lm_step_objectives"][result["joint_active"]]
    cutoffs = result["lm_cutoffs"][result["joint_active"]]
    assert (torch.isfinite(cutoffs) & (cutoffs > 0.)).all()
    assert torch.isfinite(objectives).all()
    assert (objectives[..., 1] <= objectives[..., 0] + 1e-14).all()


@pytest.mark.parametrize("step", STEPS)
def test_v3_clean_causal_fallback_matches_actual_truth(model, step):
    truth, power, equilibrium, scale = _truth(model, step=step, memory=False)
    inputs = _inputs(truth, power, equilibrium, scale)
    settings = _settings(step)
    original = predict_m3(model, *inputs, 3, **settings, prior_weight=1e-6)
    result = predict_mhe_v3(model, *inputs, 3, **settings)
    with torch.no_grad():
        reference = model.rollout(truth[:, -1], power, equilibrium, 3, max_step=step, spend=_spend)
    assert not result["joint_active"].item()
    for key in ("history", "forecast", "hidden_start"):
        torch.testing.assert_close(result[key], original[key], atol=1e-10, rtol=0.)
    assert (result["history"] - truth).abs().max() < 1e-10
    assert (result["forecast"] - reference).abs().max() < 1e-10
    _assert_finite_adaptation(result)


@pytest.mark.parametrize("step", STEPS)
def test_v3_joint_history_tangent_matches_all_28_finite_differences(model, step):
    # V3 uses the same projected RK4 tangent integrator as V1 and V2.
    _check_tangent(model, step)


@pytest.mark.parametrize("step", STEPS)
def test_v3_rejects_actual_endpoint_pulses_and_issues_finite_passive_forecast(model, step):
    _, observed, power, equilibrium, scale = _endpoint_fixture(model, step)
    indices = [12, 14]  # Machine 3, positive pulse at the first and at the last frame.
    actual_cases = [CASES[index] for index in indices]
    result = predict_mhe_v3(
        model, observed[indices, :, :10], observed[indices, :, 10:20],
        power[indices], equilibrium[indices], scale, 800, **_settings(step),
    )
    assert result["joint_active"].all()
    assert (_pulse_weights(result, actual_cases) == 0.).all()
    _assert_finite_adaptation(result)
    weights = result["outlier_weight"]
    residual = (result["smooth"] - result["history"][:, :, :20]) / scale
    discrepancy = (weights * residual.square()).sum(1) / weights.sum(1).clamp_min(1e-6)
    noise_var = result["channel_noise"].square()
    expected_variance = (discrepancy - 2.5 * noise_var).clamp_min(0.)
    torch.testing.assert_close(result["mismatch_variance"], expected_variance, atol=1e-14, rtol=0.)
    expected_gain = expected_variance / (expected_variance + noise_var + 1e-12)
    torch.testing.assert_close(result["endpoint_gain"], expected_gain, atol=1e-14, rtol=0.)
    path = result["forecast"]
    assert path.shape == (2, 801, 30)
    assert torch.isfinite(path).all()
    assert _coi_residual(path, model.mass) < 1e-12
    for row, index in enumerate(indices):
        _assert_energy_identity(model, path[row:row+1], power[index:index+1],
                                equilibrium[index:index+1])


def test_v3_track_b_scales_truth_only_without_resampling(model):
    truth = _truth_model(model, "B")
    q = model._tensor(np.linspace(-.2, .3, 10))
    for original, shifted in (
        (model.effective_stiffness, truth.effective_stiffness),
        (model.effective_damping, truth.effective_damping),
        (model.damping_matrix(q), truth.damping_matrix(q)),
    ):
        torch.testing.assert_close(shifted, 1.08 * original, atol=1e-14, rtol=1e-14)
    for name, saved in model.state_dict().items():
        if name not in {"edge_log_factor", "damping_log_factor"}:
            torch.testing.assert_close(saved, truth.state_dict()[name], atol=0., rtol=0.)
    a = _endpoint_fixture(model, STEPS[0])
    b = _endpoint_fixture(truth, STEPS[0])
    torch.testing.assert_close(a[0][:, 0], b[0][:, 0], atol=0., rtol=0.)
    assert not torch.equal(a[0][:, -1], b[0][:, -1])
    torch.testing.assert_close(a[1] - a[0][:, :, :20], b[1] - b[0][:, :, :20],
                               atol=5e-16, rtol=0.)
    for shared_a, shared_b in zip(a[2:], b[2:], strict=True):
        torch.testing.assert_close(shared_a, shared_b, atol=0., rtol=0.)


def _score(predicted, reference):
    frame = torch.arange(len(CASES), device=predicted.device)
    machine = torch.tensor([case[0] for case in CASES], device=predicted.device)
    predicted_peak = predicted[:, 1:51, 10:20].abs().amax(dim=1)
    reference_peak = reference[:, 1:51, 10:20].abs().amax(dim=1)
    target_peak_gap = (predicted_peak[frame, machine] - reference_peak[frame, machine]).abs()
    global_peak_error = (predicted[:, 1:51, 10:20] - reference[:, 1:51, 10:20]).abs().amax(dim=(1, 2))
    application_peak_gap = (predicted_peak.amax(dim=1) - reference_peak.amax(dim=1)).abs()
    nrmse_scale = predicted.new_tensor([.476753] * 10 + [1.003367] * 10)
    normalized_square = ((predicted[:, 1:801, :20] - reference[:, 1:801, :20]) / nrmse_scale) ** 2
    return {
        "target_peak_gap": target_peak_gap.tolist(),
        "global_peak_error": global_peak_error.tolist(),
        "application_peak_gap": application_peak_gap.tolist(),
        "nrmse": normalized_square.mean(dim=(1, 2)).sqrt().tolist(),
        "aggregate_nrmse": float(normalized_square.mean().sqrt()),
    }


def _support(result):
    joint = result["joint_active"]
    variance = result["mismatch_variance"]
    noisy = (result["channel_noise"] > 0.) & joint[:, None]
    positive = (variance > 0.) & noisy
    pulse_weights = _pulse_weights(result, CASES)
    accepted = result["lm_step_objectives"][joint]
    cutoffs = result["lm_cutoffs"][joint]
    return {
        "events": len(joint),
        "channels_per_event": variance.shape[1],
        "joint_active_events": int(joint.sum()),
        "noisy_active_events": int(noisy.any(dim=1).sum()),
        "noisy_active_channels": int(noisy.sum()),
        "positive_mismatch_events": int(positive.any(dim=1).sum()),
        "positive_mismatch_noisy_active_channels": int(positive.sum()),
        "positive_mismatch_noisy_active_channels_per_case": positive.sum(dim=1).tolist(),
        "pulse_weight_zero_events": int((pulse_weights == 0.).all(dim=1).sum()),
        "pulse_weight_zero_channels": int((pulse_weights == 0.).sum()),
        "adaptive_finite": bool((torch.isfinite(cutoffs) & (cutoffs > 0.)).all())
        and all(bool(torch.isfinite(result[key]).all()) for key in (
            "mismatch_variance", "endpoint_gain", "outlier_weight", "channel_noise",
            "map_lambda", "sigma_hat", "condition",
        )),
        "lm_objective_monotone": bool(torch.isfinite(accepted).all()
                                      and (accepted[..., 1] <= accepted[..., 0] + 1e-14).all()),
    }


def _candidate_gates(track, scores, endpoint_scores, support, name):
    def compare(value, reference, baseline):
        if value is None or reference is None:
            return False
        if track == "B":
            return value < reference
        return value <= (1. - TRACK_A_MINIMUM_REDUCTION if baseline == "orig" else 1.) * reference

    return {
        "comparisons": {
            baseline: {
                "first_endpoint_peak": compare(endpoint_scores["0"][name],
                                               endpoint_scores["0"][baseline], baseline),
                "last_endpoint_peak": compare(endpoint_scores["10"][name],
                                              endpoint_scores["10"][baseline], baseline),
                "all_case_nrmse": compare(scores[name]["aggregate_nrmse"],
                                          scores[baseline]["aggregate_nrmse"], baseline),
            }
            for baseline in ("orig", "v1")
        },
        "numerical_pass": scores[name]["status"] == "PASS",
        "empirical_variance_pass": (
            track == "A" or support.get(name, {}).get("positive_mismatch_noisy_active_channels", 0) > 0
        ),
    }


def synthetic_endpoint_battery(model, step, *, track, device="cuda"):
    """Compare all frozen candidates with original GN and V1 B on one track and step."""
    if step not in STEPS:
        raise ValueError(f"unregistered RK4 step {step!r}")
    model = deepcopy(model).to(device=device)
    if model.mass.dtype != torch.float64:
        raise ValueError("the synthetic battery requires float64")
    truth_model = _truth_model(model, track)
    truth, observed, power, equilibrium, scale = _endpoint_fixture(truth_model, step)
    inputs = _inputs(observed, power, equilibrium, scale)
    settings = _settings(step)
    with PLAN.open("rb") as stream:
        plan = tomllib.load(stream)
    assert plan["synthetic_gate"] == {
        "cases_per_track_step": len(CASES),
        "track_b_truth_stiffness_damping_multiplier": TRACK_B_FACTOR,
        "track_a_minimum_reduction_vs_original_gn": TRACK_A_MINIMUM_REDUCTION,
    }
    common = {key: plan[key] for key in ("gate_rad", "huber_delta")}
    candidates = {
        row["name"]: {key: row[key] for key in ("alpha", "c_var", "kappa_term", "inverse_snr")}
        for row in plan["candidates"]
    }
    assert len(candidates) == 8, "all eight frozen V3 candidates are required"
    runs = [
        ("orig", predict_m3, {"prior_weight": 1e-6}),
        ("v1", predict_mhe, {"alpha": .1, "gate_rad": .0003, "huber_delta": 1.5}),
        *((name, predict_mhe_v3, common | options) for name, options in candidates.items()),
    ]
    with torch.no_grad():
        reference = truth_model.rollout(truth[:, -1], power, equilibrium, 800,
                                        max_step=step, spend=_spend)
    assert reference.shape == (len(CASES), 801, 30) and torch.isfinite(reference).all()
    scores, support = {}, {}
    for name, predictor, options in runs:
        score = {"status": "FAIL", "finite_cases": 0, "aggregate_nrmse": None}
        try:
            result = predictor(model, *inputs, 800, **settings, **options)
            predicted = result["forecast"]
            if name in candidates:
                support[name] = _support(result)
            if predicted.shape == reference.shape:
                score["finite_cases"] = int(torch.isfinite(predicted).all(dim=(1, 2)).sum())
            if score["finite_cases"] != len(CASES):
                score["error"] = "incomplete or nonfinite forecast"
            else:
                score["max_coi_residual"] = float(_coi_residual(predicted, model.mass))
                metrics = _score(predicted, reference)
                if all(np.isfinite(value).all() for value in metrics.values()):
                    score.update(metrics)
                    score["status"] = "PASS" if score["max_coi_residual"] < 1e-12 else "FAIL"
                else:
                    score["error"] = "nonfinite score"
                if name in candidates:
                    check = support[name]
                    if not (check["adaptive_finite"] and check["lm_objective_monotone"]
                            and check["pulse_weight_zero_channels"] == 2 * len(CASES)):
                        score["status"] = "FAIL"
                        score["error"] = "nonfinite adaptation, nonmonotone LM, or nonzero pulse weight"
        except (ArithmeticError, RuntimeError, ValueError) as error:
            score["error"] = f"{type(error).__name__}: {error}"
        scores[name] = score

    def endpoint_means(case_indices):
        return {
            name: float(model.mass.new_tensor(score["target_peak_gap"])[case_indices].mean())
            if "target_peak_gap" in score else None
            for name, score in scores.items()
        }

    endpoints = {str(endpoint): endpoint_means([
        index for index, (_, case_endpoint, _) in enumerate(CASES) if case_endpoint == endpoint
    ]) for endpoint in (0, 10)}
    machines = {str(machine): {
        str(endpoint): endpoint_means([
            index for index, (case_machine, case_endpoint, _) in enumerate(CASES)
            if case_machine == machine and case_endpoint == endpoint
        ]) for endpoint in (0, 10)
    } for machine in range(10)}
    gates = {name: _candidate_gates(track, scores, endpoints, support, name) for name in candidates}
    passing = [
        name for name, gate in gates.items()
        if all(scores[baseline]["status"] == "PASS" for baseline in ("orig", "v1"))
        and gate["numerical_pass"] and gate["empirical_variance_pass"]
        and all(all(comparison.values()) for comparison in gate["comparisons"].values())
    ]
    return {
        "status": "PASS" if passing else "FAIL",
        "track": track,
        "step_s": step,
        "device": str(model.mass.device),
        "device_name": torch.cuda.get_device_name(model.mass.device) if model.mass.is_cuda else "cpu",
        "dtype": str(model.mass.dtype),
        "truth_stiffness_factor": 1. if track == "A" else TRACK_B_FACTOR,
        "truth_damping_factor": 1. if track == "A" else TRACK_B_FACTOR,
        "cases": [{"machine": m, "endpoint": endpoint, "sign": sign} for m, endpoint, sign in CASES],
        "endpoint_target_peak_means": endpoints,
        "per_machine_target_peak_means": machines,
        "models": scores,
        "support": support,
        "gates": gates,
        "passing_candidates": passing,
    }


def run_endpoint_study():
    """Run the four registered cells once on CUDA and emit each completed cell."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    summaries = []
    with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
        if not torch.cuda.is_available():
            raise RuntimeError("the registered V3 comparison requires CUDA float64")
        try:
            torch.use_deterministic_algorithms(True)
            with contextmanager(model.__wrapped__)() as predictor:
                for track in ("A", "B"):
                    for step in STEPS:
                        summary = synthetic_endpoint_battery(predictor, step, track=track)
                        print("V3_PULSE_DIAGNOSTICS " + json.dumps(
                            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
                        ), flush=True)
                        summaries.append(summary)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)
    return summaries


@pytest.mark.parametrize("track", ("A", "B"))
def test_v3_gate_checks_three_metrics_and_b_requires_positive_noisy_support(track):
    scores = {name: {"status": "PASS", "aggregate_nrmse": error}
              for name, error in (("orig", 1.), ("v1", .9), ("A", .9))}
    endpoints = {str(endpoint): {name: score["aggregate_nrmse"] for name, score in scores.items()}
                 for endpoint in (0, 10)}
    gate = _candidate_gates(track, scores, endpoints, {}, "A")
    assert all(gate["comparisons"]["orig"].values())
    assert all(gate["comparisons"]["v1"].values()) == (track == "A")
    assert gate["empirical_variance_pass"] == (track == "A")
    positive = {"A": {"positive_mismatch_noisy_active_channels": 1}}
    assert _candidate_gates(track, scores, endpoints, positive, "A")["empirical_variance_pass"]


if __name__ == "__main__":
    run_endpoint_study()
