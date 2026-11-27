"""Numerical invariants and paired zero-mismatch endpoint comparison for ES-ROM V4."""

import json
import os
import tomllib
from contextlib import contextmanager
from copy import deepcopy
from inspect import signature
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_noise_robustness_mhe_v2 import _assert_energy_identity, _inputs, _spend, _truth, model
from test_noise_robustness_mhe_v3 import (
    CASES,
    STEPS,
    _assert_finite_adaptation,
    _check_tangent,
    _coi_residual,
    _endpoint_fixture,
    _pulse_weights,
    _score,
    _settings,
    _support,
)

from kron_reduce.models.noise_robustness_mhe import predict_mhe
from kron_reduce.models.noise_robustness_mhe_v2 import kinematic_cleaner
from kron_reduce.models.noise_robustness_mhe_v4 import _pooled_noise, predict_mhe_v4, reblend_origin
from kron_reduce.models.unified_discovery_m3 import predict_m3

PLAN = Path(__file__).resolve().parents[1] / "configs/experiments/noise_robustness_esrom_v4.toml"
MINIMUM_REDUCTION = .80
MAXIMUM_V1_RATIO = 1.02


@pytest.mark.parametrize("step", STEPS)
def test_v4_joint_history_tangent_matches_all_28_finite_differences(model, step):
    _check_tangent(model, step)


@pytest.mark.parametrize("step", STEPS)
def test_v4_clean_causal_fallback_restores_noiseless_truth(model, step):
    truth, power, equilibrium, scale = _truth(model, step=step, memory=False)
    inputs = _inputs(truth, power, equilibrium, scale)
    settings = _settings(step)
    original = predict_m3(model, *inputs, 3, **settings, prior_weight=1e-6)
    result = predict_mhe_v4(model, *inputs, 3, **settings)
    with torch.no_grad():
        reference = model.rollout(truth[:, -1], power, equilibrium, 3, max_step=step, spend=_spend)
    assert not result["joint_active"].item()
    for key in ("history", "forecast", "hidden_start"):
        torch.testing.assert_close(result[key], original[key], atol=1e-10, rtol=0.)
    assert (result["history"] - truth).abs().max() < 1e-10
    assert (result["forecast"] - reference).abs().max() < 1e-10
    _assert_finite_adaptation(result)


@pytest.mark.parametrize("step", STEPS)
def test_v4_exact_zero_weights_for_all_actual_pulsed_channels(model, step):
    _, observed, _, _, scale = _endpoint_fixture(model, step)
    weights = kinematic_cleaner(observed, scale)["weight"]
    pulsed = _pulse_weights({"outlier_weight": weights}, CASES)
    assert pulsed.shape == (40, 2)
    assert torch.count_nonzero(pulsed) == 0


@pytest.mark.parametrize("step", STEPS)
def test_v4_noisy_eight_second_forecast_preserves_coi_and_energy_identity(model, step):
    _, observed, power, equilibrium, scale = _endpoint_fixture(model, step)
    indices = [12, 14]  # Positive pulses at the first and last frames on machine 3.
    inputs = _inputs(observed[indices], power[indices], equilibrium[indices], scale)
    result = predict_mhe_v4(model, *inputs, 800, **_settings(step), c_var=2.5, kappa_term=2.)
    assert result["joint_active"].all()
    assert (_pulse_weights(result, [CASES[index] for index in indices]) == 0.).all()
    _assert_finite_adaptation(result)
    raw = result["channel_noise"].reshape(2, 2, 10)
    group_floor = .7 * raw.sort(dim=2).values[:, :, 4:5]
    pooled = torch.maximum(raw, group_floor).reshape(2, 20)
    torch.testing.assert_close(result["pooled_noise"], pooled, atol=0., rtol=0.)
    weights = result["outlier_weight"]
    residual = (result["smooth"] - result["history"][:, :, :20]) / scale
    discrepancy = (weights * residual.square()).sum(1) / weights.sum(1).clamp_min(1e-6)
    variance = (discrepancy - 2.5 * pooled.square()).clamp_min(0.)
    torch.testing.assert_close(result["mismatch_variance"], variance, atol=1e-14, rtol=0.)
    gain = variance / (variance + 2. * pooled.square() + 1e-12)
    torch.testing.assert_close(result["endpoint_gain"], gain, atol=1e-14, rtol=0.)
    path = result["forecast"]
    assert path.shape == (2, 801, 30)
    assert torch.isfinite(path).all()
    assert _coi_residual(path, model.mass) < 1e-12
    for row, index in enumerate(indices):
        _assert_energy_identity(model, path[row:row + 1], power[index:index + 1],
                                equilibrium[index:index + 1])


def test_v4_pools_per_event_by_lower_median_within_angle_and_speed_groups():
    angles = torch.tensor([8., 2., 7., 1., 0., 9., 4., 3., 6., 5.], dtype=torch.float64)
    speeds = angles * 10.
    empty_group = angles.new_tensor([17., 0., 14., 0., 10., 0., 12., 0., 0., 0.])
    raw = torch.stack((torch.cat((angles, speeds)), torch.cat((empty_group, angles))))
    floors = raw.new_tensor([[.7 * 4., .7 * 40.], [0., .7 * 4.]]).repeat_interleave(10, dim=1)
    torch.testing.assert_close(_pooled_noise(raw), torch.maximum(raw, floors), atol=0., rtol=0.)


@pytest.mark.parametrize("c_var", (None, float("inf")))
def test_v4_off_keeps_pure_physical_origin_even_with_zero_noise_channels(model, c_var):
    step = STEPS[0]
    truth, power, equilibrium, scale = _truth(model, step=step)
    observed = torch.zeros_like(truth[:, :, :20])
    observed[:, :, 0] = truth[:, :, 0]
    observed[:, :, 1] = -observed[:, :, 0] * model.mass[0] / model.mass[1]
    result = predict_mhe_v4(model, *_inputs(observed, power, equilibrium, scale), 1,
                            **_settings(step), gate_rad=0., c_var=c_var)
    assert result["joint_active"].item()
    assert torch.count_nonzero(result["channel_noise"][:, 2:]) == 0
    assert torch.count_nonzero(result["pooled_noise"]) < 20
    assert torch.count_nonzero(result["mismatch_variance"]) == 0
    assert torch.count_nonzero(result["endpoint_gain"]) == 0
    torch.testing.assert_close(result["origin"], result["history"][:, -1], atol=0., rtol=0.)
    torch.testing.assert_close(reblend_origin(result, model, 2.), result["origin"], atol=0., rtol=0.)
    cutoff = 1.5 * result["kinematic_scale"][:, None]
    torch.testing.assert_close(result["lm_cutoffs"], cutoff.expand_as(result["lm_cutoffs"]),
                               atol=0., rtol=0.)
    _assert_finite_adaptation(result)


def test_v4_initialization_uses_only_its_own_eleven_visible_frames_and_held_input(model):
    step = STEPS[0]
    _, observed, power, equilibrium, scale = _endpoint_fixture(model, step)
    settings = _settings(step)
    single = predict_mhe_v4(
        model, *_inputs(observed[:1], power[:1], equilibrium[:1], scale), 0, **settings,
    )
    # An unrelated event has different noise and held input in the same batch.
    selected = [0, 39]
    batch_power = power[selected].clone()
    batch_power[1] *= 2.
    batch = predict_mhe_v4(
        model, *_inputs(observed[selected], batch_power, equilibrium[selected], scale), 2, **settings,
    )
    assert batch["joint_active"].all()
    for key in ("history", "origin", "hidden_start", "outlier_weight", "pooled_noise",
                "mismatch_variance", "endpoint_gain"):
        torch.testing.assert_close(single[key], batch[key][:1], atol=1e-12, rtol=0.)
    assert tuple(signature(predict_mhe_v4).parameters)[:7] == (
        "model", "q", "v", "power", "equilibrium", "scale", "count",
    )
    extra_frame = torch.cat((observed[:1], observed[:1, -1:]), dim=1)
    with pytest.raises(ValueError, match="eleven"):
        predict_mhe_v4(model, *_inputs(extra_frame, power[:1], equilibrium[:1], scale), 0,
                        **settings)


def _candidate_gates(scores, endpoints, name):
    candidate_scores = {
        "first_endpoint_peak": endpoints["0"],
        "last_endpoint_peak": endpoints["10"],
        "all_case_nrmse": {model: score["aggregate_nrmse"] for model, score in scores.items()},
    }
    limits = {"orig": 1. - MINIMUM_REDUCTION, "v1": MAXIMUM_V1_RATIO}
    return {
        "comparisons": {
            baseline: {
                metric: values[name] is not None and values[baseline] is not None
                and values[name] <= ratio * values[baseline]
                for metric, values in candidate_scores.items()
            }
            for baseline, ratio in limits.items()
        },
        "numerical_pass": scores[name]["status"] == "PASS",
    }


def test_v4_gate_applies_both_inclusive_bounds_to_all_three_metrics():
    original = 1. - MINIMUM_REDUCTION
    v1_bound = .125 * MAXIMUM_V1_RATIO
    scores = {name: {"status": "PASS", "aggregate_nrmse": error}
              for name, error in (("orig", 1.), ("v1", .125), ("A", v1_bound))}
    endpoints = {
        "0": {"orig": 1., "v1": 1., "A": original},
        "10": {"orig": 1., "v1": .125, "A": v1_bound},
    }
    gate = _candidate_gates(scores, endpoints, "A")
    assert gate["numerical_pass"]
    assert all(all(comparison.values()) for comparison in gate["comparisons"].values())
    endpoints["0"]["A"] = np.nextafter(original, np.inf)
    endpoints["10"]["A"] = np.nextafter(v1_bound, np.inf)
    scores["A"]["aggregate_nrmse"] = None
    gate = _candidate_gates(scores, endpoints, "A")["comparisons"]
    assert gate["orig"] == {
        "first_endpoint_peak": False, "last_endpoint_peak": True, "all_case_nrmse": False,
    }
    assert gate["v1"] == {
        "first_endpoint_peak": True, "last_endpoint_peak": False, "all_case_nrmse": False,
    }


def _v4_support(result):
    support = _support(result)
    pooled = result["pooled_noise"]
    variance = result["mismatch_variance"]
    positive = (variance > 0.) & result["joint_active"][:, None]
    pulse_zero = _pulse_weights(result, CASES) == 0.
    support.update(
        adaptive_finite=support["adaptive_finite"] and bool((torch.isfinite(pooled) & (pooled >= 0.)).all()),
        positive_mismatch_events=int(positive.any(dim=1).sum()),
        positive_mismatch_channels=int(positive.sum()),
        positive_mismatch_channels_per_case=positive.sum(dim=1).tolist(),
        pulse_weight_zero_per_case=pulse_zero.tolist(),
    )
    return support


def synthetic_endpoint_battery(model, step):
    """Compare eight V4 candidates and both baselines on the frozen 40-case fixture."""
    if step not in STEPS:
        raise ValueError(f"unregistered RK4 step {step!r}")
    model = deepcopy(model).to(device="cuda:0")
    if model.mass.dtype != torch.float64:
        raise ValueError("the synthetic battery requires float64")
    with PLAN.open("rb") as stream:
        plan = tomllib.load(stream)
    assert plan["synthetic_gate"] == {
        "cases_per_step": len(CASES),
        "minimum_reduction_vs_original_gn": MINIMUM_REDUCTION,
        "maximum_ratio_vs_v1_b": MAXIMUM_V1_RATIO,
    }
    common = {key: plan[key] for key in ("gate_rad", "huber_delta")}
    candidates = {
        row["name"]: {key: row[key] for key in ("alpha", "c_var", "kappa_term", "inverse_snr")}
        for row in plan["candidates"]
    }
    assert len(plan["candidates"]) == 8 and list(candidates) == list("ABCDEFGH")
    for options in candidates.values():
        if options["c_var"] == "off":
            options["c_var"] = None
    runs = [
        ("orig", predict_m3, {"prior_weight": 1e-6}),
        ("v1", predict_mhe, {"alpha": .1, "gate_rad": .0003, "huber_delta": 1.5}),
        *((name, predict_mhe_v4, common | options) for name, options in candidates.items()),
    ]
    truth, observed, power, equilibrium, scale = _endpoint_fixture(model, step)
    inputs = _inputs(observed, power, equilibrium, scale)
    settings = _settings(step)
    with torch.no_grad():
        reference = model.rollout(truth[:, -1], power, equilibrium, 800,
                                  max_step=step, spend=_spend)
    assert reference.shape == (len(CASES), 801, 30) and torch.isfinite(reference).all()
    scores, support = {}, {}
    for name, predictor, options in runs:
        score = {
            "status": "FAIL", "finite_cases": 0, "aggregate_nrmse": None,
            "finite_forecast_per_case": [False] * len(CASES),
        }
        try:
            result = predictor(model, *inputs, 800, **settings, **options)
            predicted = result["forecast"]
            if name in candidates:
                support[name] = _v4_support(result)
            if predicted.shape == reference.shape:
                finite = torch.isfinite(predicted).all(dim=(1, 2))
                score["finite_cases"] = int(finite.sum())
                score["finite_forecast_per_case"] = finite.tolist()
            if score["finite_cases"] != len(CASES):
                score["error"] = "incomplete or nonfinite forecast"
            else:
                score["max_coi_residual"] = float(_coi_residual(predicted, model.mass))
                metrics = _score(predicted, reference)
                if all(np.isfinite(value).all() for value in metrics.values()):
                    score.update(metrics)
                    if score["max_coi_residual"] < 1e-12:
                        score["status"] = "PASS"
                    else:
                        score["error"] = "forecast COI residual"
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
    gates = {name: _candidate_gates(scores, endpoints, name) for name in candidates}
    passing = [
        name for name, gate in gates.items()
        if all(scores[baseline]["status"] == "PASS" for baseline in ("orig", "v1"))
        and gate["numerical_pass"]
        and all(all(comparison.values()) for comparison in gate["comparisons"].values())
    ]
    return {
        "status": "PASS" if passing else "FAIL",
        "track": "A",
        "step_s": step,
        "device": str(model.mass.device),
        "device_name": torch.cuda.get_device_name(model.mass.device),
        "dtype": str(model.mass.dtype),
        "truth_stiffness_factor": 1.,
        "truth_damping_factor": 1.,
        "cases": [{"machine": m, "endpoint": endpoint, "sign": sign} for m, endpoint, sign in CASES],
        "endpoint_target_peak_means": endpoints,
        "per_machine_target_peak_means": machines,
        "models": scores,
        "support": support,
        "gates": gates,
        "passing_candidates": passing,
    }


def run_endpoint_study():
    """Emit both registered CUDA cells so the workflow can compare common candidates."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4060 Ti":
            raise RuntimeError("the registered V4 comparison requires CUDA on NVIDIA GeForce RTX 4060 Ti")
        try:
            torch.use_deterministic_algorithms(True)
            with contextmanager(model.__wrapped__)() as predictor:
                for step in STEPS:
                    summary = synthetic_endpoint_battery(predictor, step)
                    print("V4_PULSE_DIAGNOSTICS " + json.dumps(
                        summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
                    ), flush=True)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)


if __name__ == "__main__":
    run_endpoint_study()
