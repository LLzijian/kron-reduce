from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models import temporal_risk
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    fit_nonlinear_model,
)
from kron_reduce.models.temporal_risk import temporal_logsumexp_risk
from kron_reduce.physics.coordinates import coi_basis, coi_project


def _synthetic_samples(future_counts=(1, 2)):
    mass = np.array([1.0, 1.2, 1.5])
    basis = coi_basis(np.ones(3))
    system = NonlinearPassiveClosure(
        mass, np.array([1.3, 0.7, 1.1]), 0.2 * basis @ basis.T,
        basis[:, :1], np.zeros((1, 1)), np.array([[0.3]]),
    )
    rng = np.random.default_rng(20261024)
    samples = []
    for index, future_count in enumerate(future_counts):
        equilibrium = coi_project(rng.normal(0.0, 0.1, 3), mass)
        initial_delta = coi_project(rng.normal(0.0, 0.02, 3), mass)
        initial_omega = coi_project(rng.normal(0.0, 0.01, 3), mass)
        path = system.rollout(
            initial_delta, initial_omega, np.array([0.02]),
            np.arange(14, dtype=float) * 0.01, np.zeros(3), equilibrium,
            max_step=0.005,
        )
        history = CausalStateHistory(
            np.arange(11, dtype=float) * 0.01 - 0.1,
            path[:11, :3], path[:11, 3:6], np.zeros(3),
        )
        forecast_time = np.arange(future_count + 1, dtype=float) * 0.01
        future = path[10:11 + future_count]
        samples.append(EquilibriumLearningSample(
            event_id=f"synthetic-{index}", source_dataset="source-a",
            fault_line=f"line-{index}", role="train", history=history,
            forecast_time=forecast_time, target_delta=future[:, :3],
            target_omega=future[:, 3:6], equilibrium_delta=equilibrium,
        ))
    return mass, tuple(samples)


def _settings(epochs=1):
    return NonlinearFitSettings(
        latent_dimension=1, frames=2, epochs=epochs, learning_rate=1e-5,
        encoder_penalty=1e-4, history_weight=0.0, horizons_s=(0.01, 0.02), seed=419,
    )


def _assert_same_array(left, right):
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def test_default_and_explicit_original_contract_are_bitwise_identical():
    mass, samples = _synthetic_samples((2, 2))
    default = fit_nonlinear_model(samples, mass, _settings())
    explicit = fit_nonlinear_model(
        samples, mass, _settings(), training_max_step=0.01,
        temporal_temperature=None, rhs_audit=None,
    )
    for name in ("edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"):
        _assert_same_array(getattr(default.closure, name), getattr(explicit.closure, name))
    _assert_same_array(default.encoder, explicit.encoder)
    _assert_same_array(default.feature_scale, explicit.feature_scale)
    _assert_same_array(default.target_scale, explicit.target_scale)
    assert default.fit_record == explicit.fit_record
    assert "training_max_step" not in default.fit_record


def test_refined_training_uses_two_original_substeps_and_rhs_audit_counts_partial_batches():
    mass, samples = _synthetic_samples((1, 2))
    calls = {"invocations": 0, "event_rhs": 0}
    first_rk4_calls = []

    def rhs_audit(rhs, state, **kwargs):
        result = rhs(state, **kwargs)
        if len(first_rk4_calls) < 8:
            first_rk4_calls.append((state.detach().clone(), result.detach().clone()))
        calls["invocations"] += 1
        calls["event_rhs"] += int(state.shape[0])
        return result

    model = fit_nonlinear_model(
        samples, mass, _settings(epochs=2), allow_partial_horizons=True,
        training_max_step=0.005, temporal_temperature=0.06068411212461594,
        rhs_audit=rhs_audit,
    )
    # Per update: history 10 intervals * 2 substeps * 4 RHS * 2 events = 160;
    # future has N=(1,2): 2 substeps * 4 RHS * (2 active then 1 active) = 24.
    assert calls == {"invocations": 2 * (80 + 16), "event_rhs": 2 * (160 + 24)}
    states = [row[0] for row in first_rk4_calls]
    slopes = [row[1] for row in first_rk4_calls]
    substep = torch.tensor(
        [[np.diff(sample.history.time)[0] / 2] for sample in samples],
        dtype=torch.float64,
    )
    assert torch.equal(states[1], states[0] + substep * slopes[0] / 2)
    assert torch.equal(states[2], states[0] + substep * slopes[1] / 2)
    assert torch.equal(states[3], states[0] + substep * slopes[2])
    rk4_endpoint = states[0] + substep * (
        slopes[0] + 2 * slopes[1] + 2 * slopes[2] + slopes[3]
    ) / 6
    assert torch.equal(states[4], rk4_endpoint)
    record = model.fit_record
    assert record["training_max_step"] == 0.005
    assert record["temporal_temperature"] == 0.06068411212461594
    assert record["temporal_aggregation"] == "within_event_logsumexp"
    assert record["training_event_rk4_steps_per_optimizer_update"] == 10 * 2 * 2 + 3 * 2
    assert record["training_rk4_event_steps"] == 2 * (40 + 6)
    assert record["training_rhs_event_evaluations"] == 4 * record["training_rk4_event_steps"]
    assert record["actual_optimizer_steps"] == 2
    assert record["partial_horizons"] is True
    assert len(record["loss_trace"]) == 2
    for row in record["loss_trace"]:
        assert row["training_max_step"] == 0.005
        assert row["temporal_temperature"] == 0.06068411212461594
        assert row["future_objective"] >= row["future_loss"] - 1e-12
        assert math.isfinite(row["objective"])
        assert all(math.isfinite(value) for value in row["temporal_effective_time_points"])


def test_temporal_risk_changes_only_time_aggregation_and_has_finite_gradients():
    point_mse = torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.2, 0.8]], dtype=torch.float64,
                             requires_grad=True)
    time_weights = torch.tensor([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]], dtype=torch.float64)
    risk = temporal_logsumexp_risk(point_mse, time_weights, 0.3)
    weighted_mean = torch.sum(time_weights * point_mse, dim=1)
    assert torch.all(risk >= weighted_mean - 1e-12)
    risk.sum().backward()
    assert torch.isfinite(point_mse.grad).all()
    assert point_mse.grad[0, 2].item() == 0.0


def test_temporal_risk_matches_independent_weighted_logsumexp_formula():
    # Nonuniform quadrature weights, a fully masked tail, equal losses, and a
    # one-real-point event exercise distinct edges of the fixed-time formula.
    losses_np = np.array([
        [0.15, 0.55, 0.25, 0.91],
        [0.42, 0.42, 0.42, 0.42],
        [0.0, 0.73, 0.0, 0.0],
    ], dtype=np.float64)
    weights_np = np.array([
        [0.2, 0.5, 0.3, 0.0],
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float64)
    tau = 0.17
    losses = torch.tensor(losses_np, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(weights_np, dtype=torch.float64)
    risk, beta = temporal_logsumexp_risk(
        losses, weights, tau, return_time_risk_weights=True,
    )

    expected_risk = []
    expected_beta = np.zeros_like(weights_np)
    for event in range(losses_np.shape[0]):
        active = weights_np[event] > 0
        logits = np.log(weights_np[event, active]) + losses_np[event, active] / tau
        peak = float(np.max(logits))
        exp_shifted = np.exp(logits - peak)
        partition = float(np.sum(exp_shifted))
        expected_risk.append(tau * (peak + math.log(partition)))
        expected_beta[event, active] = exp_shifted / partition
    np.testing.assert_allclose(risk.detach().numpy(), expected_risk, rtol=0, atol=1e-14)
    np.testing.assert_allclose(beta.numpy(), expected_beta, rtol=0, atol=1e-14)

    risk.sum().backward()
    np.testing.assert_allclose(losses.grad.numpy(), expected_beta, rtol=0, atol=1e-14)
    assert torch.equal(losses.grad[0, 3], torch.tensor(0.0, dtype=torch.float64))
    assert torch.equal(losses.grad[2], torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.equal(risk[1], torch.tensor(0.42, dtype=torch.float64))
    torch.testing.assert_close(beta[1], weights[1], rtol=0, atol=1e-16)
    assert torch.equal(risk[2], torch.tensor(0.73, dtype=torch.float64))
    torch.testing.assert_close(beta[2], weights[2], rtol=0, atol=0)
    np.testing.assert_allclose(beta.sum(dim=1).numpy(), np.ones(3), rtol=0, atol=1e-15)

    # Adding a constant to every active loss shifts the risk by that constant
    # and leaves its derivative weights unchanged.
    shift = 2.75
    shifted_losses = torch.tensor(losses_np + shift, dtype=torch.float64)
    shifted_risk, shifted_beta = temporal_logsumexp_risk(
        shifted_losses, weights, tau, return_time_risk_weights=True,
    )
    np.testing.assert_allclose(shifted_risk.numpy(), np.asarray(expected_risk) + shift,
                               rtol=0, atol=2e-14)
    np.testing.assert_allclose(shifted_beta.numpy(), expected_beta, rtol=0, atol=1e-14)

    # For large tau, the log-sum-exp risk approaches the fixed weighted mean.
    # Hoeffding's bounded-range inequality gives R_tau - mean <= range^2/(8*tau).
    mean = (weights_np * losses_np).sum(axis=1)
    high_temperature_risks = []
    for large_tau in (1e3, 1e4, 1e5):
        mean_risk, mean_beta = temporal_logsumexp_risk(
            torch.tensor(losses_np, dtype=torch.float64), weights, large_tau,
            return_time_risk_weights=True,
        )
        deviation = mean_risk.numpy() - mean
        ranges = np.max(np.where(weights_np > 0, losses_np, -np.inf), axis=1) - np.min(
            np.where(weights_np > 0, losses_np, np.inf), axis=1
        )
        bound = ranges**2 / (8 * large_tau)
        rounding_allowance = max(2e-15, 128 * np.finfo(np.float64).eps * large_tau)
        assert np.all(deviation >= -rounding_allowance)
        assert np.all(deviation <= bound + rounding_allowance)
        np.testing.assert_allclose(mean_beta.numpy(), weights_np, rtol=0, atol=1e-4)
        high_temperature_risks.append(deviation)
    assert high_temperature_risks[2][0] < high_temperature_risks[1][0]
    assert high_temperature_risks[1][0] < high_temperature_risks[0][0]

    # The stable independent max-shifted calculation remains finite even when
    # loss/tau is extremely large; the hottest point dominates predictably.
    huge_losses_np = np.array([[1.0e6, 1.0e6 + 100.0, 8.0e5]], dtype=np.float64)
    huge_weights_np = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    huge_tau = 0.06
    huge_logits = np.log(huge_weights_np[0]) + huge_losses_np[0] / huge_tau
    huge_peak = float(np.max(huge_logits))
    huge_exp = np.exp(huge_logits - huge_peak)
    huge_partition = float(huge_exp.sum())
    huge_expected_risk = huge_tau * (huge_peak + math.log(huge_partition))
    huge_expected_beta = huge_exp / huge_partition
    huge_losses = torch.tensor(huge_losses_np, dtype=torch.float64, requires_grad=True)
    huge_risk, huge_beta = temporal_logsumexp_risk(
        huge_losses, torch.tensor(huge_weights_np, dtype=torch.float64), huge_tau,
        return_time_risk_weights=True,
    )
    assert torch.isfinite(huge_risk).all() and torch.isfinite(huge_beta).all()
    assert huge_risk.item() == pytest.approx(huge_expected_risk, abs=1e-10, rel=0)
    np.testing.assert_allclose(huge_beta.detach().numpy()[0], huge_expected_beta,
                               rtol=0, atol=1e-14)
    huge_risk.sum().backward()
    np.testing.assert_allclose(huge_losses.grad.numpy()[0], huge_expected_beta,
                               rtol=0, atol=1e-14)

    # Temperature is deliberately a nondifferentiable fixed Python scalar.
    differentiable_tau = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    with pytest.raises(TypeError, match="fixed Python int or float"):
        temporal_logsumexp_risk(losses.detach(), weights, differentiable_tau)


def test_refined_training_rejects_unregistered_history_substep_and_optimizer_resume():
    mass, samples = _synthetic_samples((2, 2))
    times = np.array([-0.1, -0.096, *np.linspace(-0.0864, 0.0, 10)])
    changed_samples = []
    for sample in samples:
        old = sample.history
        history = CausalStateHistory(
            times, np.vstack((old.delta[0], old.delta)),
            np.vstack((old.omega[0], old.omega)), old.constant_power,
        )
        changed_samples.append(replace(sample, history=history))
    with pytest.raises(ValueError, match="exactly two RK4 substeps"):
        fit_nonlinear_model(tuple(changed_samples), mass, _settings(), training_max_step=0.005)

    with pytest.raises(ValueError, match="fresh Adam"):
        fit_nonlinear_model(samples, mass, _settings(), training_max_step=0.005,
                            full_start={"synthetic": True}, optimizer_start={"synthetic": True})


def test_risk_temperature_modes_are_mutually_exclusive_and_invalid_options_reject():
    mass, samples = _synthetic_samples((2, 2))
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_nonlinear_model(samples, mass, _settings(), risk_temperature=0.1,
                            temporal_temperature=0.1)
    for value in (True, torch.tensor(0.1, dtype=torch.float64), "0.1", 0.0, math.inf):
        with pytest.raises(ValueError, match="temporal_temperature"):
            fit_nonlinear_model(samples, mass, _settings(), temporal_temperature=value)
    with pytest.raises(ValueError, match="training_max_step"):
        fit_nonlinear_model(samples, mass, _settings(), training_max_step=0.02)
    with pytest.raises(TypeError, match="rhs_audit"):
        fit_nonlinear_model(samples, mass, _settings(), rhs_audit=True)


def test_training_option_trace_contract_and_first_objectives_match_analytic(monkeypatch):
    mass, samples = _synthetic_samples((1, 2))
    temperature = 0.3
    initial = fit_nonlinear_model(
        samples, mass, _settings(epochs=2), allow_partial_horizons=True,
    )
    closure = initial.closure
    full_start = {
        "mass": closure.mass.copy(),
        "edge_stiffness": closure.edge_stiffness.copy(),
        "damping": closure.damping.copy(),
        "coupling": closure.coupling.copy(),
        "latent_skew": closure.latent_skew.copy(),
        "latent_dissipation": closure.latent_dissipation.copy(),
        "encoder": initial.encoder.copy(),
        "feature_scale": initial.feature_scale.copy(),
        "target_scale": initial.target_scale.copy(),
    }
    calls = {"mean": {"invocations": 0, "event_rhs": 0},
             "time": {"invocations": 0, "event_rhs": 0}}

    def make_audit(counter):
        def rhs_audit(rhs, state, **kwargs):
            counter["invocations"] += 1
            counter["event_rhs"] += int(state.shape[0])
            return rhs(state, **kwargs)
        return rhs_audit

    captured = []
    original = temporal_risk.temporal_logsumexp_risk

    def capture_time_inputs(point_mse, time_weights, tau, **kwargs):
        captured.append((point_mse.detach().cpu().numpy().copy(),
                         time_weights.detach().cpu().numpy().copy()))
        return original(point_mse, time_weights, tau, **kwargs)

    monkeypatch.setattr(temporal_risk, "temporal_logsumexp_risk", capture_time_inputs)
    snapshots = {"mean": [], "time": []}

    def make_checkpoint_capture(key):
        def capture(step, model, state):
            snapshots[key].append((step, model, {k: np.array(v, copy=True) for k, v in state.items()}))
        return capture

    mean_model = fit_nonlinear_model(
        samples, mass, _settings(epochs=2), allow_partial_horizons=True,
        training_max_step=0.005, temporal_temperature=None,
        rhs_audit=make_audit(calls["mean"]), full_start=full_start,
        checkpoint_steps=(2,), checkpoint_callback=make_checkpoint_capture("mean"),
    )
    time_model = fit_nonlinear_model(
        samples, mass, _settings(epochs=2), allow_partial_horizons=True,
        training_max_step=0.005, temporal_temperature=temperature,
        rhs_audit=make_audit(calls["time"]), full_start=full_start,
        checkpoint_steps=(2,), checkpoint_callback=make_checkpoint_capture("time"),
    )
    mean_trace, time_trace = mean_model.fit_record["loss_trace"], time_model.fit_record["loss_trace"]
    assert len(mean_trace) == len(time_trace) == 2
    assert len(captured) == 2
    for row in (*mean_trace, *time_trace):
        assert row["training_max_step"] == 0.005
        assert "future_objective" in row
    assert mean_trace[0]["temporal_temperature"] is None
    assert mean_trace[0]["future_objective"] == mean_trace[0]["future_loss"]
    assert mean_trace[0]["temporal_aggregation"] == "original_weighted_time_mean"
    assert time_trace[0]["temporal_temperature"] == temperature
    assert time_trace[0]["temporal_aggregation"] == "within_event_logsumexp"

    point_mse, time_weights = captured[0]
    event_weights = np.full(len(samples), 1.0 / len(samples))
    expected_event_means = np.sum(time_weights * point_mse, axis=1)
    expected_mean = float(np.sum(event_weights * expected_event_means))
    event_risks = []
    for losses, weights in zip(point_mse, time_weights, strict=True):
        eligible = weights > 0
        logits = np.log(weights[eligible]) + losses[eligible] / temperature
        peak = float(np.max(logits))
        event_risks.append(temperature * (peak + math.log(float(np.exp(logits - peak).sum()))))
    expected_time = float(np.sum(event_weights * np.asarray(event_risks)))
    assert mean_trace[0]["future_loss"] == pytest.approx(expected_mean, abs=1e-12, rel=0)
    assert time_trace[0]["future_loss"] == pytest.approx(expected_mean, abs=1e-12, rel=0)
    assert mean_trace[0]["future_objective"] == pytest.approx(expected_mean, abs=1e-12, rel=0)
    assert time_trace[0]["future_objective"] == pytest.approx(expected_time, abs=1e-12, rel=0)
    assert expected_time >= expected_mean - 1e-12
    assert calls["mean"] == calls["time"] == {"invocations": 192, "event_rhs": 368}

    for mode, model in (("mean", mean_model), ("time", time_model)):
        assert len(snapshots[mode]) == 1
        step, snapshot, state = snapshots[mode][0]
        assert step == 2
        record = model.fit_record
        assert record["initial_optimizer_step"] == 0
        assert record["initial_optimizer_moments_zero"] is True
        assert record["actual_optimizer_steps"] == record["total_optimizer_steps"] == 2
        assert snapshot.fit_record["actual_optimizer_steps"] == 2
        assert set(record["initial_parameter_sha256"]) == {"raw", "port", "skew", "decay", "encoder"}
        assert record["initial_parameter_sha256"] == snapshot.fit_record["initial_parameter_sha256"]
        assert state["completed_optimizer_steps"].item() == 2
        assert state["initial_optimizer_step"].item() == 0
        assert state["total_optimizer_steps"].item() == 2
        assert sum(np.asarray(v).size for v in record["initial_parameters"].values()) == record["parameter_count"]
        for name in ("raw", "port", "skew", "decay", "encoder"):
            assert np.all(state[f"adam_step__{name}"] == 2)
            assert np.all(np.isfinite(state[f"parameter__{name}"]))
            assert np.all(np.isfinite(state[f"adam_exp_avg__{name}"]))
            assert np.all(np.isfinite(state[f"adam_exp_avg_sq__{name}"]))
        callback_row = snapshot.fit_record["loss_trace"][-1]
        assert callback_row["epoch"] == 2
        assert callback_row == record["loss_trace"][-1]
        assert callback_row["future_objective"] == record["loss_trace"][-1]["future_objective"]
