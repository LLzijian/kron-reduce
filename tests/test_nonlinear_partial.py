from dataclasses import replace

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.learned_latent import event_weights
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    _integral_start,
    fit_nonlinear_model,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project


def examples(future_counts=(50, 50, 50, 50)):
    mass = np.array([1.0, 1.2, 1.5])
    basis = coi_basis(np.ones(3))
    system = NonlinearPassiveClosure(
        mass,
        np.array([1.3, 0.7, 1.1]),
        0.2 * basis @ basis.T,
        basis[:, :1],
        np.zeros((1, 1)),
        np.ones((1, 1)) * 0.3,
    )
    rng = np.random.default_rng(324)
    samples = []
    for i, future_count in enumerate(future_counts):
        eq = coi_project(rng.normal(0, 0.2, 3), mass)
        q = coi_project(rng.normal(0, 0.3, 3), mass)
        v = coi_project(rng.normal(0, 0.1, 3), mass)
        path = system.rollout(q, v, rng.normal(size=1), np.arange(71) * 0.01, np.zeros(3), eq)
        history = CausalStateHistory(
            np.arange(21) * 0.01 - 0.2, path[:21, :3], path[:21, 3:6], np.zeros(3)
        )
        samples.append(
            EquilibriumLearningSample(
                f"e{i}",
                "A",
                f"L{i}",
                "train",
                history,
                np.arange(future_count + 1) * 0.01,
                path[20 : 21 + future_count, :3],
                path[20 : 21 + future_count, 3:6],
                eq,
            )
        )
    return mass, tuple(samples)


def settings():
    return NonlinearFitSettings(
        latent_dimension=1, epochs=1, learning_rate=1e-15, horizons_s=(0.2, 0.5)
    )


def test_partial_fit_accepts_variable_real_target_counts_and_records_masks():
    mass, samples = examples((1, 7, 20, 50))
    with pytest.raises(ValueError, match="one clock"):
        fit_nonlinear_model(samples, mass, settings())
    model = fit_nonlinear_model(samples, mass, settings(), allow_partial_horizons=True)
    assert model.fit_record["partial_horizons"] is True
    assert model.fit_record["loss_masks_are_features"] is False
    assert model.fit_record["real_target_counts"] == {"e0": 1, "e1": 7, "e2": 20, "e3": 50}
    assert model.fit_record["missing_target_weight"] == 0.0


def test_partial_objective_replays_visible_history_and_each_event_horizons():
    mass, samples = examples((1, 7, 20, 50))
    config = settings()
    model = fit_nonlinear_model(samples, mass, config, allow_partial_horizons=True)
    weights = event_weights(samples)
    value = 0.0
    for weight, sample in zip(weights, samples, strict=True):
        _, hp = model.initialize(sample.history, sample.equilibrium_delta)
        history_error = np.mean(
            (
                (hp[1:, :6] - np.c_[sample.history.delta[1:], sample.history.omega[1:]])
                / model.target_scale
            )
            ** 2
        )
        pred = model.predict(sample.history, sample.forecast_time, sample.equilibrium_delta)
        per_horizon = []
        for horizon in config.horizons_s:
            count = min(round(horizon / 0.01), len(sample.forecast_time) - 1)
            err = (
                (
                    pred[1 : count + 1, :6]
                    - np.c_[sample.target_delta[1 : count + 1], sample.target_omega[1 : count + 1]]
                )
                / model.target_scale
            ) ** 2
            per_horizon.append(np.mean(err))
        value += weight * (config.history_weight * history_error + np.mean(per_horizon))
    value += config.encoder_penalty * np.mean(model.encoder**2)
    assert abs(value - model.fit_record["loss_trace"][0]["objective"]) <= 1e-9


def test_full_clock_partial_mode_matches_default_predictions_and_fields():
    mass, samples = examples()
    ordinary = fit_nonlinear_model(samples, mass, settings())
    partial = fit_nonlinear_model(samples, mass, settings(), allow_partial_horizons=True)
    assert np.array_equal(ordinary.encoder, partial.encoder)
    assert np.array_equal(ordinary.feature_scale, partial.feature_scale)
    assert np.array_equal(ordinary.target_scale, partial.target_scale)
    for sample in samples:
        p0 = ordinary.predict(sample.history, sample.forecast_time, sample.equilibrium_delta)
        p1 = partial.predict(sample.history, sample.forecast_time, sample.equilibrium_delta)
        assert np.allclose(p0, p1, rtol=0, atol=1e-10)
    assert ordinary.fit_record["loss_trace"] == partial.fit_record["loss_trace"]


def test_targets_and_declared_lengths_do_not_change_causal_encoder_features():
    mass, samples = examples((7, 50, 20, 1))
    model = fit_nonlinear_model(samples, mass, settings(), allow_partial_horizons=True)
    sample = samples[0]
    changed_delta = sample.target_delta.copy()
    changed_omega = sample.target_omega.copy()
    changed_delta[1:] += 100
    changed_omega[1:] -= 200
    changed = replace(sample, target_delta=changed_delta, target_omega=changed_omega)
    p0 = model.predict(sample.history, sample.forecast_time, sample.equilibrium_delta)
    p1 = model.predict(changed.history, changed.forecast_time, changed.equilibrium_delta)
    assert np.array_equal(p0, p1)
    assert model.fit_record["loss_masks_are_features"] is False


def test_integral_start_includes_one_point_partial_event_and_is_finite():
    mass, samples = examples((1, 50, 50, 50))
    scale = np.ones(6)
    start = _integral_start(samples[:1], mass, "sine", scale, include_last=True)
    assert np.isfinite(start).all()
    assert len(start) == 5
