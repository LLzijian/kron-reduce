from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.evaluation.bounded_forecast import forecast_bounded
from kron_reduce.evaluation.event_rhs_budget import EventRhsBudget, EventRhsBudgetExceeded
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    LearnedNonlinearModel,
    NonlinearFitSettings,
)
from kron_reduce.physics.coordinates import coi_project
from kron_reduce.workflows.kiki_short_history import synthetic_truth


def _fixture(cohort=7):
    rng = np.random.default_rng(882)
    closure = synthetic_truth(backbone="sine")
    settings = NonlinearFitSettings(latent_dimension=2, frames=3, epochs=1, seed=324)
    n_features = 80
    model = LearnedNonlinearModel(
        closure=closure,
        encoder=rng.normal(0, 0.002, size=(2, n_features)),
        feature_scale=np.ones(n_features),
        target_scale=np.ones(20),
        settings=settings,
        fit_record={},
    )
    samples = []
    time = np.linspace(-0.1, 0.0, 11, dtype=np.float64)
    clocks = (
        np.array([0.0, 0.01, 0.02]),
        np.array([0.0, 0.015, 0.03]),
        np.array([0.0, 0.01, 0.03, 0.05]),
        np.arange(6) * 0.01,
        np.arange(4) * 0.015,
        np.array([0.0, 0.005, 0.015, 0.025, 0.035]),
        np.arange(8) * 0.01,
    )
    for i in range(cohort):
        mass = closure.mass
        power = rng.normal(0.0, 0.002, 10)
        power -= power.sum() / 10
        eq = coi_project(rng.normal(0, 0.02, 10), mass)
        delta = np.stack([coi_project(rng.normal(0, 0.02, 10), mass) for _ in range(11)])
        omega = np.stack([coi_project(rng.normal(0, 0.01, 10), mass) for _ in range(11)])
        history = CausalStateHistory(time, delta, omega, power)
        clock = clocks[i % len(clocks)]
        target_delta = np.repeat(delta[-1][None, :], len(clock), axis=0)
        target_omega = np.repeat(omega[-1][None, :], len(clock), axis=0)
        samples.append(EquilibriumLearningSample(
            f"synthetic-{i}", "fixture", f"line-{i % 2}", "train", history,
            clock, target_delta, target_omega, eq,
        ))
    return model, tuple(samples)


def _budget(samples, max_step, horizon_s=None):
    count = len(samples)
    m = int(np.ceil(0.01 / max_step - 1e-12))
    history = count * 10 * m * 4
    clocks = [np.arange(round(horizon_s / 0.01) + 1) * 0.01 for _ in samples] if horizon_s else [s.forecast_time for s in samples]
    future = sum(sum(int(np.ceil(float(dt) / max_step - 1e-12)) for dt in np.diff(t)) * 4 for t in clocks)
    total = history + future
    return EventRhsBudget(integration_limit=total, physical_limit=0, total_limit=total,
                          rollout_limit=2*count, prediction_path_limit=count,
                          evaluation_integration_limit=total)


@pytest.mark.parametrize("max_step", [0.005, 0.0025])
def test_ragged_batched_forecast_matches_scalar_initialize_and_rollout(max_step):
    model, samples = _fixture()
    before = [(s.history.delta.copy(), s.history.omega.copy(), s.forecast_time.copy()) for s in samples]
    budget = _budget(samples, max_step)
    histories, futures = forecast_bounded(model, samples, max_step=max_step,
                                           budget=budget, identity={"case": "synthetic-ragged"})
    assert histories.shape == (7, 11, 22)
    assert [len(p) for p in futures] == [len(s.forecast_time) for s in samples]
    for i, sample in enumerate(samples):
        _z, expected_history = model.initialize(sample.history, sample.equilibrium_delta,
                                                max_step=max_step)
        expected_future = model.predict(sample.history, sample.forecast_time,
                                        sample.equilibrium_delta, max_step=max_step)
        np.testing.assert_allclose(histories[i], expected_history, rtol=0, atol=1e-10)
        np.testing.assert_allclose(futures[i], expected_future, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(futures[i][0, :20],
                                      np.r_[sample.history.delta[-1], sample.history.omega[-1]])
        assert futures[i].shape[1] == 22
    for old, sample in zip(before, samples, strict=True):
        np.testing.assert_array_equal(old[0], sample.history.delta)
        np.testing.assert_array_equal(old[1], sample.history.omega)
        np.testing.assert_array_equal(old[2], sample.forecast_time)
    snap = budget.snapshot()
    assert snap["attempted"] == snap["returned"]
    assert snap["blocked"] == 0
    assert snap["rollouts_started"] == 14
    assert snap["prediction_paths"] == 7
    assert snap["integration"]["callback_calls"] < snap["integration"]["attempted"]
    assert sum(s["attempted"] for s in snap["segments"]) == snap["attempted"]
    assert all(s["successful_rk4_updates"] == s["returned"] // 4 for s in snap["segments"])


def test_fixed_three_second_horizon_runs_full_clock_even_for_short_samples():
    model, samples = _fixture(cohort=3)
    budget = _budget(samples, 0.005, horizon_s=3.0)
    histories, futures = forecast_bounded(model, samples, max_step=0.005,
                                           horizon_s=3.0, budget=budget,
                                           identity={"case": "synthetic-fixed"})
    assert histories.shape == (3, 11, 22)
    assert [path.shape for path in futures] == [(301, 22)] * 3
    snap = budget.snapshot()
    assert snap["prediction_paths"] == 3
    assert snap["attempted"] == 7440
    assert snap["returned"] == 7440
    assert snap["rollouts_started"] == 6


def test_available_forecast_refuses_more_than_real_300_future_points():
    model, samples = _fixture(cohort=1)
    sample = samples[0]
    time = np.arange(302) * 0.01
    target = np.repeat(sample.history.delta[-1][None, :], len(time), axis=0)
    too_long = EquilibriumLearningSample(sample.event_id, sample.source_dataset, sample.fault_line,
        sample.role, sample.history, time, target,
        np.repeat(sample.history.omega[-1][None, :], len(time), axis=0), sample.equilibrium_delta)
    budget = _budget(samples, 0.005)
    with pytest.raises(ValueError, match="real 1..300"):
        forecast_bounded(model, [too_long], max_step=0.005, budget=budget, identity={})
    assert budget.snapshot()["attempted"] == 0


def test_history_failure_still_consumes_forecast_path_slots_before_rhs():
    model, samples = _fixture(cohort=1)
    budget = EventRhsBudget(integration_limit=0, physical_limit=0, total_limit=0,
                            rollout_limit=1, prediction_path_limit=1,
                            evaluation_integration_limit=0)
    with pytest.raises(EventRhsBudgetExceeded, match="event-RHS partition budget exhausted"):
        forecast_bounded(model, samples, max_step=0.005, budget=budget,
                         identity={"case": "synthetic-history-failure"})
    snapshot = budget.snapshot()
    assert snapshot["prediction_paths"] == 1
    assert snapshot["rollouts_started"] == 1
    assert snapshot["attempted"] == 0
    assert snapshot["segments"][0]["phase"] == "history_integration"
    assert snapshot["segments"][0]["prediction_paths"] == 1
    assert snapshot["segments"][0]["status"] == "failed"


@pytest.mark.parametrize("step", [0.01, 0.003, 0])
def test_bounded_forecast_accepts_only_registered_refinement_steps(step):
    model, samples = _fixture(cohort=1)
    budget = _budget(samples, 0.005)
    with pytest.raises(ValueError, match="5 ms or 2.5 ms"):
        forecast_bounded(model, samples[:1], max_step=step, budget=budget, identity={})

