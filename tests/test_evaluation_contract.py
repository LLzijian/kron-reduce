from __future__ import annotations

import math

import numpy as np

from kron_reduce.evaluation.cohorts import prospective_eligibility
from kron_reduce.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kron_reduce.evaluation.status import evaluation_status
from kron_reduce.evaluation.stability import classification_metrics


def test_forecast_metrics_exclude_observed_prefix():
    truth = np.arange(12, dtype=float).reshape(6, 2)
    prediction = truth.copy(); prediction[3:] += 100
    metric = trajectory_metrics(truth, truth, prediction, prediction, metric_start_index=3)
    assert metric["rmse"] == 100


def test_empty_or_failed_population_is_not_completed():
    assert evaluation_status(0, 0, {}) == "ineligible_by_observation_window"
    assert evaluation_status(2, 0, {}) == "attempted_failed_with_evidence"
    assert evaluation_status(2, 2, {"rmse": math.nan}) == "attempted_failed_with_evidence"


def test_prospective_event_window():
    assert prospective_eligibility({"event_observed": True, "event_time_s": 0.1}, 0.2, True)[0] is False
    assert prospective_eligibility({"event_observed": True, "event_time_s": 0.3}, 0.2, True)[0] is True


def test_missing_class_metric_is_nan_not_zero():
    metrics = classification_metrics(["stable"], ["stable"])
    assert math.isnan(metrics["FSR"])


def test_omega_r2_is_reported_and_pooled_from_sse():
    truth = np.arange(12, dtype=float).reshape(6, 2)
    prediction = truth + 0.1
    row = trajectory_metrics(truth, truth, prediction, prediction)
    aggregate = aggregate_trajectory_metrics([row, row])
    assert row["omega_r2"] > 0.99
    assert aggregate["pooled_omega_r2"] == row["omega_r2"]
