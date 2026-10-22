import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import (
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
)
from kron_reduce.workflows.kiki_stability_comparison import (
    CLOCK,
    confusion_metrics,
    forecast,
    los_classification,
    numerical_qualification,
    summarize,
)
from project_harness.evidence import SCIENTIFIC_VERDICTS


def history():
    return CausalStateHistory(
        np.arange(21) * 0.01 - 0.2,
        np.zeros((21, 10)), np.zeros((21, 10)), np.zeros(10),
    )


def classify(path, observed=None, equilibrium=None):
    return los_classification(
        history() if observed is None else observed, path,
        np.zeros(10) if equilibrium is None else equilibrium,
        np.ones(10), 2 * np.pi * 60,
    )


def exceeded_path(start, length):
    path = np.zeros((801, 20))
    path[start : start + length, 0] = 4
    path[start : start + length, 10] = 10
    return path


def test_los_threshold_duration_and_record_clock():
    assert not classify(exceeded_path(1, 4))["predicted_los"]
    c = classify(exceeded_path(1, 5))
    assert c["predicted_los"] and c["trigger_index"] == 25
    assert c["predicted_los_time_record_s"] == 0.25
    assert c["predicted_los_time_forecast_s"] == pytest.approx(0.05)
    assert not c["window_settled"]
    path = exceeded_path(1, 5)
    path[:, 10:] = 0
    assert not classify(path)["predicted_los"]  # Both conditions required.


def test_equilibrium_restored_and_angles_not_wrapped():
    path = exceeded_path(1, 5)
    path[:, 0] -= 1
    assert not classify(path)["predicted_los"]
    eq = np.r_[1.0, np.zeros(9)]
    assert classify(path, equilibrium=eq)["predicted_los"]
    path[1:6, 0] += 4 * np.pi
    assert classify(path)["predicted_los"]


def test_observed_history_spliced_once_without_decoder_origin():
    h = history()
    q, v = h.delta.copy(), h.omega.copy()
    q[-4:, 0], v[-4:, 0] = 4, 10
    h = CausalStateHistory(h.time, q, v, h.constant_power)
    path = exceeded_path(1, 1)
    path[0] = 1e3  # Origin reconstruction must not replace actual observations.
    assert classify(path, h)["trigger_index"] == 21
    path[1] = 0
    assert not classify(path, h)["predicted_los"]


def test_fixed_eight_seconds_and_nonfinite_tail_not_stable():
    assert CLOCK[-1] == 8 and len(CLOCK) == 801
    path = np.zeros((801, 20))
    assert classify(path)["window_settled"]
    path[-1, 1] = np.nan
    c = classify(path)
    assert not c["finite_full_path"] and not c["window_settled"]
    with pytest.raises(ValueError):
        classify(path[:30])


def test_common_frequency_does_not_trigger_relative_speed():
    path = np.zeros((801, 20))
    path[1:6, 0] = 4
    path[1:6, 10:] = 100
    assert not classify(path)["predicted_los"]


def test_numerical_gate_uses_detection_endpoint_or_full_no_los_window():
    fine = exceeded_path(1, 5)
    coarse = fine.copy()
    coarse[10:, 0] = 1e8
    a, b = classify(fine), classify(coarse)
    assert numerical_qualification(fine, coarse, a, b)["qualified"]
    fine[:] = 0
    coarse[:] = 0
    coarse[-1, 0] = 10
    assert not numerical_qualification(fine, coarse, classify(fine), classify(coarse))["qualified"]
    coarse = exceeded_path(10, 5)
    fine = exceeded_path(1, 5)
    assert not numerical_qualification(fine, coarse, classify(fine), classify(coarse))["qualified"]


def test_unresolved_retained_and_no_fake_confusion_accuracy():
    metrics = confusion_metrics(2, 1, 3, 4, 1, 2)
    assert metrics["eligible"] == 13
    assert metrics["fsr"] is None and metrics["accuracy"] is None
    assert metrics["fsr_lower"] == 0.25 and metrics["fsr_upper"] == 0.5
    assert metrics["fur_lower"] == 1 / 3 and metrics["fur_upper"] == 5 / 9
    full = confusion_metrics(8, 4, 1, 94)
    assert full["fsr"] == 1 / 3 and full["fur"] == 1 / 95
    assert full["accuracy"] == 102 / 107
    assert full["balanced_accuracy"] == pytest.approx((8 / 12 + 94 / 95) / 2)


def test_preorigin_exclusion_and_failure_are_not_dropped():
    rows = [
        {"label": "unstable", "prospective_eligible": True, "qualified": False,
         "predicted_los": False, "window_settled": False},
        {"label": "stable", "prospective_eligible": True, "qualified": True,
         "predicted_los": False, "window_settled": True},
        {"label": "unstable", "prospective_eligible": False, "qualified": True,
         "predicted_los": True, "window_settled": False},
    ]
    summary = summarize(rows)
    assert summary["history_eligible"] == 3 and summary["eligible"] == 2
    assert summary["already_los_within_observation"] == 1
    assert summary["unstable_unresolved"] == 1 and summary["tn"] == 1
    assert summary["accuracy"] is None


def test_ten_ms_forecast_preserves_frozen_model_semantics():
    n = 10
    closure = NonlinearPassiveClosure(
        np.ones(n), np.ones(45), np.zeros((n, n)), np.zeros((n, 2)),
        np.zeros((2, 2)), np.eye(2),
    )
    model = LearnedNonlinearModel(
        closure, np.ones((2, 80)) * 0.1, np.ones(80), np.ones(20),
        NonlinearFitSettings(), {},
    )
    h, eq = history(), np.arange(10) * 0.03
    expected = model.predict(h, CLOCK, eq)
    actual = forecast(model, h, eq, 0.01)
    np.testing.assert_array_equal(actual, expected)


def test_registered_terminal_verdict_and_source_cohort_labels():
    from kron_reduce.workflows.kiki_stability_comparison import _cohort_name

    assert "RETRY" in SCIENTIFIC_VERDICTS
    assert [_cohort_name(s) for s in ("DEV", "TEST", "DEV_TEST")] == [
        "val", "test", "val_plus_test",
    ]
