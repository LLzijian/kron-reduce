import numpy as np
import pytest

from kron_reduce.evaluation.trajectory_stability import classify, summarize


def _history():
    return np.zeros((11, 10)), np.zeros((11, 10))


def _path():
    return np.zeros((801, 20))


def _event(truth, prediction, *, qualified=True, prospective=True, **extra):
    return {
        "truth_label": truth,
        "predicted_label": prediction,
        "qualified": qualified,
        "prospective_eligible": prospective,
        **extra,
    }


def test_los_requires_five_consecutive_points_and_both_thresholds():
    q, v = _history()
    path = _path()
    path[1:5, 0], path[1:5, 10] = 4., 20.
    row = classify(q, v, path, np.zeros(10), np.ones(10))
    assert not row["predicted_los"]

    path[5, 0], path[5, 10] = 4., 20.
    row = classify(q, v, path, np.zeros(10), np.ones(10))
    assert row["predicted_los"]
    assert row["label"] == "unstable"

    angle_only = _path()
    angle_only[1:6, 0] = 4.
    assert not classify(q, v, angle_only, np.zeros(10), np.ones(10))["predicted_los"]

    speed_only = _path()
    speed_only[1:6, 10] = 20.
    assert not classify(q, v, speed_only, np.zeros(10), np.ones(10))["predicted_los"]


def test_history_splices_once_and_trigger_time_uses_point_one_second_origin():
    q, v = _history()
    q[8:, 0], v[8:, 0] = 4., 20.
    path = _path()
    path[0, :10] = 1e6  # The history endpoint is the forecast origin.
    path[1:3, 0], path[1:3, 10] = 4., 20.

    row = classify(q, v, path, np.zeros(10), np.ones(10))

    assert row["trigger_index"] == 12
    assert row["predicted_los_time_forecast_s"] == pytest.approx(0.02)
    assert row["label"] == "unstable"


def test_untriggered_but_unsettled_trajectory_is_unresolved():
    q, v = _history()
    path = _path()
    path[-100:, 0] = 4.  # Angle condition fails settled, while speed never triggers LOS.

    row = classify(q, v, path, np.zeros(10), np.ones(10))

    assert not row["predicted_los"]
    assert not row["window_settled"]
    assert row["label"] is None


def test_nonfinite_forecast_is_not_assigned_a_stability_label():
    q, v = _history()
    path = _path()
    path[-1, 0] = np.nan

    row = classify(q, v, path, np.zeros(10), np.ones(10))

    assert not row["finite_full_path"]
    assert row["label"] is None


def test_summary_counts_confusion_unresolved_uncovered_and_conditional_accuracy():
    events = [
        _event("unstable", "unstable"),  # TP
        _event("unstable", "stable"),    # FN
        _event("stable", "unstable"),    # FP
        _event("stable", "stable"),      # TN
        _event("unstable", None, qualified=False, numerical_failure=True),
        _event("stable", None, qualified=False, unsettled=True),
        _event(None, None),  # Outside the frozen truth-label population.
    ]

    row = summarize(events)

    assert (row["tp"], row["fn"], row["fp"], row["tn"]) == (1, 1, 1, 1)
    assert row["accuracy"] is None
    assert row["accuracy_on_resolved"] == pytest.approx(0.5)
    assert row["allocated"] == 7
    assert row["resolved"] == 4
    assert row["unresolved"] == 2
    assert row["uncovered"] == 1
    assert row["numerical_failures"] == 1
    assert row["unsettled_predictions"] == 1


def test_missing_truth_class_has_null_balanced_accuracy():
    row = summarize([_event("stable", "stable")])

    assert row["tn"] == 1
    assert row["accuracy"] == 1.
    assert row["balanced_accuracy"] is None
    assert row["recall"] is None
    assert row["fsr"] is None
