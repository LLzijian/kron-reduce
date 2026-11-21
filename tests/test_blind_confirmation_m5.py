from types import SimpleNamespace

import numpy as np
import pytest

from kron_reduce.evaluation import blind_confirmation_m5 as m5


def test_failed_batch_consumes_each_event_step_and_budget_errors_escape(monkeypatch):
    events = [{"id": key, "x": np.zeros((12, 20)), "p0": np.zeros(10), "equilibrium": np.zeros(10)}
              for key in ("one", "two")]
    scales = {"history": np.ones(20), "memory_radius": 1.}
    calls = []

    def fail(model, q, v, p, eq, scale, count, **options):
        calls.append((len(q), count, options["soft_radius"]))
        raise FloatingPointError("synthetic GN failure")

    monkeypatch.setattr(m5, "predict_first_swing_fit", fail)
    predict, paths, attempted, blocked = m5._predict_once(
        "v4_hard_r", None, events, scales, {"prior_weight": 1e-6}, SimpleNamespace(spend=None))
    with pytest.raises(FloatingPointError, match="synthetic GN failure"):
        predict(events, .0025)
    for event in events:
        with pytest.raises(FloatingPointError, match="already consumed"):
            predict([event], .0025)
    assert calls == [(2, 800, False)]
    assert attempted == {("one", "0.0025"), ("two", "0.0025")}
    assert blocked == {"0.0025": 2}
    assert np.isnan(paths["0.0025"]).all()

    def stop(*args, **kwargs):
        raise RuntimeError("unregistered or exhausted operation budget: rollouts")

    monkeypatch.setattr(m5, "predict_first_swing_fit", stop)
    with pytest.raises(RuntimeError, match="operation budget"):
        predict(events, .00125)


def test_engineering_warnings_only_include_numerically_qualified_unsettled():
    common = {"prospective_eligible": True, "predicted_label": None}
    rows = [common | {"truth_label": "unstable", "qualified": True, "unsettled": True, "numerical_failure": False},
            common | {"truth_label": "stable", "qualified": False, "unsettled": False, "numerical_failure": True}]
    strict = m5._classification(rows)
    alarm = m5._classification(rows, engineering=True)
    assert strict["confirmed_recall"] == 0 and strict["unsettled_predictions"] == 1
    assert alarm["confirmed_recall"] == 1 and alarm["unsettled_predictions"] == 0
    assert strict["numerical_failures"] == alarm["numerical_failures"] == 1
    assert strict["accuracy"] is alarm["accuracy"] is None
    assert alarm["tp"] == 1 and alarm["fp"] == 0 and alarm["coverage"] == .5


def test_bootstrap_reuses_group_draws_and_retains_invalid_denominators():
    events = [{"label": label, "group": group}
              for label, group in (("unstable", "line34-35"), ("unstable", "line34-35"),
                                   ("stable", "bus12"), ("stable", "bus12"))]
    hard = (np.array([1, 0, 0, 0]), np.array([1., 2., np.nan, np.nan]),
            np.array([np.nan, np.nan, .4, .6]))
    soft = (np.array([1, 1, 0, 0]), np.array([.5, 1.5, np.nan, np.nan]),
            np.array([np.nan, np.nan, .3, .5]))
    missing = (soft[0], np.array([np.nan, 1.5, np.nan, np.nan]), soft[2])
    vectors = {name: dict.fromkeys(m5.STEPS, values)
               for name, values in zip(m5.MODEL_NAMES, (hard, soft, missing), strict=True)}
    result = m5._cluster_bootstrap(events, vectors, 117)
    by_step = result["by_step"]
    assert result["replicates"] == m5.BOOTSTRAP_REPLICATES == 10_000
    assert result["groups"] == {"bus12": 2, "line34-35": 2}
    assert by_step[m5.STEPS[0]] == by_step[m5.STEPS[1]]
    a, b = (by_step[m5.STEPS[0]][name] for name in m5.MODEL_NAMES[1:])
    recall, peak, stable = a.values()
    assert recall["delta"] == .5 and recall["percentile_ci95"] == [.5, .5]
    assert peak["delta"] == -.5 and peak["invalid_replicates"] == recall["invalid_replicates"] > 0
    assert stable["delta"] == pytest.approx(-.1)
    assert stable["valid_replicates"] + stable["invalid_replicates"] == m5.BOOTSTRAP_REPLICATES
    assert b["unstable_median_positive_peak_deficit_rad_s"] == {
        "delta": None, "percentile_ci95": None, "valid_replicates": 0, "invalid_replicates": 10_000,
    }
