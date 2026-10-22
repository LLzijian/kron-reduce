from dataclasses import replace

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import EquilibriumLearningSample
from kron_reduce.workflows.kiki_history_learning import available_prefix, score_predictions


def sample(event_id, count, source="A", line="L1"):
    history = CausalStateHistory(
        np.arange(21) * 0.01 - 0.2, np.zeros((21, 3)), np.zeros((21, 3)), np.zeros(3)
    )
    return EquilibriumLearningSample(
        event_id,
        source,
        line,
        "val",
        history,
        np.arange(count + 1) * 0.01,
        np.zeros((count + 1, 3)),
        np.zeros((count + 1, 3)),
        np.zeros(3),
    )


def test_available_score_denominator_and_complete_cohort():
    samples = [sample("short", 4), sample("long", 300, line="L2")]
    predictions = {"short": np.ones((5, 6)) * 2, "long": np.ones((301, 6)) * 4}
    score = score_predictions(predictions, samples, 3.0, np.ones(6))
    assert score["event_count"] == 2
    assert score["full_horizon_event_count"] == 1
    assert score["rows"][0]["scored_real_future_points"] == 4
    assert np.isclose(score["normalized_rmse"], np.sqrt((4 + 16) / 2))
    assert np.isclose(score["pooled_delta_rmse_rad"], np.sqrt(10))
    complete = score_predictions(predictions, samples, 1.0, np.ones(6), require_full=3.0)
    assert [r["event_id"] for r in complete["rows"]] == ["long"]
    assert complete["normalized_rmse"] == 4.0
    assert len(available_prefix(samples, 1.0)[0].forecast_time) == 5
    assert len(available_prefix(samples, 1.0)[1].forecast_time) == 101


def test_sources_lines_events_equal_and_failure_not_removed():
    samples = [sample("a1", 4), sample("a2", 4), sample("b1", 4, source="B")]
    predictions = {
        s.event_id: np.ones((5, 6)) * (1 if s.source_dataset == "A" else 3) for s in samples
    }
    score = score_predictions(predictions, samples, 1.0, np.ones(6))
    assert np.isclose(score["normalized_rmse"], np.sqrt(5))
    predictions["b1"] = "numeric failure"
    failed = score_predictions(predictions, samples, 1.0, np.ones(6))
    assert failed["event_count"] == 3 and failed["failed_event_count"] == 1
    assert failed["normalized_rmse"] is None
    assert failed["failures_imputed_as_zero"] is False


def test_no_complete_horizon_is_missing_not_zero():
    s = sample("short", 4)
    score = score_predictions({"short": np.zeros((5, 6))}, [s], 1.0, np.ones(6), require_full=1.0)
    assert score["event_count"] == 0 and score["normalized_rmse"] is None
    changed = replace(s, source_dataset="renamed")
    assert np.array_equal(changed.history.delta, s.history.delta)
