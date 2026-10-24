from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from kron_reduce.evaluation.future_transfer_metrics import (
    FutureMetricsError,
    compare_future,
    hierarchy_weights,
    weighted_cvar,
)


def _fixture():
    rows = [
        {"event_id": "a1", "source_dataset": "A", "fault_line": "L1"},
        {"event_id": "a2", "source_dataset": "A", "fault_line": "L1"},
        {"event_id": "b1", "source_dataset": "B", "fault_line": "L1"},
        {"event_id": "b2", "source_dataset": "B", "fault_line": "L2"},
    ]
    weights = hierarchy_weights(rows)
    event_times = (np.arange(301) * 0.01, np.arange(51) * 0.01,
                   np.arange(151) * 0.01, np.arange(101) * 0.01)
    offsets = [0]
    for values in event_times:
        offsets.append(offsets[-1] + len(values))
    times = np.concatenate([np.asarray(values) for values in event_times])
    rng = np.random.default_rng(113)
    observed = rng.normal(size=(offsets[-1], 20))
    candidate = np.zeros((offsets[-1], 22), dtype=np.float64)
    baseline = np.zeros_like(candidate)
    scale = np.linspace(0.8, 1.7, 20)
    for start, end in pairwise(offsets):
        candidate[start:end, :20] = observed[start:end] + 0.1
        baseline[start:end, :20] = observed[start:end] + 0.2
        candidate[start, :20] = observed[start]
        baseline[start, :20] = observed[start]
        candidate[start:end, 20:] = 0.3
        baseline[start:end, 20:] = 0.2
    return candidate, baseline, observed, times, np.asarray(offsets), scale, weights


def test_future_metrics_use_noninitial_true_available_points_and_hierarchical_weights():
    candidate, baseline, observed, times, offsets, scale, weights = _fixture()
    result = compare_future(candidate, baseline, observed, times, offsets, scale, weights)
    assert result["I_pass"] is True
    assert result["failure_reasons"] == []
    assert result["event_count"] == 4
    assert np.array_equal(weights, np.array([0.25, 0.25, 0.25, 0.25]))
    assert np.array_equal(result["original_weights"], weights)
    assert result["candidate"]["complete_3s"]["event_count"] == 1
    assert result["candidate"]["complete_3s"]["original_weight_mass"] == 0.25
    assert result["candidate"]["duration_groups"]["all"]["under_1s"]["event_count"] == 1
    assert result["candidate"]["duration_groups"]["all"]["from_1s_to_under_3s"]["event_count"] == 2
    assert result["candidate"]["duration_groups"]["all"]["complete_3s"]["event_count"] == 1
    primary = result["candidate"]["primary_weighted_mse"]["all"]
    horizons = result["candidate"]["horizon_weighted_mse"]
    assert primary == pytest.approx(0.5 * (horizons["1.0"]["all"] + horizons["3.0"]["all"]))
    # The initial row was set to the exact origin and never appears in the scored point counts.
    assert result["candidate_events"][0]["horizon_point_counts"] == {"1.0": 100, "3.0": 300}
    assert result["candidate_events"][1]["horizon_point_counts"] == {"1.0": 50, "3.0": 50}
    assert result["candidate_events"][2]["horizon_point_counts"] == {"1.0": 100, "3.0": 150}
    assert result["candidate_events"][3]["horizon_point_counts"] == {"1.0": 100, "3.0": 100}
    assert result["candidate_events"][3]["duration_group"] == "from_1s_to_under_3s"


def test_finite_metric_inputs_with_derived_overflow_preserve_completed_event_prefix():
    candidate, baseline, observed, times, offsets, scale, weights = _fixture()
    candidate[offsets[1] + 1, :20] = 1e200
    assert np.isfinite(candidate).all()
    with pytest.raises(FloatingPointError) as caught:
        compare_future(candidate, baseline, observed, times, offsets, scale, weights)
    assert len(caught.value.partial_record["candidate_events"]) == 1
    assert len(caught.value.partial_record["baseline_events"]) == 1


def test_fractional_weighted_cvar_respects_boundary_mass_and_frozen_order():
    assert weighted_cvar([10.0, 8.0, 8.0, 1.0], [0.04, 0.08, 0.08, 0.8]) == pytest.approx(8.8)
    with pytest.raises(FutureMetricsError, match="sum to one"):
        weighted_cvar([1.0, 2.0], [0.2, 0.2])


def test_strict_primary_margin_and_nonincrease_tolerance_are_distinct():
    candidate, baseline, observed, times, offsets, scale, weights = _fixture()
    # Tiny strict improvement is intentionally insufficient under the 1e-12 rule.
    candidate[:, :20] = observed + np.sqrt(0.2**2 - 5e-13)
    for start in offsets[:-1]:
        candidate[start, :20] = observed[start]
    result = compare_future(candidate, baseline, observed, times, offsets, scale, weights)
    assert result["conditions"]["primary_strictly_lower"] is False
    assert "primary_strictly_lower" in result["failure_reasons"]

    candidate, baseline, observed, times, offsets, scale, weights = _fixture()
    # Reduce q error enough to improve primary, but make v exceed its own allowed tolerance.
    for start, end in pairwise(offsets):
        candidate[start:end, :10] = observed[start:end, :10] + 0.01
        candidate[start:end, 10:20] = observed[start:end, 10:20] + np.sqrt(0.2**2 + 4e-12)
        candidate[start, :20] = observed[start]
    result = compare_future(candidate, baseline, observed, times, offsets, scale, weights)
    assert result["conditions"]["primary_strictly_lower"] is True
    assert result["conditions"]["primary_v_not_higher"] is False
    assert "h1_q_not_higher" not in result["conditions"]


def test_full_three_second_conditional_mse_uses_original_subset_weights():
    candidate, baseline, observed, times, offsets, scale, weights = _fixture()
    result = compare_future(candidate, baseline, observed, times, offsets, scale, weights)
    # Only event 0 reaches 3 s, so its conditional weight is its full event loss.
    candidate_full = result["candidate"]["complete_3s"]["conditional_3s_mse"]["all"]
    expected = result["candidate_events"][0]["horizon_mse"]["3.0"]["all"]
    assert candidate_full == pytest.approx(expected)
    assert result["candidate"]["complete_3s"]["event_count"] == 1


def test_invalid_shapes_offsets_clocks_nonfinite_and_wrong_origin_are_rejected():
    candidate, baseline, observed, times, offsets, scale, weights = _fixture()
    with pytest.raises(FutureMetricsError, match="integer vector"):
        compare_future(candidate, baseline, observed, times, offsets.astype(float), scale, weights)
    bad_offsets = offsets.copy()
    bad_offsets[-1] -= 1
    with pytest.raises(FutureMetricsError, match="partition every event"):
        compare_future(candidate, baseline, observed, times, bad_offsets, scale, weights)
    bad_times = times.copy()
    bad_times[1] = bad_times[0]
    with pytest.raises(FutureMetricsError, match="strictly increase"):
        compare_future(candidate, baseline, observed, bad_times, offsets, scale, weights)
    bad_path = candidate.copy()
    bad_path[0, 0] += 1e-15
    with pytest.raises(FutureMetricsError, match="t0 does not match"):
        compare_future(bad_path, baseline, observed, times, offsets, scale, weights)
    bad_path = candidate.copy()
    bad_path[-1, 0] = np.nan
    with pytest.raises(FutureMetricsError, match="finite"):
        compare_future(bad_path, baseline, observed, times, offsets, scale, weights)
    with pytest.raises(FutureMetricsError, match="target_scale"):
        compare_future(candidate, baseline, observed, times, offsets, np.zeros(20), weights)
    with pytest.raises(FutureMetricsError, match="tolerances"):
        compare_future(candidate, baseline, observed, times, offsets, scale, weights, atol=True)
    extra_offsets = offsets.copy()
    extra_offsets[1:] += 1
    split = int(offsets[1])
    extra_candidate = np.vstack((candidate[:split], candidate[split - 1:split], candidate[split:]))
    extra_baseline = np.vstack((baseline[:split], baseline[split - 1:split], baseline[split:]))
    extra_observed = np.vstack((observed[:split], observed[split - 1:split], observed[split:]))
    extra_times = np.r_[times[:split], times[split - 1] + 0.01, times[split:]]
    with pytest.raises(FutureMetricsError, match="more than 300"):
        compare_future(extra_candidate, extra_baseline, extra_observed,
                       extra_times, extra_offsets, scale, weights)


def test_hierarchy_weights_are_source_local_and_reject_duplicate_ids():
    rows = [
        {"event_id": "one", "source_dataset": "A", "fault_line": "L1"},
        {"event_id": "two", "source_dataset": "A", "fault_line": "L2"},
        {"event_id": "three", "source_dataset": "B", "fault_line": "L1"},
    ]
    assert np.array_equal(hierarchy_weights(rows), np.array([0.25, 0.25, 0.5]))
    with pytest.raises(FutureMetricsError, match="duplicate event_id"):
        hierarchy_weights([rows[0], rows[0]])
