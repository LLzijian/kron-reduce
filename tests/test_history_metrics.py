"""Synthetic checks for A14 population metrics and frozen weighting rules."""
from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation import history_metrics as metrics


def test_event_losses_use_only_noninitial_qv_points_with_correct_denominators():
    observed = np.zeros((2, 11, 20), dtype=np.float64)
    paths = np.zeros((2, 11, 22), dtype=np.float64)
    paths[:, 0, :20] = 1e6  # initial point is intentionally excluded
    paths[0, 1:, :10] = 2.0
    paths[0, 1:, 10:20] = 1.0
    paths[1, 1:, :10] = 1.0
    paths[1, 1:, 10:20] = 3.0

    losses = metrics.event_losses(paths, observed, np.ones(20))
    np.testing.assert_array_equal(losses["q"], [4.0, 1.0])
    np.testing.assert_array_equal(losses["v"], [1.0, 9.0])
    np.testing.assert_array_equal(losses["all"], [2.5, 5.0])


def test_hierarchical_weights_are_source_then_line_then_event_equal_mass():
    rows = [
        {"source_dataset": "A", "fault_line": "X"},
        {"source_dataset": "A", "fault_line": "X"},
        {"source_dataset": "A", "fault_line": "Y"},
        {"source_dataset": "B", "fault_line": "X"},
    ]
    weights = metrics.hierarchy_weights(rows)
    np.testing.assert_allclose(weights, [0.125, 0.125, 0.25, 0.5], rtol=0, atol=1e-15)
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)
    assert metrics.hierarchy_weights(rows[::-1])[::-1].tolist() == weights.tolist()


def test_weighted_cvar_uses_fractional_tail_mass_and_preserves_tie_order():
    assert metrics.weighted_cvar([1.0, 3.0, 5.0], [0.75, 0.15, 0.10], 0.20) == pytest.approx(4.0)
    # The tail takes .05 from each tied top event; stable ordering makes the
    # selected boundary deterministic while equal losses preserve the value.
    assert metrics.weighted_cvar([10.0, 10.0, 0.0], [0.05, 0.15, 0.80], 0.10) == pytest.approx(10.0)
    assert metrics.weighted_cvar([1.0, 2.0, 3.0], [0.2, 0.3, 0.5], 1.0) == pytest.approx(2.3)


def test_comparison_returns_full_population_and_applies_strict_I_conditions():
    observed = np.zeros((2, 11, 20), dtype=np.float64)
    baseline = np.zeros((2, 11, 22), dtype=np.float64)
    candidate = np.zeros_like(baseline)
    baseline[:, 1:, :10] = 2.0
    baseline[:, 1:, 10:20] = 2.0
    candidate[:, 1:, :10] = 1.0
    candidate[:, 1:, 10:20] = 2.0
    result = metrics.comparison(candidate, baseline, observed, np.ones(20), [0.5, 0.5])
    assert result["I_pass"] is True
    assert result["conditions"] == {
        "all_strictly_lower": True,
        "q_not_higher": True,
        "v_not_higher": True,
        "worst10_cvar_not_higher": True,
    }
    assert result["candidate_event_losses"]["all"].shape == (2,)
    assert result["weighted_mse"]["all"]["candidate"] < result["weighted_mse"]["all"]["baseline"]


@pytest.mark.parametrize("bad", [
    "nonfinite", "bad_shape", "bad_scale", "bad_weights", "negative_loss", "partial_weights",
])
def test_invalid_arrays_or_weights_refuse_without_subset_renormalization(bad):
    observed = np.zeros((2, 11, 20), dtype=np.float64)
    paths = np.zeros((2, 11, 22), dtype=np.float64)
    scale = np.ones(20)
    weights = np.array([0.5, 0.5])
    if bad == "nonfinite":
        paths[1, 4, 0] = np.nan
    elif bad == "bad_shape":
        paths = paths[:, :, :21]
    elif bad == "bad_scale":
        scale[0] = 0
    elif bad == "bad_weights":
        weights = np.array([0.2, 0.2])
    elif bad == "negative_loss":
        with pytest.raises(ValueError):
            metrics.weighted_cvar([-1.0, 2.0], [0.5, 0.5])
        return
    elif bad == "partial_weights":
        with pytest.raises(ValueError, match="sum to one"):
            metrics.weighted_cvar([2.0], [0.4], 0.1)
        return
    if bad == "bad_weights":
        metrics.event_losses(paths, observed, scale)
    else:
        with pytest.raises(ValueError):
            metrics.event_losses(paths, observed, scale)
    with pytest.raises(ValueError):
        metrics.comparison(paths, np.zeros((2, 11, 22)), observed, scale, weights)
