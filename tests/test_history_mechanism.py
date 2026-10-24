from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation.history_mechanism import (
    residual_statistics,
    weighted_residual_summary,
)


def _result():
    intervals, nodes = 10, 10
    h = np.linspace(.008, .012, intervals)
    c = np.linspace(.04, .14, intervals)
    scale = np.r_[np.linspace(.5, 1.4, nodes), np.linspace(.7, 1.6, nodes)]
    force2 = np.arange(1, 101, dtype=np.float64).reshape(10, 10) / 1000
    force4 = -force2 * 1.7
    average2, average4 = force2 / h[:, None], force4 / h[:, None]
    S = np.linspace(.8, 1.7, nodes)
    normalized2 = force2 * S[None, :]
    normalized4 = force4 * S[None, :]
    dynamic2 = np.arange(101, 201, dtype=np.float64).reshape(10, 10) / 100
    dynamic4 = -dynamic2 / 2
    dynamic_scaled2 = c[:, None] * dynamic2 * S[None, :]
    dynamic_scaled4 = c[:, None] * dynamic4 * S[None, :]
    kin = np.arange(1, 101, dtype=np.float64).reshape(10, 10) / 1e5
    return {
        "c": c, "h": h, "S": S, "force_residual2": force2, "force_residual4": force4,
        "force_residual_average_force2": average2, "force_residual_average_force4": average4,
        "force_residual_scaled2": normalized2, "force_residual_scaled4": normalized4,
        "dynamic_residual2": dynamic2, "dynamic_residual4": dynamic4,
        "dynamic_residual_scaled2": dynamic_scaled2, "dynamic_residual_scaled4": dynamic_scaled4,
        "kin": kin,
    }, scale


def test_residual_statistics_keep_force_dynamic_and_kinematics_separate_with_expected_weights():
    result, scale = _result()
    values = residual_statistics(result, scale)
    c = result["c"]
    h = result["h"]
    normalized_force = result["force_residual_scaled2"]
    dynamic_scaled = result["dynamic_residual_scaled2"]
    kin_scaled = result["kin"] / scale[:10]

    assert values["gauss2_force_max_abs_momentum"] == pytest.approx(
        np.max(np.abs(result["force_residual2"]))
    )
    assert values["gauss2_force_max_abs_average_force"] == pytest.approx(
        np.max(np.abs(result["force_residual_average_force2"]))
    )
    assert values["gauss2_force_max_abs_normalized"] == pytest.approx(
        np.max(np.abs(normalized_force))
    )
    assert values["gauss2_force_weighted_squared_residual"] == pytest.approx(
        np.sum((c[:, None] * normalized_force) ** 2) / 10
    )
    assert values["gauss2_dynamic_max_abs_momentum"] == pytest.approx(
        np.max(np.abs(result["dynamic_residual2"]))
    )
    assert values["gauss2_dynamic_max_abs_normalized"] == pytest.approx(
        np.max(np.abs(result["dynamic_residual2"] * result["S"][None, :]))
    )
    assert values["gauss2_dynamic_max_abs_weighted_design"] == pytest.approx(
        np.max(np.abs(dynamic_scaled))
    )
    assert values["gauss2_dynamic_weighted_squared_residual"] == pytest.approx(
        np.sum(dynamic_scaled**2) / 10
    )
    assert values["kinematic_max_abs_rad"] == pytest.approx(np.max(np.abs(result["kin"])))
    assert values["kinematic_max_abs_normalized"] == pytest.approx(np.max(np.abs(kin_scaled)))
    assert values["kinematic_weighted_squared_residual"] == pytest.approx(
        np.sum((h / h.sum())[:, None] * kin_scaled**2) / 10
    )
    assert "force_dynamic_total" not in values
    assert values["gauss2_force_weighted_squared_residual"] != pytest.approx(
        values["gauss2_dynamic_weighted_squared_residual"]
    )


def test_descriptive_overflow_is_a_local_numerical_error():
    result, scale = _result()
    result["dynamic_residual_scaled2"][:] = 1e308
    with np.errstate(over="ignore"), pytest.raises(FloatingPointError, match="nonfinite derived"):
        residual_statistics(result, scale)


def test_failed_population_is_none_and_never_success_subset_renormalized():
    weights = np.full(258, 1 / 258, dtype=np.float64)
    slots = []
    for seed in (324, 325, 326):
        for index in range(258):
            slots.append({
                "seed": seed, "cache_index": index,
                "status": "failed" if seed == 325 and index == 71 else "complete",
                "residual_statistics": {"force": float(index)},
            })
    summary = weighted_residual_summary(slots, weights)
    assert summary == {
        "complete": False,
        "reason": "at_least_one_allocated_slot_unresolved_or_failed",
        "weighted_residuals": None,
        "success_subset_renormalized": False,
    }


def test_complete_population_uses_every_original_seed_weight_and_frozen_order():
    weights = np.full(258, 1 / 258, dtype=np.float64)
    slots = []
    for seed in (324, 325, 326):
        for index in range(258):
            slots.append({
                "seed": seed, "cache_index": index, "status": "complete",
                "residual_statistics": {"force": float(index), "kin": float(seed)},
            })
    summary = weighted_residual_summary(slots, weights)
    assert summary["complete"] is True
    assert summary["events_per_seed"] == 258
    assert summary["success_subset_renormalized"] is False
    for seed in (324, 325, 326):
        row = summary["weighted_residuals"][str(seed)]
        assert row["force"]["weighted_mean"] == pytest.approx(weights @ np.arange(258))
        assert row["force"]["population_max"] == 257
        assert row["kin"]["weighted_mean"] == pytest.approx(seed)
    reordered = list(slots)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="frozen cache order"):
        weighted_residual_summary(reordered, weights)
