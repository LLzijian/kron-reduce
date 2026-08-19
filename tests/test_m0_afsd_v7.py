from __future__ import annotations

from kron_reduce.workflows.m0_afsd_v7_modal_structure_discovery import (
    _select_candidate,
    _validate_scope,
)


def _summary(branch, rmse, frequency, damping, mac, pairs):
    return {
        "branch": branch,
        "mode_pairs": pairs,
        "rollout_10s": {"mean_rmse": rmse, "failure_rate": 0.0},
        "modal_10s": {
            "matched_modes": 5,
            "frequency_error": frequency,
            "damping_error": damping,
            "MAC": mac,
        },
        "physics": {"status": "pass"},
    }


def test_v7_selector_rejects_frequency_gain_that_degrades_mode_shape() -> None:
    baseline = _summary("SS-DM", 1.0, 0.24, 0.09, 0.50, [])
    shape_bad = _summary("SS-DM+R1.6z0.05", 0.88, 0.08, 0.06, 0.42, [[1.6, 0.05]])
    joint_gain = _summary("SS-DM+R1.3z0.1", 0.91, 0.18, 0.04, 0.51, [[1.3, 0.1]])
    selected = _select_candidate(
        [baseline, shape_bad, joint_gain],
        {"minimum_model_improvement": 0.02, "maximum_mac_degradation": 0.0},
    )
    assert selected["route"] == "SS-DM+R1.3z0.1"
    assert not shape_bad["selection_gates"]["MAC_noninferior"]


def test_v7_selector_uses_registered_modal_repeatability_margins() -> None:
    baseline = _summary("SS-DM", 1.0, 0.078, 0.068, 0.586, [])
    within_repeatability = _summary(
        "SS-DM+R1.3z0.1", 0.96, 0.080, 0.072, 0.627, [[1.3, 0.1]]
    )
    selected = _select_candidate(
        [baseline, within_repeatability],
        {
            "minimum_model_improvement": 0.02,
            "maximum_mac_degradation": 0.0,
            "modal_frequency_noninferiority_margin": 0.02,
            "modal_damping_noninferiority_margin": 0.02,
            "modal_mac_noninferiority_margin": 0.05,
        },
    )
    assert selected["route"] == "SS-DM+R1.3z0.1"


def test_v7_scope_locks_test_other_folds_and_observed_pm_name() -> None:
    valid = {
        "fold": "F1", "seed": 324, "pm_mode": "observed",
        "test_access": "forbidden", "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden",
        "rollout_label": "conditional rollout with observed Pm",
    }
    _validate_scope(valid)
    for key, value in (("fold", "F2"), ("test_access", "allowed"), ("pm_mode", "zoh")):
        invalid = {**valid, key: value}
        try:
            _validate_scope(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"v7 scope accepted forbidden {key}={value}")
