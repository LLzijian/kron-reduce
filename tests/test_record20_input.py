from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation.record20_input import audit_record20_rule


def _arrays(n: int):
    t = np.arange(n, dtype=np.float64) * 0.01
    delta = np.zeros((n, 2), dtype=np.float64)
    omega = np.ones((n, 2), dtype=np.float64)
    return t, delta, omega


def _make_joint(delta, omega, indices):
    for index in indices:
        delta[index] = (0.0, np.pi + 0.1)
        omega[index] = (1.0 - 0.03, 1.0 + 0.03)


def test_shortest_five_joint_points_trigger_at_the_fifth_point():
    t, delta, omega = _arrays(5)
    _make_joint(delta, omega, range(5))
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["status"] == "valid"
    assert result["first_joint_violation"]["index"] == 0
    assert result["pre_record_onset_unknown"] is True
    assert result["first_five_point_window"]["start"]["index"] == 0
    assert result["first_five_point_window"]["end"]["index"] == 4
    assert result["origins"]["2"]["continuous_joint_violations_ending_at_origin"] == 3
    assert result["origins"]["2"]["five_point_trigger_confirmed_by_origin"] is False
    assert result["origins"]["10"] is None


def test_hold_counter_breaks_on_any_nonjoint_point_and_first_run_is_reported():
    t, delta, omega = _arrays(11)
    _make_joint(delta, omega, [0, 3, 4, 6, 7, 8, 9, 10])
    # At index 5, angle remains over its threshold but speed is not.
    omega[5] = (1.0, 1.0)
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["first_joint_violation"]["index"] == 0
    assert result["pre_record_onset_unknown"] is True
    assert result["first_five_point_window"]["start"]["index"] == 6
    assert result["first_five_point_window"]["end"]["index"] == 10
    assert result["origins"]["10"]["continuous_joint_violations_ending_at_origin"] == 5
    assert result["origins"]["10"]["five_point_trigger_confirmed_by_origin"] is True


def test_angle_and_speed_must_both_strictly_exceed_their_limits():
    t, delta, omega = _arrays(5)
    delta[:, 1] = np.pi
    omega[:] = (0.0, 0.04)
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["angle_violation"] == [False] * 5
    assert result["speed_violation"] == [False] * 5
    assert result["joint_violation"] == [False] * 5
    assert result["first_five_point_window"]["end"]["index"] is None
    assert result["pre_record_onset_unknown"] is False

    delta[:, 1] = np.pi + 1e-12
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["angle_violation"] == [True] * 5
    assert result["speed_violation"] == [False] * 5
    assert result["first_five_point_window"]["end"]["index"] is None


def test_nonuniform_inertia_uses_raw_per_unit_weighted_coi_and_actual_clock_difference():
    t, delta, omega = _arrays(12)
    delta[:, 1] = 4.0
    omega[:] = (1.0, 1.0)
    omega[3] = (1.1, 1.0)
    result = audit_record20_rule(t, delta, omega, np.array([3.0, 1.0]))
    expected = omega[3] - np.sum(omega[3] * np.array([3.0, 1.0])) / 4.0
    assert result["coi_speed_pu"][3] == np.max(np.abs(expected))
    at_two = result["origins"]["2"]
    first_speed = at_two["first_speed_violation"]
    assert first_speed["index"] == 3
    assert first_speed["actual_relative_s"] == float(t[3] - t[2])
    assert first_speed["nominal_relative_s"] == pytest.approx(0.01)
    assert first_speed["actual_relative_s"] != first_speed["nominal_relative_s"]


def test_three_point_origin_count_and_100_point_tail_have_exact_endpoints():
    t, delta, omega = _arrays(100)
    _make_joint(delta, omega, [8, 9, 10, 11, 12])
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["origins"]["2"]["continuous_joint_violations_ending_at_origin"] == 0
    assert result["origins"]["10"]["continuous_joint_violations_ending_at_origin"] == 3
    assert result["origins"]["10"]["five_point_trigger_confirmed_by_origin"] is False
    assert result["stable_tail_100"]["start_index"] == 0
    assert result["stable_tail_100"]["end_index"] == 99
    assert result["stable_source_acceptance"] is False


def test_1501_point_stable_tail_is_1401_through_1500_and_acceptance_needs_15s():
    t, delta, omega = _arrays(1501)
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 2.0]))
    tail = result["stable_tail_100"]
    assert tail["start_index"] == 1401
    assert tail["end_index"] == 1500
    assert tail["both_conditions"] is True
    assert tail["start_time_s"] == t[1401]
    assert tail["end_time_s"] == t[1500]
    assert result["stable_source_acceptance"] is True

    t[-1] -= 5e-11
    tolerant_endpoint = audit_record20_rule(t, delta, omega, np.array([1.0, 2.0]))
    assert tolerant_endpoint["status"] == "valid"
    assert tolerant_endpoint["stable_source_acceptance"] is True


@pytest.mark.parametrize("n", (1, 2))
def test_less_than_three_points_keeps_origin_two_unavailable(n):
    t, delta, omega = _arrays(n)
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["status"] == "valid"
    assert result["origins"]["2"] is None
    assert result["origins"]["10"] is None
    assert result["stable_tail_100"] is None
    assert result["stable_source_acceptance"] is False


def test_nonfinite_or_misclock_is_invalid_not_a_no_trigger():
    t, delta, omega = _arrays(6)
    delta[2, 0] = np.nan
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["status"] == "invalid"
    assert result["invalid_reason"] == "nonfinite_visible_record"
    assert result["first_five_point_window"]["end"]["index"] is None

    t, delta, omega = _arrays(6)
    t[3] += 1e-5
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["status"] == "invalid"
    assert result["invalid_reason"] == "clock_invalid"


@pytest.mark.parametrize("overflow_kind", ("angle_range", "speed_center"))
def test_finite_raw_values_with_overflowing_derived_statistics_are_invalid(overflow_kind):
    t, delta, omega = _arrays(5)
    if overflow_kind == "angle_range":
        delta[:, 0] = 1e308
        delta[:, 1] = -1e308
    else:
        omega[:] = 1e308
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["status"] == "invalid"
    assert result["invalid_reason"] == "nonfinite_derived_statistic"
    assert result["angle_spread_rad"] is None
    assert result["coi_speed_pu"] is None
    assert result["joint_violation"] is None
    assert result["first_joint_violation"]["index"] is None
    assert result["first_five_point_window"]["end"]["index"] is None


def test_registered_length_bound_is_enforced_before_rule_classification():
    t, delta, omega = _arrays(1502)
    result = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    assert result["status"] == "invalid"
    assert result["invalid_reason"] == "record_exceeds_registered_1501_point_bound"

