from __future__ import annotations

import inspect

import numpy as np
import pytest

from kron_reduce.evaluation.record20_coordinates import inspect_record20_arrays
from kron_reduce.physics.coordinates import coi_project, coi_project_power

OMEGA_S = 120.0 * np.pi


def _inputs(n: int = 5):
    inertia = np.arange(1.0, 11.0, dtype=np.float64) ** 1.2
    eq_delta = np.linspace(-0.2, 0.3, 10, dtype=np.float64)
    eq_pm = np.linspace(-0.5, 0.4, 10, dtype=np.float64)
    delta = np.arange(n * 10, dtype=np.float64).reshape(n, 10) * 0.002 + eq_delta
    omega = 1.0 + np.arange(n * 10, dtype=np.float64).reshape(n, 10) * 0.0001
    t = np.arange(n, dtype=np.float64) * 0.01
    origin_pm = np.linspace(-0.3, 0.7, 10, dtype=np.float64)
    return {
        "t": t,
        "delta": delta,
        "omega": omega,
        "origin_pm": origin_pm,
        "inertia": inertia,
        "equilibrium_delta": eq_delta,
        "equilibrium_pm": eq_pm,
        "omega_s": OMEGA_S,
    }


def _inspect(**kwargs):
    return inspect_record20_arrays(**kwargs)


def test_nonuniform_mass_q_v_and_power_projections_are_distinct_and_correct():
    inputs = _inputs()
    summary, data, transformed = _inspect(**inputs)
    mass = inputs["inertia"] / OMEGA_S
    expected_eq = coi_project(inputs["equilibrium_delta"], mass)
    expected_q = coi_project(inputs["delta"], mass) - expected_eq
    expected_v = coi_project((inputs["omega"] - 1.0) * OMEGA_S, mass)
    expected_power = coi_project_power(inputs["origin_pm"] - inputs["equilibrium_pm"], mass)

    assert summary["record20_input_supported"] is True
    np.testing.assert_allclose(transformed["q"], expected_q, rtol=0, atol=1e-14)
    np.testing.assert_allclose(transformed["v"], expected_v, rtol=0, atol=1e-14)
    np.testing.assert_allclose(transformed["power"], expected_power, rtol=0, atol=1e-14)
    np.testing.assert_allclose(data["power"], expected_power, rtol=0, atol=1e-14)
    assert abs(np.sum(transformed["power"])) <= 1e-10
    assert abs(np.sum(transformed["q"] * mass, axis=1)).max() <= 1e-10
    assert abs(np.sum(transformed["v"] * mass, axis=1)).max() <= 1e-10


def test_api_exposes_only_causal_arrays_and_static_priors():
    parameters = set(inspect.signature(inspect_record20_arrays).parameters)
    assert parameters == {
        "t", "delta", "omega", "origin_pm", "inertia", "equilibrium_delta",
        "equilibrium_pm", "omega_s",
    }
    summary, data, transformed = _inspect(**_inputs())
    assert set(data) == {"time", "q", "v", "power", "mass", "equilibrium_delta", "equilibrium_pm", "omega_s"}
    assert set(transformed) == {"q", "v", "power"}
    assert "label" not in summary and "event_id" not in summary and "valid_length" not in summary
    with pytest.raises(TypeError):
        _inspect(**_inputs(), label="unstable")


def test_history_origin_preserves_the_actual_float_time_differences():
    inputs = _inputs()
    inputs["t"] = np.array([0.0, 0.01000000000003, 0.02000000000007, 0.03, 0.04])
    _, data, _ = _inspect(**inputs)
    expected = inputs["t"][:3] - inputs["t"][2]
    np.testing.assert_array_equal(data["time"], expected)
    assert data["time"][0] == -inputs["t"][2]


def test_five_points_supported_four_have_only_history_and_two_have_no_input():
    five, five_data, _ = _inspect(**_inputs(5))
    assert five["history_available"] is True
    assert five["at_least_two_real_future_points"] is True
    assert five_data is not None

    four, four_data, _ = _inspect(**_inputs(4))
    assert four["history_available"] is True
    assert four["at_least_two_real_future_points"] is False
    assert four["record20_input_supported"] is False
    assert four_data is not None

    two, two_data, _ = _inspect(**_inputs(2))
    assert two["history_available"] is False
    assert two["record20_input_supported"] is False
    assert two_data is None


def test_later_nan_invalidates_full_record_but_keeps_legal_history_arrays():
    inputs = _inputs(6)
    inputs["delta"][5, 4] = np.nan
    summary, data, transformed = _inspect(**inputs)
    assert summary["history_available"] is True
    assert summary["at_least_two_real_future_points"] is True
    assert summary["full_visible_record_valid"] is False
    # A later diagnostic failure does not erase the legal causal input or
    # the first two future points. The complete-record gate still rejects it.
    assert summary["record20_input_supported"] is True
    assert "full_visible_record_invalid" in summary["reasons"]
    assert data is not None
    assert np.isfinite(data["q"][:3]).all()
    assert np.isnan(transformed["q"][5]).any()


def test_nan_origin_power_prevents_input_storage():
    inputs = _inputs()
    inputs["origin_pm"][2] = np.nan
    summary, data, transformed = _inspect(**inputs)
    assert summary["history_available"] is False
    assert summary["record20_input_supported"] is False
    assert summary["full_visible_record_valid"] is True
    assert summary["at_least_two_real_future_points"] is True
    assert "origin_power_unavailable_or_invalid" in summary["reasons"]
    assert data is None
    assert summary["power_sum_abs"] is None
    assert np.isnan(transformed["power"]).any()


def test_missing_origin_power_keeps_visible_diagnostics_without_fabricated_power():
    inputs = _inputs(6)
    inputs["origin_pm"] = None
    summary, data, transformed = _inspect(**inputs)
    assert summary["full_visible_record_valid"] is True
    assert summary["at_least_two_real_future_points"] is True
    assert summary["history_available"] is False
    assert summary["record20_input_supported"] is False
    assert summary["power_sum_abs"] is None
    assert data is None
    assert set(transformed) == {"q", "v"}
    assert "origin_power_unavailable_or_invalid" in summary["reasons"]


def test_bad_clock_is_unavailable_and_unsafe_static_priors_raise():
    inputs = _inputs()
    inputs["t"][2] += 1e-5
    summary, data, _ = _inspect(**inputs)
    assert summary["history_available"] is False
    assert summary["record20_input_supported"] is False
    assert data is None

    for key, value in (
        ("omega_s", 1.0),
        ("inertia", np.r_[0.0, np.ones(9)]),
        ("equilibrium_delta", np.r_[np.nan, np.zeros(9)]),
        ("equilibrium_pm", np.r_[np.inf, np.zeros(9)]),
    ):
        invalid = _inputs()
        invalid[key] = value
        with pytest.raises(ValueError, match="source_static_prior_invalid"):
            _inspect(**invalid)
