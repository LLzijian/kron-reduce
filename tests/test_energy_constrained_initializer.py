"""Small algebra-only tests for the energy-ball latent initializer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kron_reduce.evaluation.energy_constrained_initializer import solve_energy_ball


def _rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _case(sigma=(3.0, 1.0), g=(1.0, 2.0), *, angle_u=0.0, angle_v=0.0,
          threshold=0.0):
    u, v = _rotation(angle_u), _rotation(angle_v)
    singular = np.asarray(sigma, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    matrix = u @ np.diag(singular) @ v.T
    right = u @ g
    rank = int(np.count_nonzero(singular > threshold))
    svd = {"u": u, "s": singular, "vh": v.T, "rank": rank, "threshold": threshold}
    coefficients = np.divide(g, singular, out=np.zeros_like(g), where=singular != 0)
    unconstrained = v @ coefficients
    return matrix, right, svd, unconstrained


def _assert_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, np.ndarray):
        assert np.isfinite(value).all()
    elif isinstance(value, (float, np.floating)):
        assert math.isfinite(float(value))


def _assert_active_solution(result, matrix, right, radius):
    z = result["z"]
    multiplier = result["multiplier"]
    assert result["status"] == "active"
    assert result["multiplier_kind"] == "finite"
    assert result["constraint_active"] is True
    assert multiplier > 0
    assert np.linalg.norm(z) <= radius * (1 + 1e-12)
    stationarity = matrix.T @ (matrix @ z - right) + multiplier * z
    scale = max(1.0, np.linalg.norm(matrix.T @ right), multiplier * np.linalg.norm(z))
    assert np.linalg.norm(stationarity) / scale <= 1e-10
    assert result["kkt_stationarity_scaled"] <= 1e-10
    assert result["kkt_complementarity_scaled"] <= 3e-12
    assert result["iterations"] <= 64
    assert result["secular_evaluations"] <= 80
    _assert_finite(result)


def test_zero_radius_precedes_rank_deficiency_and_returns_json_null_multiplier():
    u, v = _rotation(0.2), _rotation(-0.4)
    sigma = np.array([2.0, 0.0])
    matrix = u @ np.diag(sigma) @ v.T
    svd = {"u": u, "s": sigma, "vh": v.T, "rank": 1, "threshold": 1e-12}
    result = solve_energy_ball(matrix, u @ np.array([1.0, 3.0]), svd, 0.0)

    assert result["status"] == "zero_baseline"
    assert result["z"] is not None and np.array_equal(result["z"], np.zeros(2))
    assert result["multiplier"] is None
    assert result["multiplier_kind"] == "zero_radius_limit"
    assert result["constraint_active"] is True
    assert result["constraint_binding"] is True
    assert result["regularization_active"] is False
    assert result["energy"] == 0
    _assert_finite(result)


def test_positive_radius_rank_deficiency_is_unresolved_without_a_candidate():
    matrix, right, svd, _ = _case(sigma=(2.0, 0.0), g=(1.0, 0.0), threshold=1e-12)
    result = solve_energy_ball(matrix, right, svd, 0.5)

    assert result["status"] == "unresolved"
    assert result["z"] is None
    assert result["multiplier"] is None
    assert result["multiplier_kind"] == "rank_unresolved"
    assert result["constraint_active"] is None
    assert result["constraint_binding"] is None
    assert result["regularization_active"] is None
    _assert_finite(result)


def test_inactive_constraint_returns_the_supplied_unconstrained_candidate_unchanged():
    matrix, right, svd, unconstrained = _case(angle_u=0.3, angle_v=-0.7)
    result = solve_energy_ball(matrix, right, svd, 3.0, unconstrained=unconstrained)

    assert result["status"] == "inactive"
    assert result["constraint_active"] is False
    assert result["constraint_binding"] is False
    assert result["regularization_active"] is False
    np.testing.assert_array_equal(result["z"], unconstrained)
    _assert_finite(result)


def test_active_anisotropic_solution_satisfies_kkt_and_is_not_radial_clipping():
    matrix, right, svd, unconstrained = _case(
        sigma=(3.0, 1.0), g=(1.0, 2.0), angle_u=0.4, angle_v=-0.6
    )
    expected = _rotation(-0.6) @ np.array([3.0 / 11.0, 2.0 / 3.0])
    radius = math.hypot(*expected)
    result = solve_energy_ball(matrix, right, svd, radius, unconstrained=unconstrained)

    _assert_active_solution(result, matrix, right, radius)
    assert result["multiplier"] == pytest.approx(2.0, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(result["z"], expected, rtol=0.0, atol=1e-11)
    radial = unconstrained * (radius / np.linalg.norm(unconstrained))
    assert np.linalg.norm(result["z"] - radial) > 1e-2


def test_rectangular_svd_orthogonal_rhs_residual_does_not_change_solution_or_multiplier():
    matrix, right, svd, unconstrained = _case(
        sigma=(3.0, 1.0), g=(1.0, 2.0), angle_u=0.4, angle_v=-0.6
    )
    expected = _rotation(-0.6) @ np.array([3.0 / 11.0, 2.0 / 3.0])
    radius = math.hypot(*expected)
    square = solve_energy_ball(matrix, right, svd, radius, unconstrained=unconstrained)
    orthogonal_residual = 3.5
    rectangular_matrix = np.vstack((matrix, np.zeros((1, 2))))
    rectangular_right = np.r_[right, orthogonal_residual]
    rectangular_svd = {**svd, "u": np.vstack((svd["u"], np.zeros((1, 2))))}

    rectangular = solve_energy_ball(rectangular_matrix, rectangular_right, rectangular_svd,
                                    radius, unconstrained=unconstrained)

    _assert_active_solution(rectangular, rectangular_matrix, rectangular_right, radius)
    np.testing.assert_allclose(rectangular["z"], square["z"], rtol=0.0, atol=1e-12)
    assert rectangular["multiplier"] == pytest.approx(square["multiplier"], rel=1e-12)
    assert rectangular["residual_norm"] == pytest.approx(
        math.hypot(square["residual_norm"], orthogonal_residual), rel=1e-12
    )
    assert rectangular["objective"] == pytest.approx(
        square["objective"] + orthogonal_residual**2 / 2, rel=1e-12
    )


def test_unconstrained_norm_exactly_on_radius_uses_zero_multiplier():
    matrix, right, svd, unconstrained = _case(sigma=(2.0, 1.0), g=(2.0, 3.0))
    radius = math.hypot(*unconstrained)
    result = solve_energy_ball(matrix, right, svd, radius, unconstrained=unconstrained)

    assert result["status"] == "inactive"
    assert result["multiplier"] == 0
    assert result["multiplier_kind"] == "finite"
    assert result["constraint_active"] is False
    assert result["constraint_binding"] is True
    assert result["regularization_active"] is False
    np.testing.assert_array_equal(result["z"], unconstrained)
    _assert_finite(result)


def test_repeated_singular_values_have_finite_active_solution():
    matrix, right, svd, unconstrained = _case(
        sigma=(2.0, 2.0), g=(4.0, 2.0), angle_u=-0.2, angle_v=0.5
    )
    radius = 1.0
    result = solve_energy_ball(matrix, right, svd, radius, unconstrained=unconstrained)

    _assert_active_solution(result, matrix, right, radius)
    assert np.linalg.norm(result["z"] / np.linalg.norm(result["z"]) -
                          unconstrained / np.linalg.norm(unconstrained)) <= 1e-12


def test_zero_rhs_returns_finite_zero_candidate_without_a_secular_search():
    matrix, right, svd, unconstrained = _case(sigma=(2.0, 1.0), g=(0.0, 0.0))
    result = solve_energy_ball(matrix, right, svd, 0.5, unconstrained=unconstrained)

    assert result["status"] == "inactive"
    assert np.array_equal(result["z"], np.zeros(2))
    assert result["multiplier"] == 0
    assert result["secular_evaluations"] == 0
    _assert_finite(result)


@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_joint_matrix_rhs_scaling_preserves_z_and_scales_multiplier_quadratically(scale):
    matrix, right, svd, unconstrained = _case(
        sigma=(3.0, 1.0), g=(1.0, 2.0), angle_u=0.4, angle_v=-0.6
    )
    expected = _rotation(-0.6) @ np.array([3.0 / 11.0, 2.0 / 3.0])
    radius = math.hypot(*expected)
    baseline = solve_energy_ball(matrix, right, svd, radius, unconstrained=unconstrained)
    scaled_svd = {**svd, "s": svd["s"] * scale, "threshold": svd["threshold"] * scale}
    scaled = solve_energy_ball(matrix * scale, right * scale, scaled_svd, radius,
                               unconstrained=unconstrained)

    _assert_active_solution(scaled, matrix * scale, right * scale, radius)
    np.testing.assert_allclose(scaled["z"], baseline["z"], rtol=1e-9, atol=1e-11)
    assert scaled["multiplier"] == pytest.approx(baseline["multiplier"] * scale**2,
                                                  rel=1e-8, abs=1e-20)


def test_high_condition_number_near_rank_threshold_fails_closed_with_bounded_diagnostics():
    sigma_min = np.nextafter(1e-12, math.inf)
    matrix, right, svd, unconstrained = _case(
        sigma=(1.0, sigma_min), g=(0.2, sigma_min * 0.4), threshold=1e-12
    )
    assert svd["rank"] == 2
    with pytest.raises(FloatingPointError) as caught:
        solve_energy_ball(matrix, right, svd, 0.3, unconstrained=unconstrained)
    diagnostics = caught.value.diagnostics
    assert diagnostics["status"] == "numerical_failure"
    assert diagnostics["z"] is None
    assert diagnostics["iterations"] <= 64
    assert diagnostics["secular_evaluations"] <= 80
    assert diagnostics["trace"]
    assert diagnostics["bracket_initial"] is not None
    _assert_finite(diagnostics)


def test_singular_value_equal_to_threshold_is_numerically_rank_deficient():
    matrix, right, svd, unconstrained = _case(
        sigma=(1.0, 1e-12), g=(0.2, 0.4e-12), threshold=1e-12
    )
    assert svd["rank"] == 1
    result = solve_energy_ball(matrix, right, svd, 0.3, unconstrained=unconstrained)

    assert result["status"] == "unresolved"
    assert result["z"] is None
    assert result["multiplier_kind"] == "rank_unresolved"


@pytest.mark.parametrize(
    ("radius", "error"),
    [(-1.0, ValueError), (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError)],
)
def test_invalid_radius_is_rejected(radius, error):
    matrix, right, svd, unconstrained = _case()
    with pytest.raises(error):
        solve_energy_ball(matrix, right, svd, radius, unconstrained=unconstrained)


@pytest.mark.parametrize(
    "svd", [
        {},
        {"u": np.eye(2), "s": np.array([1.0]), "vh": np.eye(2), "rank": 1, "threshold": 0.0},
        {"u": np.eye(2), "s": np.array([1.0, float("nan")]), "vh": np.eye(2), "rank": 2, "threshold": 0.0},
        {"u": np.eye(2), "s": np.array([1.0, -0.1]), "vh": np.eye(2), "rank": 2, "threshold": 0.0},
        {"u": np.eye(2), "s": np.array([1.0, 0.1]), "vh": np.eye(2), "rank": True, "threshold": 0.0},
        {"u": np.eye(2), "s": np.array([1.0, 0.1]), "vh": np.eye(2), "rank": 2, "threshold": -1.0},
    ],
)
def test_invalid_saved_svd_metadata_is_rejected(svd):
    matrix, right, _, unconstrained = _case()
    with pytest.raises((ValueError, TypeError)):
        solve_energy_ball(matrix, right, svd, 0.5, unconstrained=unconstrained)


def test_nonfinite_svd_validation_arithmetic_fails_with_diagnostics():
    matrix, right, svd, _ = _case(sigma=(1.0, 1.0), g=(0.1, 0.2))
    svd["u"] = np.eye(2) * 1e308

    with pytest.raises(FloatingPointError) as caught:
        solve_energy_ball(matrix, right, svd, 0.5)
    assert caught.value.diagnostics["status"] == "numerical_failure"
    assert caught.value.diagnostics["z"] is None
    _assert_finite(caught.value.diagnostics)


def test_overflowing_residual_objective_fails_closed():
    matrix = np.eye(2)
    right = np.array([1e308, 1e308])
    svd = {"u": np.eye(2), "s": np.ones(2), "vh": np.eye(2), "rank": 2, "threshold": 0.0}

    with pytest.raises(FloatingPointError) as caught:
        solve_energy_ball(matrix, right, svd, 0.0)
    assert caught.value.diagnostics["status"] == "numerical_failure"
    assert caught.value.diagnostics["z"] is None
    _assert_finite(caught.value.diagnostics)


def test_wrong_saved_unconstrained_representative_is_rejected():
    matrix, right, svd, unconstrained = _case(angle_u=0.3, angle_v=-0.7)
    wrong = unconstrained + np.array([0.1, 0.0])

    with pytest.raises(ValueError, match="unconstrained representative"):
        solve_energy_ball(matrix, right, svd, 0.5, unconstrained=wrong)
