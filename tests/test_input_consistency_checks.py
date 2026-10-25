"""Constructed A17 numerical-check examples; no real files, models or RHS."""
import json
import math

import numpy as np
import pytest

from kron_reduce.evaluation import input_consistency_checks as checks


def factors(singular=(3.0, 2.0)):
    u = np.zeros((100, 2))
    u[0, 0] = u[1, 1] = 1.0
    s = np.array(singular)
    vh = np.array([[.6, .8], [-.8, .6]])
    return (u * s) @ vh, u, s, vh


def forbidden(*args, **kwargs):
    raise AssertionError("forbidden solver/decomposition/file operation")


def assert_counts(result, *, products, calls, matrix=0, vector=0):
    counts = result["operation_counts"]
    for name, expected in [("matmul", products), ("norm_calls", calls),
                           ("matrix_norms", matrix), ("vector_norms", vector)]:
        assert counts[name + "_attempted"] == counts[name + "_returned"] == expected


def test_nontrivial_saved_svd_and_exact_operation_counts(monkeypatch):
    b, u, s, vh = factors()
    for name in ("svd", "qr", "eig", "eigh", "eigvalsh", "solve", "lstsq", "inv", "pinv"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    dispatched = []

    def dispatch(category, function):
        dispatched.append(category)
        return function()

    result = checks.check_decomposition(B=b, U=u, singular=s, Vh=vh, dispatch=dispatch)
    assert result["qualified"]
    assert dispatched == ["decomposition_check"]
    assert_counts(result, products=3, calls=4, matrix=4)
    changed_u, changed_vh = u.copy(), vh.copy()
    changed_u[:, 0] *= -1
    changed_vh[0] *= -1
    assert checks.check_decomposition(B=b, U=changed_u, singular=s, Vh=changed_vh)["qualified"]


@pytest.mark.parametrize("corrupt", ["U", "Vh", "B"])
def test_saved_svd_corruptions_are_rejected(corrupt):
    b, u, s, vh = factors()
    args = {"B": b, "U": u, "singular": s, "Vh": vh}
    args[corrupt] = args[corrupt].copy()
    args[corrupt][0, 0] += .01
    result = checks.check_decomposition(**args)
    assert not result["qualified"]
    assert result["reasons"]


def test_nonzero_fit_residual_can_be_a_valid_ls_solution():
    b = np.eye(100, 2)
    rhs = np.zeros(100)
    rhs[7] = 5
    result = checks.check_ls_solution(B=b, rhs=rhs, z0=np.zeros(2),
                                      determinant_lower=1., trace_upper=2.)
    assert result["qualified"]
    assert result["arrays"]["residual"][7] == 5
    assert result["values"]["normal_residual_norm"] == 0
    assert result["values"]["conditioning_proxy"] > 0
    assert_counts(result, products=2, calls=4, matrix=1, vector=3)


def test_small_normal_residual_does_not_hide_ill_conditioning():
    b = np.eye(100, 2)
    b[:, 1] *= 1e-7
    result = checks.check_ls_solution(B=b, rhs=np.zeros(100), z0=np.array([0., 1.]),
                                      determinant_lower=1e-14, trace_upper=1.00000000000001)
    assert result["gates"]["normal_residual"]
    assert not result["gates"]["conditioning_proxy"]
    assert not result["qualified"]


def test_absolute_scale_proxy_may_reject_well_conditioned_small_matrix():
    result = checks.check_ls_solution(B=np.eye(100, 2) * 1e-4, rhs=np.zeros(100),
                                      z0=np.zeros(2), determinant_lower=1e-16, trace_upper=2e-8)
    assert result["gates"]["normal_residual"]
    assert not result["gates"]["conditioning_proxy"]


def test_normal_residual_boundary_has_no_isclose_allowance():
    def run(value):
        return checks.check_ls_solution(B=np.eye(100, 2), rhs=np.zeros(100),
                                        z0=np.array([value, 0.]), determinant_lower=1., trace_upper=2.)

    assert run(checks.TAU_NUM)["gates"]["normal_residual"]
    assert not run(np.nextafter(checks.TAU_NUM, np.inf))["gates"]["normal_residual"]


@pytest.mark.parametrize("det,trace", [(0., 1.), (-1., 1.), (1., 0.), (1., -1.),
                                      (math.nan, 1.), (1., math.inf),
                                      (np.nextafter(0., 1.), 1.), (1e308, 1e-308)])
def test_unusable_gram_bounds_never_reach_matrix_operations(det, trace):
    result = checks.check_ls_solution(B=np.eye(100, 2), rhs=np.zeros(100), z0=np.zeros(2),
                                      determinant_lower=det, trace_upper=trace)
    assert not result["qualified"]
    assert_counts(result, products=0, calls=0)
    json.dumps(result["values"], allow_nan=False)


def test_no_rhs_broadcast_and_no_nonfinite_normal_error_hiding():
    bad_shape = checks.check_ls_solution(B=np.eye(100, 2), rhs=np.zeros((100, 1)),
                                        z0=np.zeros(2), determinant_lower=1., trace_upper=2.)
    assert not bad_shape["qualified"]
    b = np.zeros((100, 2))
    b[0] = 1e308
    result = checks.check_ls_solution(B=b, rhs=np.zeros(100), z0=np.ones(2),
                                      determinant_lower=1., trace_upper=2.)
    assert not result["qualified"]
    assert np.isinf(result["arrays"]["residual"][0])
    assert_counts(result, products=1, calls=0)


def quadrature(**updates):
    args = {"zh2": np.zeros(2), "zh4": np.zeros(2), "zTobs2": np.zeros(2),
            "zTobs4": np.zeros(2), "PhiT": np.eye(2), "wT": np.zeros(2), "z0": np.zeros(2)}
    args.update(updates)
    return checks.check_inherited_quadrature(**args)


def test_quadrature_is_one_map_with_no_second_solve(monkeypatch):
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    result = quadrature(PhiT=np.array([[2., 1.], [-1., 3.]]),
                        wT=np.array([1., -2.]), z0=np.array([4., 5.]))
    assert result["qualified"]
    assert np.array_equal(result["arrays"]["observed_driven_endpoint"], [14., 9.])
    assert result["endpoint_for_diagnostics_only"]
    assert_counts(result, products=1, calls=4, vector=4)


def test_quadrature_rechecks_new_scale_and_endpoint_independently():
    shrinking = quadrature(zh4=np.array([1e6, 0.]), zh2=np.array([1e6 + .001, 0.]))
    assert not shrinking["gates"]["inherited_initial_quadrature"]
    cancelling = quadrature(zh4=np.array([1e4, 0.]), zh2=np.array([1e4 + 5e-5, 0.]),
                            zTobs2=np.array([5e-5, 0.]), z0=np.array([100., 0.]),
                            wT=np.array([-100., 0.]))
    assert cancelling["gates"]["inherited_initial_quadrature"]
    assert not cancelling["gates"]["inherited_endpoint_quadrature"]


def test_quadrature_map_overflow_is_retained_as_failure():
    result = quadrature(PhiT=np.full((2, 2), 1e308), z0=np.ones(2))
    assert not result["qualified"]
    assert np.isinf(result["arrays"]["observed_driven_endpoint"]).all()
    assert_counts(result, products=1, calls=0)


def test_refinement_uses_pointwise_visible_and_two_channel_hidden_norms():
    fine = np.zeros((11, 22))
    coarse = fine.copy()
    coarse[5, 3] = np.nextafter(1e-5, np.inf)
    visible = checks.check_refinement(coarse, fine, np.ones(20))
    assert not visible["gates"]["visible_refinement"]
    coarse[:] = 0
    coarse[4, 20:] = .8e-5
    hidden = checks.check_refinement(coarse, fine, np.ones(20))
    assert not hidden["gates"]["hidden_refinement"]
    assert_counts(hidden, products=0, calls=2, vector=22)


def test_refinement_includes_origin_and_exact_boundary():
    fine = np.zeros((2, 22))
    coarse = fine.copy()
    coarse[0, 20] = 1e-5
    assert checks.check_refinement(coarse, fine, np.ones(20))["qualified"]
    coarse[0, 20] = np.nextafter(1e-5, np.inf)
    assert not checks.check_refinement(coarse, fine, np.ones(20))["qualified"]


def test_scaled_norms_do_not_square_underflow_or_overflow():
    fine = np.zeros((2, 22))
    fine[:, 20:] = 1e308
    result = checks.check_refinement(fine, fine.copy(), np.ones(20))
    assert result["qualified"]
    assert math.isfinite(result["values"]["fine_hidden_max_norm"])
    tiny = np.zeros((2, 22))
    tiny[0, 20] = 1e-200
    result = checks.check_refinement(tiny, np.zeros_like(tiny), np.ones(20))
    assert result["values"]["hidden_max_norm"] == 1e-200


@pytest.mark.parametrize("invalid", ["scale", "nan", "shape"])
def test_refinement_invalid_input_fails_before_norms(invalid):
    a, b, scale = np.zeros((3, 22)), np.zeros((3, 22)), np.ones(20)
    if invalid == "scale":
        scale[3] = 0
    elif invalid == "nan":
        a[1, 5] = np.nan
    else:
        b = b[:, :21]
    result = checks.check_refinement(a, b, scale)
    assert not result["qualified"]
    assert_counts(result, products=0, calls=0)


def physical(**updates):
    record = {"energy_rate": 0., "supply_minus_dissipation": 0., "identity_error": 0.,
              "q_mass_coi": 0., "v_mass_coi": 0.}
    record.update(updates)
    return record


def test_physical_judgement_counts_and_exact_thresholds():
    records = [physical(energy_rate=1e-8, identity_error=1e-8, q_mass_coi=1e-10), physical()]
    result = checks.judge_physical_checks(records, expected_count=2)
    assert result["qualified"]
    assert result["valid_point_count"] == 2
    assert_counts(result, products=0, calls=0)
    records[1]["v_mass_coi"] = np.nextafter(1e-10, np.inf)
    assert not checks.judge_physical_checks(records, expected_count=2)["qualified"]


@pytest.mark.parametrize("record", [physical(energy_rate=np.nan), physical(q_mass_coi=-1.),
                                    physical(energy_rate=1., identity_error=0.), {},
                                    physical(energy_rate=True),
                                    physical(energy_rate=1e308, supply_minus_dissipation=-1e308)])
def test_invalid_physical_values_cannot_be_hidden_by_max(record):
    result = checks.judge_physical_checks([physical(), record], expected_count=2)
    assert not result["qualified"]
    assert result["valid_point_count"] == 1
    assert result["failed_points"][0]["index"] == 1
    json.dumps(result, allow_nan=False)


def test_missing_physics_and_dispatch_failures_are_not_success():
    assert not checks.judge_physical_checks([], expected_count=1)["qualified"]
    b, u, s, vh = factors()

    def failed_dispatch(category, function):
        raise OSError("durable reservation unavailable")

    with pytest.raises(OSError, match="durable reservation"):
        checks.check_decomposition(B=b, U=u, singular=s, Vh=vh, dispatch=failed_dispatch)
