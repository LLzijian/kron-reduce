"""Independent exact-rational A16 oracles; only constructed synthetic arrays."""
from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np
import pytest

from kron_reduce.evaluation import pm_impulse_audit as k


def exact(value):
    return F.from_float(float(value))


def encloses(dual, truth):
    assert exact(dual.lower) <= truth <= exact(dual.upper), (dual, truth)


def encloses_sqrt(dual, square):
    assert dual.lower >= 0.0
    assert exact(dual.lower) ** 2 <= square <= exact(dual.upper) ** 2


def ledger(kind, **kwargs):
    return k.ScalarLedger(k.expected_budget(kind, **kwargs))


def assert_exact_count(book, kind, **kwargs):
    expected = k.expected_budget(kind, **kwargs)
    assert dict(book.attempted) == expected
    assert book.attempted == book.returned
    assert book.closed


def assert_bounded(book, kind):
    maximum = k.expected_budget(kind)
    assert all(n <= maximum.get(key, 0) for key, n in book.attempted.items())
    assert all(n <= book.attempted[key] for key, n in book.returned.items())


def impulse_from_known(d):
    values = [k.Dual.exact(float(x)) for x in np.asarray(d).reshape(100)]
    zero = k.Dual.exact(0.0)
    return k.HistoryImpulse(tuple((zero,) * 10 for _ in range(11)),
                            tuple(tuple(values[10*j:10*j+10]) for j in range(10)),
                            (zero,) * 10, {"qualified": True}, {})


def slot_inputs():
    matrix = np.zeros((100, 2))
    matrix[0, 0] = matrix[1, 1] = 1.0
    return {"Ltilde": matrix, "c": np.ones(10), "S": np.ones(10),
            "g4": np.zeros(100), "btilde4": np.zeros(100),
            "dynamic_residual_scaled4": np.zeros(100),
            "force_residual4": np.zeros((10, 10)),
            "force_residual_projector": np.eye(10)}


def exact_slot_oracle(d, args):
    """Exact linear-algebra identities, independent of scalar execution order."""
    dw = [exact(d[j*10+i]) * exact(args["c"][j]) * exact(args["S"][i])
          for j in range(10) for i in range(10)]
    b = [[exact(v) for v in row] for row in args["Ltilde"]]
    gram = [[sum(row[j] * row[l] for row in b) for l in range(2)] for j in range(2)]
    a = [sum(b[i][j] * dw[i] for i in range(100)) for j in range(2)]
    det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    r = [exact(v) for v in args["dynamic_residual_scaled4"]]
    dyn0 = sum(x*x for x in r)
    dyn1 = sum((x+y)**2 for x, y in zip(r, dw, strict=True))
    rc = [[exact(v) for v in row] for row in args["force_residual_projector"]]
    fd = [sum(rc[i][l]*exact(d[j*10+l]) for l in range(10))
          for j in range(10) for i in range(10)]
    f = [exact(v) for v in args["force_residual4"].flat]
    fw0 = [f[j*10+i]*exact(args["c"][j])*exact(args["S"][i])
           for j in range(10) for i in range(10)]
    fw1 = [(f[j*10+i]+fd[j*10+i])*exact(args["c"][j])*exact(args["S"][i])
           for j in range(10) for i in range(10)]
    force0, force1 = sum(v*v for v in fw0), sum(v*v for v in fw1)
    return {"a": a, "gram": gram, "det": det, "trace": gram[0][0]+gram[1][1],
            "dynamic_before_sq": dyn0, "dynamic_after_sq": dyn1,
            "dynamic_delta_direct": dyn1-dyn0, "dynamic_delta_cross": dyn1-dyn0,
            "force_before_sq": force0, "force_after_sq": force1,
            "force_delta_direct": force1-force0, "force_delta_cross": force1-force0}


def check_slot_oracle(result, oracle):
    out = result.values
    for j in range(2):
        encloses(out["normal_rhs"][j], oracle["a"][j])
        for l in range(2):
            encloses(out["gram"][j][l], oracle["gram"][j][l])
    encloses(out["determinant"], oracle["det"])
    encloses(out["trace"], oracle["trace"])
    for name in ("dynamic_before_sq", "dynamic_after_sq", "dynamic_delta_direct",
                 "dynamic_delta_cross", "force_before_sq", "force_after_sq",
                 "force_delta_direct", "force_delta_cross"):
        encloses(out[name], oracle[name])
    encloses_sqrt(out["normal_rhs_norm"], sum(x*x for x in oracle["a"]))
    if out["sensitivity_bounds"] is not None:
        # Exact 2x2 solution is used only by this synthetic oracle, never by the kernel.
        g, a, det = oracle["gram"], oracle["a"], oracle["det"]
        x = (g[1][1]*a[0] - g[0][1]*a[1])/det
        y = (g[0][0]*a[1] - g[1][0]*a[0])/det
        bounds = out["sensitivity_bounds"]
        assert exact(bounds["delta_norm_lower"])**2 <= x*x+y*y
        assert x*x+y*y <= exact(bounds["delta_norm_upper"])**2


def test_nonuniform_mass_clock_source_and_impulse_exact_oracle_and_counts():
    mass = np.arange(1.0, 11.0)
    prior_book = ledger("source")
    prior = k.prepare_source(mass*4.0, 4.0, np.arange(10.0)/8,
                             ledger=prior_book)
    assert_exact_count(prior_book, "source")
    for got, want in zip(prior.mass, mass, strict=True):
        encloses(got, exact(want))
    pm = np.tile(np.arange(10.0)/8, (11, 1))
    pm[:, 0] += np.arange(11.0)
    p = np.eye(10) - mass[:, None]/55.0
    h = np.array([.0092, .0105, .0087, .0111, .0098, .0102, .0089, .0113, .0094, .0109])
    origin = np.array([float(10*(F(int(i == 0))-F(i+1, 55))) for i in range(10)])
    book = ledger("impulse")
    result = k.build_history_impulse(prior, pm, origin, h, p, ledger=book)
    assert_exact_count(book, "impulse")
    assert result.anchor["qualified"]
    u = [[F(j)*(F(int(i == 0))-F(i+1, 55)) for i in range(10)] for j in range(11)]
    for j in range(11):
        for i in range(10):
            encloses(result.u11[j][i], u[j][i])
    for j in range(10):
        before_p = [exact(h[j])*((u[j][i]+u[j+1][i])/2-exact(origin[i])) for i in range(10)]
        for i in range(10):
            want = sum(exact(p[i, l])*before_p[l] for l in range(10))
            encloses(result.d[j][i], want)


def test_common_injection_is_zero_and_origin_mismatch_is_global():
    prior = k.prepare_source(np.ones(10), 1.0, np.zeros(10), ledger=ledger("source"))
    pm = np.repeat(np.arange(11.0)[:, None], 10, axis=1)
    result = k.build_history_impulse(prior, pm, np.zeros(10), np.full(10, .01),
                                     np.eye(10)-.1, ledger=ledger("impulse"))
    for row in result.d:
        for value in row:
            encloses(value, F(0))
    book = ledger("impulse")
    with pytest.raises(k.IdentityError) as caught:
        k.build_history_impulse(prior, pm, np.full(10, 1e-6), np.full(10, .01),
                                 np.eye(10)-.1, ledger=book)
    assert caught.value.partial_record["anchor"]["qualified"] is False
    assert "d" not in caught.value.partial_record
    assert_bounded(book, "impulse")
    assert book.closed


@pytest.mark.parametrize("coupling_scale", [1.0, 2.0**-20])
def test_well_conditioned_oracle_absolute_gain_not_condition(coupling_scale):
    args = slot_inputs()
    args["Ltilde"] *= coupling_scale
    d = np.zeros(100)
    d[:3] = [3, -4, 12]
    args["dynamic_residual_scaled4"][:3] = [-2, 4, -1]
    book = ledger("slot")
    result = k.audit_slot(impulse_from_known(d), **args, ledger=book)
    assert result.complete and result.status == "resolved_nonzero"
    assert_exact_count(book, "slot")
    check_slot_oracle(result, exact_slot_oracle(d, args))
    bounds = result.values["sensitivity_bounds"]
    assert bounds["delta_norm_lower"] <= 5/coupling_scale <= bounds["delta_norm_upper"]
    assert exact(bounds["sigma_min_lower"])**2 <= exact(coupling_scale)**2
    assert exact(bounds["sigma_max_upper"])**2 >= exact(coupling_scale)**2
    assert 1 <= bounds["condition_upper"] < 2.001


@pytest.mark.parametrize("almost", [False, True])
def test_rank_unconfirmed_never_invents_unique_state(almost):
    args = slot_inputs()
    args["Ltilde"][:] = 0.0
    args["Ltilde"][0] = [1, 1 if almost else 2]
    if almost:
        args["Ltilde"][1, 1] = 2.0**-27
    d = np.zeros(100)
    d[1] = 2.0**-27 if almost else 1.0
    book = ledger("slot")
    result = k.audit_slot(impulse_from_known(d), **args, ledger=book)
    assert result.complete and result.status == "numerically_unresolved"
    assert result.values["sensitivity_bounds"] is None
    assert_exact_count(book, "slot", resolved=False)
    oracle = exact_slot_oracle(d, args)
    check_slot_oracle(result, oracle)
    assert oracle["det"] == (F(1, 2**54) if almost else 0)


def test_strong_cancellation_normal_rhs_exactly_one_is_enclosed():
    args = slot_inputs()
    args["Ltilde"][:] = 0.0
    args["Ltilde"][:3, 0] = 1.0
    args["Ltilde"][3, 1] = 1.0
    d = np.zeros(100)
    d[:3] = [1e16, 1, -1e16]
    result = k.audit_slot(impulse_from_known(d), **args, ledger=ledger("slot"))
    oracle = exact_slot_oracle(d, args)
    assert oracle["a"][0] == 1
    assert result.values["normal_rhs"][0].point == 0.0
    assert result.status == "rounding_unresolved"
    check_slot_oracle(result, oracle)


def test_port_projection_before_scaling_and_cross_identity():
    args = slot_inputs()
    args["S"] = np.arange(1.0, 11.0)
    port = np.zeros(10)
    port[:2] = [1, -1]
    rc = np.eye(10)-np.outer(port, port)/2
    args["force_residual_projector"] = rc
    g = np.zeros(10)
    g[[0, 2]] = [1, -1]
    args["force_residual4"][0] = rc @ g
    d = np.zeros(100)
    d[:10] = -2*g
    result = k.audit_slot(impulse_from_known(d), **args, ledger=ledger("slot"))
    oracle = exact_slot_oracle(d, args)
    assert oracle["force_before_sq"] == F(41, 4)
    assert oracle["force_delta_direct"] == 0
    check_slot_oracle(result, oracle)
    encloses(result.values["force_impulse_sq"], F(41))


def test_dense_all_rows_nonuniform_scales_match_exact_rational_identities():
    args = slot_inputs()
    for i in range(100):
        args["Ltilde"][i] = [(i % 7-3)/8, ((3*i) % 11-5)/16]
        args["dynamic_residual_scaled4"][i] = (i % 5-2)/32
        args["force_residual4"].flat[i] = (i % 9-4)/16
    args["c"] = np.arange(1.0, 11.0)/16
    args["S"] = np.arange(2.0, 12.0)/8
    args["force_residual_projector"][:2, :2] = .5
    d = np.array([((13*i) % 17-8)/32 for i in range(100)])
    book = ledger("slot")
    result = k.audit_slot(impulse_from_known(d), **args, ledger=book)
    assert result.complete and result.status == "resolved_nonzero"
    assert_exact_count(book, "slot")
    check_slot_oracle(result, exact_slot_oracle(d, args))


def test_subnormal_square_and_sqrt_do_not_flush_to_zero():
    tiny = math.nextafter(0.0, 1.0)
    assert tiny != 0.0 and tiny+tiny != tiny
    book = ledger("slot")
    e = k._Arithmetic(book)
    value = k.Dual.exact(tiny)
    encloses(e.add(value, value), exact(tiny)*2)
    square = e.square(value)
    assert square.point == 0.0
    encloses(square, exact(tiny)**2)
    encloses_sqrt(e.sqrt(square, nonnegative_sum=True), exact(tiny)**2)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_input_rejected_before_arithmetic(bad):
    args = slot_inputs()
    args["Ltilde"][0, 0] = bad
    book = ledger("slot")
    with pytest.raises(k.InputError):
        k.audit_slot(impulse_from_known(np.zeros(100)), **args, ledger=book)
    assert book.closed and dict(book.attempted) == {"input.finite": 1}


def test_overflow_keeps_gram_prefix_and_stops_without_retry():
    args = slot_inputs()
    args["Ltilde"][0, 0] = 1e308
    book = ledger("slot")
    result = k.audit_slot(impulse_from_known(np.zeros(100)), **args, ledger=book)
    assert result.status == "numerical_failure" and not result.complete
    assert "weighted_impulse" in result.values and "gram" not in result.values
    assert math.isinf(result.counts["last_numeric_return"]["value"])
    assert_bounded(book, "slot")
    with pytest.raises(k.BudgetError):
        k.audit_slot(impulse_from_known(np.zeros(100)), **args, ledger=book)


def test_budget_refusal_precedes_operation_and_no_blas(monkeypatch):
    args = slot_inputs()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("BLAS/decomposition/solve is forbidden in the kernel")
    for name in ("dot", "matmul"):
        monkeypatch.setattr(np, name, forbidden)
    for name in ("svd", "qr", "solve", "eig", "eigh", "inv"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    book = ledger("slot")
    result = k.audit_slot(impulse_from_known(np.zeros(100)), **args, ledger=book)
    assert result.complete and result.status == "rounding_unresolved"
    limits = k.expected_budget("source")
    limits["point.div"] = 0
    book = k.ScalarLedger(limits)
    with pytest.raises(k.BudgetError):
        k.prepare_source(np.ones(10), 1.0, np.zeros(10), ledger=book)
    assert book.attempted["point.div"] == book.returned["point.div"] == 0
    assert book.closed


def test_zero_divisor_and_overflow_are_distinct_counted_failures():
    e = k._Arithmetic(ledger("slot"))
    with pytest.raises(k.NumericalError, match="contains zero"):
        e.div(k.Dual.exact(1.0), k.Dual(1.0, -1.0, 1.0))
    assert e.ledger.attempted["point.div"] == 0
    e = k._Arithmetic(ledger("slot"))
    with pytest.raises(k.NumericalError, match="nonfinite"):
        e.div(k.Dual.exact(1.0), k.Dual.exact(math.nextafter(0.0, 1.0)))
    assert e.ledger.returned["point.div"] == 1


@pytest.mark.parametrize("missing", [0, 1, 258])
def test_original_weight_aggregation_no_success_renormalization(missing):
    values = {name: k.Dual.exact(2.0) for name in k.SUMMARY_FIELDS}
    good = k.SlotResult("resolved_nonzero", True, values, {})
    bad = k.SlotResult("numerical_failure", False, {}, {}, "synthetic")
    rows = [bad]*missing + [good]*(258-missing)
    weights = np.full(258, 1.0/258)
    book = ledger("aggregate")
    result = k.aggregate_slots(rows, weights, ledger=book)
    assert_exact_count(book, "aggregate", completed_slots=258-missing)
    encloses(result.values["total_weight"], exact(weights[0])*258)
    encloses(result.values["missing_weight"], exact(weights[0])*missing)
    for value in result.values["partial_contributions"].values():
        encloses(value, 2*exact(weights[0])*(258-missing))
    assert (result.values["population_metrics"] is None) == (missing > 0)
    assert result.complete == (missing == 0)


def test_budget_function_is_integer_only_and_shapes_are_fixed(monkeypatch):
    monkeypatch.setattr(math, "sqrt", lambda *_: pytest.fail("budget must not run arithmetic"))
    for kind in ("source", "impulse", "slot", "aggregate"):
        assert all(type(v) is int and v > 0 for v in k.expected_budget(kind).values())
    with pytest.raises(k.InputError):
        k.prepare_source(np.ones(9), 1, np.zeros(10), ledger=ledger("source"))


@pytest.mark.parametrize("values", [np.ones(10, dtype=complex), ["1"]*10])
def test_input_conversion_does_not_silently_discard_information(values):
    book = ledger("source")
    with pytest.raises(k.InputError):
        k.prepare_source(values, 1.0, np.zeros(10), ledger=book)
    assert book.closed and not book.attempted
