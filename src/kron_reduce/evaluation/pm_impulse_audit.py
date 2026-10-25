"""A16 fixed-order scalar algebra; no IO, model, integration, or state solve.

Point and outward-rounded interval channels are both counted. The caller must
durably reserve an entire invocation before dispatch, then persist its outputs
and counts. An interrupted invocation consumes that reservation without retry.
"""
from __future__ import annotations

import math
import struct
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class BudgetError(RuntimeError):
    """An operation was refused before execution; stop the outer audit."""


class InputError(ValueError):
    """The declared input contract is invalid."""

    def __init__(self, message: str):
        super().__init__(message)
        self.partial_record: dict[str, Any] = {}
        self.counts: dict[str, Any] = {}


class IdentityError(InputError):
    """The source origin differs from the fixed saved input; global stop."""


class NumericalError(ArithmeticError):
    """Keep the completed prefix and consume this invocation's reservation."""

    def __init__(self, message: str):
        super().__init__(message)
        self.partial_record: dict[str, Any] = {}
        self.counts: dict[str, Any] = {}


@dataclass
class ScalarLedger:
    """One invocation, exact in-memory transitions, caller-supplied hard limits."""

    limits: Mapping[str, int]
    attempted: Counter = field(default_factory=Counter, init=False)
    returned: Counter = field(default_factory=Counter, init=False)
    closed: bool = field(default=False, init=False)
    last_numeric_return: dict | None = field(default=None, init=False)

    def __post_init__(self):
        self.limits = dict(self.limits)
        if any(type(n) is not int or n < 0 for n in self.limits.values()):
            raise InputError("limits must be nonnegative integer counts")

    def call(self, key: str, function: Callable[[], Any]):
        if self.closed:
            raise BudgetError("ledger is closed; an invocation cannot be retried")
        if self.attempted[key] >= self.limits.get(key, 0):
            self.closed = True
            raise BudgetError(f"scalar budget exhausted before {key}")
        self.attempted[key] += 1
        value = function()
        self.returned[key] += 1
        return value

    def snapshot(self):
        return {"attempted": dict(sorted(self.attempted.items())),
                "returned": dict(sorted(self.returned.items())), "closed": self.closed,
                "last_numeric_return": self.last_numeric_return}


@dataclass(frozen=True)
class Dual:
    point: float
    lower: float
    upper: float

    @classmethod
    def exact(cls, value: float):
        return cls(value, value, value)

    def record(self):
        return {"point": self.point, "lower": self.lower, "upper": self.upper}


class _Arithmetic:
    def __init__(self, ledger: ScalarLedger):
        self.ledger = ledger

    def compare(self, channel, left, right, relation):
        return self.ledger.call(f"{channel}.compare", lambda: relation(left, right))

    def _number(self, channel, kind, function):
        try:
            value = self.ledger.call(f"{channel}.{kind}", function)
        except (OverflowError, ZeroDivisionError, ValueError) as error:
            raise NumericalError(f"{channel}.{kind}: {error}") from error
        self.ledger.last_numeric_return = {"key": f"{channel}.{kind}", "value": value}
        valid = self.ledger.call(f"{channel}.finite", lambda: math.isfinite(value))
        if not valid:
            raise NumericalError(f"nonfinite {channel}.{kind} return")
        return value

    def _out(self, lower, upper):
        return (self._number("interval", "nextafter", lambda: math.nextafter(lower, -math.inf)),
                self._number("interval", "nextafter", lambda: math.nextafter(upper, math.inf)))

    def _min(self, a, b):
        return a if self.compare("interval", a, b, lambda x, y: x <= y) else b

    def _max(self, a, b):
        return a if self.compare("interval", a, b, lambda x, y: x >= y) else b

    def add(self, a, b):
        p = self._number("point", "add", lambda: a.point + b.point)
        low = self._number("interval", "add", lambda: a.lower + b.lower)
        high = self._number("interval", "add", lambda: a.upper + b.upper)
        return Dual(p, *self._out(low, high))

    def sub(self, a, b):
        p = self._number("point", "sub", lambda: a.point - b.point)
        low = self._number("interval", "sub", lambda: a.lower - b.upper)
        high = self._number("interval", "sub", lambda: a.upper - b.lower)
        return Dual(p, *self._out(low, high))

    def mul(self, a, b):
        p = self._number("point", "mul", lambda: a.point * b.point)
        corners = [self._number("interval", "mul", lambda x=x, y=y: x * y)
                   for x in (a.lower, a.upper) for y in (b.lower, b.upper)]
        low = high = corners[0]
        for value in corners[1:]:
            low, high = self._min(low, value), self._max(high, value)
        return Dual(p, *self._out(low, high))

    def div(self, a, b):
        point_zero = self.compare("point", b.point, 0.0, lambda x, y: x == y)
        lower_zero = self.compare("interval", b.lower, 0.0, lambda x, y: x <= y)
        upper_zero = self.compare("interval", b.upper, 0.0, lambda x, y: x >= y)
        if point_zero or (lower_zero and upper_zero):
            raise NumericalError("division denominator interval contains zero")
        p = self._number("point", "div", lambda: a.point / b.point)
        corners = [self._number("interval", "div", lambda x=x, y=y: x / y)
                   for x in (a.lower, a.upper) for y in (b.lower, b.upper)]
        low = high = corners[0]
        for value in corners[1:]:
            low, high = self._min(low, value), self._max(high, value)
        return Dual(p, *self._out(low, high))

    def square(self, a):
        p = self._number("point", "mul", lambda: a.point * a.point)
        left = self._number("interval", "mul", lambda: a.lower * a.lower)
        right = self._number("interval", "mul", lambda: a.upper * a.upper)
        low, high = self._min(left, right), self._max(left, right)
        below = self.compare("interval", a.lower, 0.0, lambda x, y: x <= y)
        above = self.compare("interval", a.upper, 0.0, lambda x, y: x >= y)
        if below and above:
            low = 0.0
        return Dual(p, *self._out(low, high))

    def sqrt(self, a, *, nonnegative_sum=False):
        low, high = a.lower, a.upper
        if nonnegative_sum:
            # Only used on a declared sum of squares, never an arbitrary interval.
            low, high = self._max(0.0, low), self._max(0.0, high)
        negative_point = self.compare("point", a.point, 0.0, lambda x, y: x < y)
        negative_lower = self.compare("interval", low, 0.0, lambda x, y: x < y)
        if negative_point or negative_lower:
            raise NumericalError("negative square-root argument")
        p = self._number("point", "sqrt", lambda: math.sqrt(a.point))
        lo = self._number("interval", "sqrt", lambda: math.sqrt(low))
        hi = self._number("interval", "sqrt", lambda: math.sqrt(high))
        lo, hi = self._out(lo, hi)
        # The mathematical sqrt is nonnegative, including subnormal/zero cases.
        lo = self._max(0.0, lo)
        return Dual(p, lo, hi)

    def total(self, values):
        result = Dual.exact(0.0)
        for value in values:
            result = self.add(result, value)
        return result

    def dot(self, left, right):
        return self.total(self.mul(a, b) for a, b in zip(left, right, strict=True))

    def squared_norm(self, values):
        return self.total(self.square(value) for value in values)


def _array(values, shape, ledger, name, *, positive=False):
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "fiu":
            raise InputError(f"{name}: expected real numeric inputs")
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError(f"{name}: invalid real numeric input: {error}") from error
    if array.shape != shape:
        raise InputError(f"{name}: expected shape {shape}, got {array.shape}")
    result = []
    for number in array.flat:
        value = float(number)
        valid = ledger.call("input.finite", lambda value=value: math.isfinite(value))
        if not valid:
            raise InputError(f"{name}: nonfinite input")
        if positive:
            valid = ledger.call("input.positive", lambda value=value: value > 0.0)
            if not valid:
                raise InputError(f"{name}: input must be positive")
        result.append(Dual.exact(value))
    return tuple(result)


def _begin(ledger):
    if ledger.closed or ledger.attempted:
        raise BudgetError("a new invocation requires a fresh reserved ledger")
    return _Arithmetic(ledger)


def _raise_with_prefix(error, partial, ledger):
    ledger.closed = True
    if isinstance(error, (NumericalError, InputError)):
        error.partial_record = partial
        error.counts = ledger.snapshot()
    raise error


@dataclass(frozen=True)
class SourcePrior:
    mass: tuple[Dual, ...]
    equilibrium_pm: tuple[Dual, ...]
    mass_sum: Dual
    counts: dict


@dataclass(frozen=True)
class HistoryImpulse:
    u11: tuple[tuple[Dual, ...], ...]
    d: tuple[tuple[Dual, ...], ...]
    origin_difference: tuple[Dual, ...]
    anchor: dict
    counts: dict


def prepare_source(inertia, omega_s, equilibrium_pm, *, ledger):
    """Original source mass=inertia/omega_s; no invented factor two."""
    e, partial = _begin(ledger), {}
    try:
        raw = _array(inertia, (10,), ledger, "inertia", positive=True)
        omega = _array([omega_s], (1,), ledger, "omega_s", positive=True)[0]
        equilibrium = _array(equilibrium_pm, (10,), ledger, "equilibrium_pm")
        mass = []
        partial["mass"] = mass
        for value in raw:
            mass.append(e.div(value, omega))
        mass_sum = e.total(mass)
    except (NumericalError, InputError) as error:
        _raise_with_prefix(error, partial, ledger)
    finally:
        ledger.closed = True
    return SourcePrior(tuple(mass), equilibrium, mass_sum, ledger.snapshot())


def build_history_impulse(prior, pm11, saved_u10, saved_h, saved_P, *, ledger):
    """All Pm observations are historical; saved_u10 is already COI transformed.

    The workflow independently verifies time/source identities and the saved
    origin anchor. No nominal clock or new eligibility is inferred here.
    """
    e, partial = _begin(ledger), {}
    try:
        if not isinstance(prior, SourcePrior):
            raise InputError("prior must be the result of prepare_source")
        raw = _array(pm11, (11, 10), ledger, "pm11")
        origin = _array(saved_u10, (10,), ledger, "saved_u10")
        h = _array(saved_h, (10,), ledger, "saved_h", positive=True)
        projector = _array(saved_P, (10, 10), ledger, "saved_P")
        rows = []
        partial["u11"] = rows
        for j in range(11):
            power = [e.sub(raw[10*j+k], prior.equilibrium_pm[k]) for k in range(10)]
            imbalance = e.total(power)
            # Preserve power - sum(power)*mass/sum(mass), including operation order.
            rows.append(tuple(e.sub(power[k], e.div(e.mul(imbalance, prior.mass[k]),
                                                   prior.mass_sum)) for k in range(10)))
        delta = tuple(e.sub(rows[10][k], origin[k]) for k in range(10))
        partial["origin_difference"] = delta
        max_abs, all_bits = 0.0, True
        for k in range(10):
            difference = e._number("point", "abs", lambda k=k: abs(delta[k].point))
            if e.compare("point", difference, max_abs, lambda x, y: x > y):
                max_abs = difference
            same = ledger.call("identity.bits", lambda k=k:
                               struct.pack("!d", rows[10][k].point) ==
                               struct.pack("!d", origin[k].point))
            all_bits = all_bits and same
        qualified = e.compare("point", max_abs, 1e-12, lambda x, y: x <= y)
        anchor = {"max_abs_difference": max_abs, "bitwise_equal": all_bits,
                  "qualified": qualified, "absolute_tolerance": 1e-12}
        partial["anchor"] = anchor
        if not qualified:
            raise IdentityError("Pm10 source anchor exceeds the fixed absolute 1e-12 tolerance")
        impulses = []
        partial["d"] = impulses
        for j in range(10):
            before_p = [e.mul(h[j], e.sub(e.mul(Dual.exact(0.5),
                                                e.add(rows[j][k], rows[j+1][k])), origin[k]))
                        for k in range(10)]
            impulses.append(tuple(e.dot(projector[10*k:10*k+10], before_p) for k in range(10)))
    except (NumericalError, InputError) as error:
        _raise_with_prefix(error, partial, ledger)
    finally:
        ledger.closed = True
    return HistoryImpulse(tuple(rows), tuple(impulses), delta, anchor, ledger.snapshot())


SUMMARY_FIELDS = (
    "impulse_sq", "weighted_impulse_sq", "g4_reference_sq", "btilde4_reference_sq",
    "normal_rhs_sq", "dynamic_before_sq", "dynamic_after_sq", "force_before_sq",
    "force_after_sq", "force_impulse_sq", "dynamic_delta_direct", "dynamic_delta_cross",
    "force_delta_direct", "force_delta_cross",
)


@dataclass
class SlotResult:
    status: str
    complete: bool
    values: dict[str, Any]
    counts: dict
    error: str | None = None


def _scale(e, values, c, scale):
    return [e.mul(e.mul(c[j], scale[k]), values[10*j+k]) for j in range(10) for k in range(10)]


def _sensitivity(e, a_norm, trace, determinant):
    t = Dual.exact(trace.upper)
    det = Dual.exact(determinant.lower)
    lower = e.div(Dual.exact(a_norm.lower), t)
    upper = e.div(e.mul(Dual.exact(a_norm.upper), t), det)
    sigma_min = e.sqrt(e.div(det, t))
    sigma_max = e.sqrt(t)
    condition = e.div(t, e.sqrt(det))
    return {"delta_norm_lower": e._max(0.0, lower.lower), "delta_norm_upper": upper.upper,
            "sigma_min_lower": sigma_min.lower, "sigma_max_upper": sigma_max.upper,
            "condition_upper": condition.upper}


def audit_slot(impulse, *, Ltilde, c, S, g4, btilde4, dynamic_residual_scaled4,
               force_residual4, force_residual_projector, ledger):
    """Fixed-parameter diagnostic only. Never compute delta-z or any state.

    A nonpositive determinant lower bound means unresolved, not rank deficient.
    Numeric failures preserve the completed prefix; source/shape and budget
    errors propagate and must stop the caller's scientific invocation.
    """
    e, out = _begin(ledger), {}
    try:
        if not isinstance(impulse, HistoryImpulse):
            raise InputError("impulse must be returned by build_history_impulse")
        matrix = _array(Ltilde, (100, 2), ledger, "Ltilde")
        c = _array(c, (10,), ledger, "c", positive=True)
        scale = _array(S, (10,), ledger, "S", positive=True)
        g = _array(g4, (100,), ledger, "g4")
        b = _array(btilde4, (100,), ledger, "btilde4")
        r0 = _array(dynamic_residual_scaled4, (100,), ledger, "dynamic_residual_scaled4")
        f0 = _array(force_residual4, (10, 10), ledger, "force_residual4")
        projector = _array(force_residual_projector, (10, 10), ledger,
                           "force_residual_projector")
        d = [value for row in impulse.d for value in row]
        dw = _scale(e, d, c, scale)
        out["weighted_impulse"] = dw
        fd = [e.dot(projector[10*k:10*k+10], impulse.d[j])
              for j in range(10) for k in range(10)]
        out["force_impulse"] = fd
        f1 = [e.add(x, dx) for x, dx in zip(f0, fd, strict=True)]
        fw0, fw1, fdw = (_scale(e, v, c, scale) for v in (f0, f1, fd))
        r1 = [e.add(x, dx) for x, dx in zip(r0, dw, strict=True)]
        out.update(dynamic_after=r1, weighted_force_before=fw0, weighted_force_after=fw1,
                   weighted_force_impulse=fdw)
        columns = [matrix[k::2] for k in range(2)]
        gram = [[e.dot(columns[j], columns[k]) for k in range(2)] for j in range(2)]
        out["gram"] = gram
        a = [e.dot(column, dw) for column in columns]
        out["normal_rhs"] = a
        trace = e.add(gram[0][0], gram[1][1])
        determinant = e.sub(e.mul(gram[0][0], gram[1][1]), e.mul(gram[0][1], gram[1][0]))
        out.update(trace=trace, determinant=determinant)
        vectors = (d, dw, g, b, a, r0, r1, fw0, fw1, fdw)
        for name, vector in zip(SUMMARY_FIELDS[:10], vectors, strict=True):
            square = e.squared_norm(vector)
            out[name] = square
            out[name.removesuffix("_sq") + "_norm"] = e.sqrt(square, nonnegative_sum=True)
        out["dynamic_delta_direct"] = e.sub(out["dynamic_after_sq"], out["dynamic_before_sq"])
        out["dynamic_delta_cross"] = e.add(e.mul(Dual.exact(2.0), e.dot(r0, dw)),
                                            out["weighted_impulse_sq"])
        out["force_delta_direct"] = e.sub(out["force_after_sq"], out["force_before_sq"])
        out["force_delta_cross"] = e.add(e.mul(Dual.exact(2.0), e.dot(fw0, fdw)),
                                          out["force_impulse_sq"])
        positive_trace = e.compare("interval", trace.upper, 0.0, lambda x, y: x > y)
        positive_det = e.compare("interval", determinant.lower, 0.0, lambda x, y: x > y)
        nonzero = e.compare("interval", out["normal_rhs_norm"].lower, 0.0, lambda x, y: x > y)
        out["sensitivity_bounds"] = None
        if positive_trace and positive_det:
            out["sensitivity_bounds"] = _sensitivity(e, out["normal_rhs_norm"], trace, determinant)
            status = "resolved_nonzero" if nonzero else "rounding_unresolved"
        else:
            status = "numerically_unresolved"
    except NumericalError as error:
        ledger.closed = True
        return SlotResult("numerical_failure", False, out, ledger.snapshot(), str(error))
    finally:
        ledger.closed = True
    return SlotResult(status, True, out, ledger.snapshot())


def aggregate_slots(records: Sequence[SlotResult], weights, *, ledger):
    """All 258 allocated weights remain in the denominator; never renormalize.

    On incomplete slots, complete-population metrics are None. Partial raw
    weighted contributions and the original missing weight are separately kept.
    """
    e, out = _begin(ledger), {}
    try:
        if len(records) != 258 or not all(isinstance(row, SlotResult) for row in records):
            raise InputError("aggregate requires all 258 allocated SlotResult objects")
        w = _array(weights, (258,), ledger, "weights", positive=True)
        total = e.total(w)
        missing = e.total(w[i] for i, row in enumerate(records) if not row.complete)
        out.update(total_weight=total, missing_weight=missing)
        sums = {}
        out["partial_contributions"] = sums
        for name in SUMMARY_FIELDS:
            terms = []
            for weight, row in zip(w, records, strict=True):
                if row.complete:
                    value = row.values.get(name)
                    if not isinstance(value, Dual):
                        raise InputError(f"completed slot is missing {name}")
                    terms.append(e.mul(weight, value))
            sums[name] = e.total(terms)
        complete = all(row.complete for row in records)
        out["population_metrics"] = dict(sums) if complete else None
        out["completed_slots"] = sum(row.complete for row in records)
        out["unresolved_slots"] = sum(row.complete and row.status != "resolved_nonzero"
                                       for row in records)
    except NumericalError as error:
        ledger.closed = True
        return SlotResult("numerical_failure", False, out, ledger.snapshot(), str(error))
    finally:
        ledger.closed = True
    return SlotResult("complete_population" if complete else "incomplete_population",
                      complete, out, ledger.snapshot())


def _expand_budget(high, *, inputs, positive, extra_interval_comparisons=0):
    """Integer-only expansion of the fixed scalar-expression schedule."""
    result = Counter({"input.finite": inputs, "input.positive": positive,
                      "interval.compare": extra_interval_comparisons})
    for operation, count in high.items():
        point_kind = "mul" if operation == "square" else operation.removesuffix("_nn")
        result[f"point.{point_kind}"] += count
        result["point.finite"] += count
        endpoints = 4 if operation in {"mul", "div"} else 2
        result[f"interval.{point_kind}"] += endpoints * count
        result["interval.nextafter"] += 2 * count
        result["interval.finite"] += (endpoints + 2) * count
        comparisons = {"mul": 6, "div": 8, "square": 4,
                       "sqrt": 2, "sqrt_nn": 4}.get(operation, 0)
        result["interval.compare"] += comparisons * count
        if operation in {"div", "sqrt", "sqrt_nn"}:
            result["point.compare"] += count
    return {key: value for key, value in sorted(result.items()) if value}


def expected_budget(kind: str, *, resolved=True, completed_slots=258):
    """Static counts; does no floating arithmetic or synthetic execution.

    Use resolved=True as the slot hard maximum. resolved=False is the exact
    determinant-unresolved schedule. Earlier arithmetic/input failures consume
    a prefix of the reserved maximum. Aggregate maxima occur at 258 completed.
    """
    if kind == "source":
        return _expand_budget({"div": 10, "add": 10}, inputs=21, positive=11)
    if kind == "impulse":
        budget = Counter(_expand_budget({"sub": 330, "add": 1210, "mul": 1310, "div": 110},
                                        inputs=230, positive=10))
        budget.update({"point.abs": 10, "point.finite": 10,
                       "point.compare": 11, "identity.bits": 10})
        return dict(sorted(budget.items()))
    if kind == "slot":
        operations = Counter({"mul": 2604, "add": 2905, "sub": 3,
                              "square": 902, "sqrt_nn": 10})
        if resolved:
            operations.update({"div": 4, "mul": 1, "sqrt": 3})
        return _expand_budget(operations, inputs=720, positive=20,
                              extra_interval_comparisons=4 if resolved else 3)
    if kind == "aggregate":
        if type(completed_slots) is not int or not 0 <= completed_slots <= 258:
            raise InputError("completed_slots must lie in 0..258")
        terms = len(SUMMARY_FIELDS) * completed_slots
        return _expand_budget({"mul": terms, "add": terms + 258 + (258-completed_slots)},
                              inputs=258, positive=258)
    raise InputError(f"unknown budget kind: {kind}")
