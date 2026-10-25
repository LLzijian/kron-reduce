"""Pure-array A17 diagnostics; none of these functions solves or integrates.

The caller owns source permissions and durable ``dispatch(category, callable)``.
Dispatch exceptions propagate. Numeric/input failures return qualified=False
with finite JSON scalars (nonfinite returns are recorded by representation).

Successful work per call, excluding elementary arithmetic and finite checks:
* decomposition_check: 3 matrix products, 4 Frobenius norms.
* ls_diagnostics: 2 matrix-vector products, 1 Frobenius + 3 vector norms.
* quadrature_proxy: 1 matrix-vector product, 4 vector norms.
* path_refinement with N rows: 0 products, 2 vectorized norm calls / 2N norms.
* physical_judgement: 0 products/norms; inspect N precomputed dictionaries.

Norms use a fixed max-absolute-value scaling to avoid avoidable square overflow
or underflow. Counts describe array operations, NOT scalar-primitive auditing.
No decomposition, solver, model, RHS, energy function or file access is used.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real

import numpy as np

TAU_NUM = float(64 * np.finfo(np.float64).eps * 100)


class _CheckFailure(Exception):
    pass


class _Work:
    def __init__(self, category):
        self.record = {
            "category": category, "qualified": False, "reasons": [], "values": {},
            "nonfinite_values": {}, "arrays": {}, "operation_counts": {
                f"{kind}_{state}": 0
                for kind in ("matmul", "norm_calls", "matrix_norms", "vector_norms")
                for state in ("attempted", "returned")
            }, "counting_scope": "explicit_array_operations_not_scalar_primitives",
        }

    def scalar(self, name, value):
        value = float(value)
        if not math.isfinite(value):
            self.record["values"][name] = None
            self.record["nonfinite_values"][name] = repr(value)
            raise _CheckFailure(f"nonfinite:{name}")
        self.record["values"][name] = value
        return value

    def array(self, name, value):
        self.record["arrays"][name] = value
        if not np.isfinite(value).all():
            raise _CheckFailure(f"nonfinite:{name}")
        return value

    def matmul(self, left, right):
        self.record["operation_counts"]["matmul_attempted"] += 1
        result = left @ right
        self.record["operation_counts"]["matmul_returned"] += 1
        return result

    def norm(self, value, *, axis=None):
        counts = self.record["operation_counts"]
        kind = "matrix_norms" if value.ndim == 2 and axis is None else "vector_norms"
        amount = len(value) if axis == 1 else 1
        counts["norm_calls_attempted"] += 1
        counts[f"{kind}_attempted"] += amount
        scale = np.max(np.abs(value), axis=axis)
        if axis is None:
            result = 0.0 if scale == 0 else scale * np.sqrt(np.sum((value / scale) ** 2))
        else:
            divisor = np.where(scale == 0, 1.0, scale)
            result = scale * np.sqrt(np.sum((value / divisor[:, None]) ** 2, axis=axis))
        counts["norm_calls_returned"] += 1
        counts[f"{kind}_returned"] += amount
        return result

    def gate(self, name, condition):
        self.record.setdefault("gates", {})[name] = bool(condition)
        if not condition:
            self.record["reasons"].append(name)


def _array(value, name, shape=None):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise _CheckFailure(f"invalid_real_array:{name}")
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _CheckFailure(f"invalid_array:{name}") from error
    if shape is not None and array.shape != shape:
        raise _CheckFailure(f"invalid_shape:{name}")
    if not np.isfinite(array).all():
        raise _CheckFailure(f"nonfinite_input:{name}")
    return array


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _CheckFailure(f"invalid_real_scalar:{name}")
    value = float(value)
    if not math.isfinite(value):
        raise _CheckFailure(f"nonfinite_input:{name}")
    return value


def _execute(category, function, dispatch):
    def calculate():
        work = _Work(category)
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                function(work)
        except _CheckFailure as error:
            work.record["reasons"].append(str(error))
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            work.record["reasons"].append(f"arithmetic_failure:{type(error).__name__}")
        work.record["qualified"] = not work.record["reasons"]
        return work.record

    return calculate() if dispatch is None else dispatch(category, calculate)


def check_decomposition(*, B, U, singular, Vh, dispatch: Callable | None = None):
    """Check the saved factors without recomputing a factorization or rank."""
    def calculate(work):
        matrix = _array(B, "B", (100, 2))
        left = _array(U, "U", (100, 2))
        values = _array(singular, "singular", (2,))
        right = _array(Vh, "Vh", (2, 2))
        if not (values[0] >= values[1] > 0):
            raise _CheckFailure("invalid_singular_order_or_sign")
        gram_u = work.array("left_orthogonality_residual", work.matmul(left.T, left) - np.eye(2))
        gram_v = work.array("right_orthogonality_residual", work.matmul(right, right.T) - np.eye(2))
        defect = work.array("reconstruction_residual", matrix - work.matmul(left * values, right))
        b_norm = work.scalar("B_frobenius", work.norm(matrix))
        u_error = work.scalar("left_orthogonality_frobenius", work.norm(gram_u))
        v_error = work.scalar("right_orthogonality_frobenius", work.norm(gram_v))
        b_error = work.scalar("reconstruction_frobenius", work.norm(defect))
        work.scalar("tau_num", TAU_NUM)
        bound = work.scalar("reconstruction_bound", TAU_NUM * max(1.0, b_norm))
        work.gate("left_orthogonality", u_error <= TAU_NUM)
        work.gate("right_orthogonality", v_error <= TAU_NUM)
        work.gate("reconstruction", b_error <= bound)

    return _execute("decomposition_check", calculate, dispatch)


def check_ls_solution(*, B, rhs, z0, determinant_lower, trace_upper,
                      dispatch: Callable | None = None):
    """Frozen normal-equation and conditioning proxies, not a forward certificate."""
    def calculate(work):
        matrix = _array(B, "B", (100, 2))
        right = _array(rhs, "rhs", (100,))
        initial = _array(z0, "z0", (2,))
        det = work.scalar("determinant_lower", _real(determinant_lower, "determinant_lower"))
        trace = work.scalar("trace_upper", _real(trace_upper, "trace_upper"))
        if det <= 0 or trace <= 0:
            raise _CheckFailure("nonpositive_gram_bounds")
        ratio = work.scalar("gram_ratio_before_outward_rounding", det / trace)
        lower = work.scalar("lambda_lower", np.nextafter(ratio, -np.inf))
        if lower <= 0:
            raise _CheckFailure("unresolved_lambda_lower")
        residual = work.array("residual", right - work.matmul(matrix, initial))
        normal = work.array("normal_residual", work.matmul(matrix.T, residual))
        b_norm = work.scalar("B_frobenius", work.norm(matrix))
        rhs_norm = work.scalar("rhs_norm", work.norm(right))
        z_norm = work.scalar("z0_norm", work.norm(initial))
        normal_norm = work.scalar("normal_residual_norm", work.norm(normal))
        scale = work.scalar("normal_scale", max(1.0, b_norm * (rhs_norm + b_norm * z_norm)))
        work.scalar("tau_num", TAU_NUM)
        normal_bound = work.scalar("normal_residual_bound", TAU_NUM * scale)
        proxy = work.scalar("conditioning_proxy", (normal_norm + normal_bound) / lower)
        proxy_bound = work.scalar("conditioning_proxy_bound", 1e-6 * (1.0 + z_norm))
        work.gate("normal_residual", normal_norm <= normal_bound)
        work.gate("conditioning_proxy", proxy <= proxy_bound)
        work.record["certificate_kind"] = "numerical_proxy_not_strict_forward_error_bound"

    return _execute("ls_diagnostics", calculate, dispatch)


def check_inherited_quadrature(*, zh2, zh4, zTobs2, zTobs4, PhiT, wT, z0,
                               dispatch: Callable | None = None):
    """One new observed-driven map; no new two-point solve or future initializer."""
    def calculate(work):
        old2, old4 = _array(zh2, "zh2", (2,)), _array(zh4, "zh4", (2,))
        end2, end4 = _array(zTobs2, "zTobs2", (2,)), _array(zTobs4, "zTobs4", (2,))
        phi, driven = _array(PhiT, "PhiT", (2, 2)), _array(wT, "wT", (2,))
        initial = _array(z0, "z0", (2,))
        endpoint = work.array("observed_driven_endpoint", work.matmul(phi, initial) + driven)
        delta_z = work.array("inherited_initial_difference", old2 - old4)
        delta_t = work.array("inherited_endpoint_difference", end2 - end4)
        dz = work.scalar("inherited_initial_difference_norm", work.norm(delta_z))
        dt = work.scalar("inherited_endpoint_difference_norm", work.norm(delta_t))
        initial_norm = work.scalar("new_initial_norm", work.norm(initial))
        endpoint_norm = work.scalar("new_observed_driven_endpoint_norm", work.norm(endpoint))
        bound_z = work.scalar("initial_bound", 1e-6 * (1.0 + initial_norm))
        bound_t = work.scalar("endpoint_bound", 1e-6 * (1.0 + endpoint_norm))
        work.gate("inherited_initial_quadrature", dz <= bound_z)
        work.gate("inherited_endpoint_quadrature", dt <= bound_t)
        work.record["certificate_kind"] = "inherited_quadrature_proxy_not_new_two_order_solve"
        work.record["endpoint_for_diagnostics_only"] = True

    return _execute("quadrature_proxy", calculate, dispatch)


def check_refinement(coarse, fine, target_scale, *, dispatch: Callable | None = None):
    """Check one full pair of saved histories or futures, including their origins."""
    def calculate(work):
        low, high = _array(coarse, "coarse"), _array(fine, "fine")
        if (low.shape != high.shape or low.ndim != 2 or low.shape[1] != 22
                or not 1 <= len(low) <= 301):
            raise _CheckFailure("invalid_path_pair_shape")
        scale = _array(target_scale, "target_scale", (20,))
        if np.any(scale <= 0):
            raise _CheckFailure("nonpositive_target_scale")
        visible = work.array("visible_normalized_difference", (low[:, :20] - high[:, :20]) / scale)
        hidden = work.array("hidden_difference_norms", work.norm(low[:, 20:] - high[:, 20:], axis=1))
        high_norms = work.array("fine_hidden_norms", work.norm(high[:, 20:], axis=1))
        v_max = work.scalar("visible_max_normalized", np.max(np.abs(visible)))
        h_max = work.scalar("hidden_max_norm", np.max(hidden))
        high_max = work.scalar("fine_hidden_max_norm", np.max(high_norms))
        bound = work.scalar("hidden_bound", 1e-5 * (1.0 + high_max))
        work.scalar("visible_bound", 1e-5)
        work.record["point_count"] = len(low)
        work.gate("visible_refinement", v_max <= 1e-5)
        work.gate("hidden_refinement", h_max <= bound)

    return _execute("path_refinement", calculate, dispatch)


def judge_physical_checks(records, *, expected_count, dispatch: Callable | None = None):
    """Judge precomputed energy_identity/COI records; never calls a closure."""
    fields = ("energy_rate", "supply_minus_dissipation", "identity_error", "q_mass_coi", "v_mass_coi")
    nonnegative = ("identity_error", "q_mass_coi", "v_mass_coi")

    def calculate(work):
        if (type(expected_count) is not int or not 1 <= expected_count <= 301
                or not isinstance(records, Sequence) or isinstance(records, (str, bytes))
                or len(records) != expected_count):
            raise _CheckFailure("invalid_physical_record_count")
        work.record["point_count"] = expected_count
        work.record["failed_points"] = []
        maxima = dict.fromkeys(nonnegative, 0.0)
        valid_count = 0
        for index, record in enumerate(records):
            reasons, values, representations = [], {}, {}
            if not isinstance(record, Mapping):
                reasons.append("not_a_mapping")
            else:
                for field in fields:
                    try:
                        values[field] = _real(record.get(field), field)
                    except _CheckFailure as error:
                        values[field] = None
                        representations[field] = repr(record.get(field))
                        reasons.append(str(error))
                for field in nonnegative:
                    if values[field] is not None and values[field] < 0:
                        reasons.append(f"negative:{field}")
                if not reasons:
                    recomputed = abs(values["energy_rate"] - values["supply_minus_dissipation"])
                    if not math.isfinite(recomputed):
                        reasons.append("nonfinite:recomputed_identity_error")
                        representations["recomputed_identity_error"] = repr(recomputed)
                    elif recomputed != values["identity_error"]:
                        reasons.append("inconsistent:identity_error")
            if reasons:
                work.record["failed_points"].append({"index": index, "reasons": reasons,
                                                     "values": values, "representations": representations})
                continue
            valid_count += 1
            for field in nonnegative:
                maxima[field] = max(maxima[field], values[field])
        work.record["valid_point_count"] = valid_count
        for name, value in maxima.items():
            work.scalar(f"valid_records_max_{name}", value)
        work.scalar("energy_identity_bound", 1e-8)
        work.scalar("coi_bound", 1e-10)
        complete = valid_count == expected_count
        work.gate("complete_finite_physical_records", complete)
        work.gate("energy_identity", complete and maxima["identity_error"] <= 1e-8)
        work.gate("q_mass_coi", complete and maxima["q_mass_coi"] <= 1e-10)
        work.gate("v_mass_coi", complete and maxima["v_mass_coi"] <= 1e-10)

    return _execute("physical_judgement", calculate, dispatch)
