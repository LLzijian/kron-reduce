"""Pure-array A17 initializer/path composition with retained failure evidence.

The caller owns source identities, durable scopes, persistence and population.
No file access, model construction, factorization, fit or replay is performed.
``dispatch(category, callable)`` must call the supplied function exactly once.
Counters distinguish actual callback work from a durable dispatch return.
RK4 updates and the observed-driven map are derived facts, not dummy calls.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType

import numpy as np

from kron_reduce.evaluation.input_consistency import (
    reset_visible_origin,
    rollout_fixed_input,
    solve_saved_decomposition,
)
from kron_reduce.evaluation.input_consistency_checks import (
    check_decomposition,
    check_inherited_quadrature,
    check_ls_solution,
    judge_physical_checks,
)

INTEGRAL_SHAPES = MappingProxyType({
    "Ltilde": (100, 2), "U": (100, 2), "s": (2,), "Vh": (2, 2),
    "btilde4": (100,), "c": (10,), "S": (10,), "h": (10,), "time": (11,),
    "zh2": (2,), "zh4": (2,), "zTobs2": (2,), "zTobs4": (2,),
    "PhiT": (2, 2), "wT": (2,),
})
_NUMERICAL_ERRORS = (FloatingPointError, OverflowError, np.linalg.LinAlgError)
_COUNT_FIELDS = (
    "dispatch_attempted", "callback_attempted", "callback_returned",
    "callback_failed", "dispatch_returned", "dispatch_failed", "dispatch_contract_failed",
)


def initializer_success_caps():
    """Actual process/operator calls for one successful initializer."""
    return {"initializers": 1, "decomposition_check": 1, "ls": 1,
            "ls_diagnostics": 1, "quadrature_proxy": 1}


def path_success_caps(*, point_count, rk4_updates, linear):
    """Caps from already registered point/update counts, without integration."""
    if (type(point_count) is not int or not 1 <= point_count <= 301
            or type(rk4_updates) is not int or rk4_updates < 0
            or type(linear) is not bool or (linear and point_count != 11)):
        raise ValueError("invalid registered path operation counts")
    return {"path_evaluations": 1, "rollouts": 1, "path_preflight": 1,
            "input_eval": 4 * rk4_updates if linear else 0,
            "integration_rhs": 4 * rk4_updates, "physical_rhs": point_count,
            "physical_judgement": 1}


def reset_success_caps():
    return {"visible_resets": 1, "energy_values": 2}


def _array(value, name, shape):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{name} must be a real array of shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")  # noqa: TRY004 - source contract
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


class _Process:
    def __init__(self, name, dispatch):
        if not callable(dispatch):
            raise TypeError("a counted dispatcher is required")
        self.dispatch = dispatch
        self.record = {
            "operation": name, "status": "running", "phase": "validation",
            "call_counts": {}, "derived_actual": {}, "checks": {},
            "diagnostic_operation_counts": {},
            "counting_scope": "callback_calls_and_explicit_array_operations_not_scalar_primitives",
        }

    def call(self, category, function):
        counts = self.record["call_counts"].setdefault(category, dict.fromkeys(_COUNT_FIELDS, 0))
        counts["dispatch_attempted"] += 1
        called = False
        returned = False
        actual = None
        callback_error = None

        def calculate():
            nonlocal called, returned, actual, callback_error
            if called:
                raise RuntimeError("dispatcher attempted the same callback twice")
            called = True
            counts["callback_attempted"] += 1
            try:
                actual = function()
            except BaseException as error:
                callback_error = error
                counts["callback_failed"] += 1
                raise
            returned = True
            counts["callback_returned"] += 1
            # The outer callback returns this record; do not create a cycle.
            self.record["last_callback_return"] = {
                "category": category, "index": counts["callback_returned"] - 1,
                **({"is_process_record": True} if actual is self.record else {"value": actual}),
            }
            return actual

        try:
            value = self.dispatch(category, calculate)
        except BaseException as error:
            counts["dispatch_failed"] += 1
            if error is not callback_error:
                self.record["dispatcher_failure"] = {
                    "category": category, "type": type(error).__name__, "message": str(error),
                    "callback_started": called, "callback_returned": returned,
                }
                if isinstance(error, _NUMERICAL_ERRORS):
                    failure = RuntimeError("A17 dispatcher failed outside the numerical callback")
                    for name in ("partial_record", "input_consistency_partial_record",
                                 "input_consistency_operation_partial"):
                        if hasattr(error, name):
                            setattr(failure, name, getattr(error, name))
                    raise failure from error
            raise
        counts["dispatch_returned"] += 1
        if not returned or value is not actual:
            counts["dispatch_contract_failed"] += 1
            raise RuntimeError("dispatcher did not return the actual single callback value")
        return value

    def check(self, category, function):
        self.record["phase"] = category

        def calculate():
            result = function()
            self.record["checks"][category] = result
            self.record["diagnostic_operation_counts"][category] = dict(result["operation_counts"])
            if category == "quadrature_proxy":
                self.record["derived_actual"]["observation_map_attempted"] = (
                    result["operation_counts"]["matmul_attempted"])
                self.record["derived_actual"]["observation_map_returned"] = (
                    result["operation_counts"]["matmul_returned"])
            return result

        result = self.call(category, calculate)
        if result["qualified"] is not True:
            error = FloatingPointError(f"A17 {category} gate failed: {result['reasons']}")
            error.partial_record = self.record
            raise error
        return result

    def run(self, category, function):
        try:
            result = self.call(category, function)
            self.record["status"] = "complete"
            return result
        except BaseException as error:
            self.record["status"] = (
                "numerical_failure" if isinstance(error, _NUMERICAL_ERRORS) else "global_failure")
            self.record["failure"] = {"type": type(error).__name__, "message": str(error)}
            kernel_partial = getattr(error, "input_consistency_partial_record", None)
            if kernel_partial is not None and kernel_partial is not self.record:
                self.record["kernel_partial"] = kernel_partial
            error.input_consistency_paths_partial_record = self.record
            if not hasattr(error, "partial_record"):
                error.partial_record = self.record
            raise


def estimate_initial(integral, weighted_impulse_point, *, rank, threshold, natural_norm,
                     determinant_lower, trace_upper, dispatch):
    """Validate the original factors, solve once, then apply frozen diagnostics.

    ``integral`` contains exactly the fifteen independently registered members.
    Provenance, same-slot Gram/B, clock and model identity remain caller gates.
    c/S/h/time are retained and shape/finite checked, never used to rescale d_w.
    """
    work = _Process("estimate_initial", dispatch)
    record = work.record
    record["success_caps"] = initializer_success_caps()
    record["derived_actual"] = {"observation_map_attempted": 0, "observation_map_returned": 0}
    record["inputs"] = {"integral": integral, "weighted_impulse_point": weighted_impulse_point,
                        "rank": rank, "threshold": threshold, "natural_norm": natural_norm,
                        "determinant_lower": determinant_lower, "trace_upper": trace_upper}

    def calculate():
        if not isinstance(integral, Mapping) or set(integral) != set(INTEGRAL_SHAPES):
            raise ValueError("integral must contain exactly the fifteen registered members")
        saved = {}
        record["validated_integral"] = saved
        for name, shape in INTEGRAL_SHAPES.items():
            saved[name] = _array(integral[name], name, shape)
        impulse = _array(weighted_impulse_point, "weighted_impulse_point", (100,))
        cutoff, natural = _scalar(threshold, "threshold"), _scalar(natural_norm, "natural_norm")
        det, trace = _scalar(determinant_lower, "determinant_lower"), _scalar(trace_upper, "trace_upper")
        singular = saved["s"]
        if (type(rank) is not int or rank != 2 or cutoff < 0 or natural < 0
                or singular[0] < singular[1] or np.any(singular <= cutoff)):
            raise ValueError("the original full-rank identity must hold before diagnostics or LS")
        expected = float(64 * np.finfo(np.float64).eps * 100
                         * max(float(singular[0]), natural))
        record["expected_original_threshold"] = expected
        if cutoff != expected:
            raise ValueError("the original saved A14 threshold differs")
        work.check("decomposition_check", lambda: check_decomposition(
            B=saved["Ltilde"], U=saved["U"], singular=singular, Vh=saved["Vh"]))
        record["phase"] = "ls"

        def solve():
            result = solve_saved_decomposition(
                U=saved["U"], singular=singular, Vh=saved["Vh"], b=saved["btilde4"],
                weighted_impulse=impulse, rank=rank, threshold=cutoff, natural_norm=natural,
                dispatch=lambda fn: work.call("ls", fn))
            record["ls"] = result
            return result

        solution = solve()
        work.check("ls_diagnostics", lambda: check_ls_solution(
            B=saved["Ltilde"], rhs=solution["rhs"], z0=solution["z0"],
            determinant_lower=det, trace_upper=trace))
        work.check("quadrature_proxy", lambda: check_inherited_quadrature(
            zh2=saved["zh2"], zh4=saved["zh4"], zTobs2=saved["zTobs2"],
            zTobs4=saved["zTobs4"], PhiT=saved["PhiT"], wT=saved["wT"], z0=solution["z0"]))
        record["z0"] = solution["z0"]
        record["phase"] = "calculation_complete"
        return record

    return work.run("initializers", calculate)


def evaluate_fixed_path(closure, q, v, z, clock, power, equilibrium, *,
                        max_step, linear, target_scale, dispatch):
    """One closed-loop rollout and its output-point physical checks, no retry.

    The complete returned path is retained before any physical callback. On
    exceptions the caller can persist it from input_consistency_paths_partial_record.
    target_scale is validated/retained for the separately counted pair refinement;
    it does not affect this path, its inputs or its physical checks.
    """
    work = _Process("evaluate_fixed_path", dispatch)
    record = work.record
    record["derived_actual"] = {"successful_rk4_updates": 0}
    record["physical_points"] = []
    record["inputs"] = {"q": q, "v": v, "z": z, "clock": clock, "power": power,
                        "equilibrium": equilibrium, "max_step": max_step, "linear": linear,
                        "target_scale": target_scale}

    def calculate():
        if type(linear) is not bool:
            raise ValueError("linear must be the registered boolean input mode")
        step = _scalar(max_step, "max_step")
        if step not in (.005, .0025):
            raise ValueError("only the two frozen A17 integration steps are permitted")
        scale = _array(target_scale, "target_scale", (20,))
        if np.any(scale <= 0):
            raise ValueError("target_scale must be positive")
        times = np.asarray(clock)
        if times.ndim != 1 or not (len(times) == 11 if linear else 2 <= len(times) <= 301):
            raise ValueError("invalid registered history/future point count")
        times = _array(times, "clock", (len(times),))
        inputs = _array(power, "power", (len(times), 10) if linear else (10,))
        eq = _array(equilibrium, "equilibrium", (10,))
        initial_q = _array(q, "initial q", (10,))
        initial_v = _array(v, "initial v", (10,))
        initial_z = _array(z, "initial z", (2,))
        record["target_scale"] = scale
        expected_updates = (20 if linear else 2 * (len(times) - 1)) * (2 if step == .0025 else 1)
        record["success_caps"] = path_success_caps(
            point_count=len(times), rk4_updates=expected_updates, linear=linear)
        record["derived_expected"] = {"successful_rk4_updates": expected_updates}
        record["phase"] = "rollout"

        def preflight(function):
            def checked():
                value = function()
                record["preflight_return"] = value
                with np.errstate(over="ignore", invalid="ignore"):
                    widths = np.diff(value[1])
                    ratios = widths / step
                if (not np.isfinite(ratios).all() or np.any(widths <= 0)
                        or np.any(ratios > expected_updates + 1)
                        or (not linear and value[1][-1] > 3.0 + 1e-12)):
                    raise ValueError("clock exceeds the frozen A17 path budget or future horizon")
                counts = [max(1, int(np.ceil(width / step - 1e-12))) for width in widths]
                record["interval_update_counts"] = counts
                if sum(counts) != expected_updates:
                    raise ValueError("clock does not have frozen 20/40 or 2N/4N update counts")
                return value

            return work.call("path_preflight", checked)

        def update_sink(completed):
            previous = record["derived_actual"]["successful_rk4_updates"]
            if type(completed) is not int or completed != previous + 1:
                raise RuntimeError("RK4 completion sink is not the next actual update")
            record["derived_actual"]["successful_rk4_updates"] = completed

        def integrate():
            result = rollout_fixed_input(
                closure, initial_q, initial_v, initial_z, times, inputs, eq,
                max_step=step, linear=linear,
                preflight_dispatch=preflight,
                rhs_dispatch=lambda fn: work.call("integration_rhs", fn),
                power_dispatch=(lambda fn: work.call("input_eval", fn)) if linear else None,
                update_sink=update_sink)
            record["rollout"] = result
            record["path"] = result["path"]
            return result

        rollout = work.call("rollouts", integrate)
        if (rollout["rk4_updates"] != record["derived_actual"]["successful_rk4_updates"]
                or rollout["rk4_updates"] != expected_updates):
            raise RuntimeError("RK4 returned count differs from actual sink or frozen count")
        record["phase"] = "physical_checks"
        for index, state in enumerate(rollout["path"]):
            point = {"index": index}
            record["physical_points"].append(point)
            record["current_physical_index"] = index

            def inspect_point(state=state, index=index, point=point):
                value = closure.energy_identity(state, inputs[index] if linear else inputs, eq)
                point["energy_identity_return"] = value
                if not isinstance(value, Mapping):
                    raise TypeError("energy_identity must return its physical dictionary")
                point.update(value)
                with np.errstate(over="ignore", invalid="ignore"):
                    point["q_mass_coi"] = abs(float(closure.mass @ state[:10]))
                    point["v_mass_coi"] = abs(float(closure.mass @ state[10:20]))
                return point

            work.call("physical_rhs", inspect_point)
        work.check("physical_judgement", lambda: judge_physical_checks(
            record["physical_points"], expected_count=len(times)))
        record["phase"] = "calculation_complete"
        return record

    return work.run("path_evaluations", calculate)


def reset_path_origin(closure, history_path, observed_endpoint, equilibrium, *, dispatch):
    """Reset visible observations once while preserving this path's own z_T."""
    work = _Process("reset_path_origin", dispatch)
    record = work.record
    record["success_caps"] = reset_success_caps()
    record["inputs"] = {"history_path": history_path, "observed_endpoint": observed_endpoint,
                        "equilibrium": equilibrium}

    def calculate():
        path = _array(history_path, "history_path", (11, 22))
        observed = _array(observed_endpoint, "observed_endpoint", (20,))
        eq = _array(equilibrium, "equilibrium", (10,))
        record["phase"] = "visible_reset"
        result = reset_visible_origin(
            closure, path, observed, eq, energy_dispatch=lambda fn: work.call("energy_values", fn))
        record["reset"] = result
        if not np.array_equal(result["origin"][20:], path[-1, 20:]):
            raise RuntimeError("visible reset changed the path's own hidden endpoint")
        record["phase"] = "calculation_complete"
        return record

    return work.run("visible_resets", calculate)
