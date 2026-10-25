"""A17 pure-array input consistency kernel, inactive until separately reviewed.

The caller supplies fixed physical parameters and already permitted arrays.
There are no providers, file reads, fitting, decompositions or default changes.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project


def finite_array(value, name, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def solve_saved_decomposition(*, U, singular, Vh, b, weighted_impulse,
                              rank, threshold, natural_norm, dispatch):
    """One new least-squares solution, using only the frozen A14 decomposition.

    ``dispatch`` owns the single-use LS reservation and attempts/returns.
    This function neither replays the old solution nor makes a new SVD.
    """
    U = finite_array(U, "U", (100, 2))
    singular = finite_array(singular, "singular values", (2,))
    Vh = finite_array(Vh, "Vh", (2, 2))
    b = finite_array(b, "b", (100,))
    impulse = finite_array(weighted_impulse, "weighted impulse", (100,))
    if (type(rank) is not int or rank != 2
            or not np.isfinite(threshold) or threshold < 0
            or not np.isfinite(natural_norm) or natural_norm < 0
            or singular[0] < singular[1] or np.any(singular <= threshold)):
        raise ValueError("the saved full-rank rule must hold without a new cutoff")
    expected = float(64 * np.finfo(np.float64).eps * 100
                     * max(float(singular[0]), float(natural_norm)))
    if threshold != expected:
        raise ValueError("the original saved A14 threshold differs")
    partial: dict[str, Any] = {}

    def solve():
        partial["rhs"] = b + impulse
        partial["projected_rhs"] = U.T @ partial["rhs"]
        partial["coefficients"] = partial["projected_rhs"] / singular
        partial["z0"] = Vh.T @ partial["coefficients"]
        if not all(np.isfinite(value).all() for value in partial.values()):
            error = FloatingPointError("nonfinite saved-decomposition LS result")
            error.partial_record = partial
            raise error
        return partial

    try:
        return dispatch(solve)
    except BaseException as error:
        error.input_consistency_partial_record = {"operation": "saved_ls", "intermediates": partial}
        if not hasattr(error, "partial_record"):
            error.partial_record = partial
        raise


def path_preflight(closure, q, v, z, clock, power, equilibrium, max_step, *, linear):
    """Check each path without constructing another structural validator."""
    n, r = closure.n_nodes, closure.latent_dimension
    if (n, r) != (10, 2):
        raise ValueError("A17 requires the fixed ten-node, two-state closure")
    q = finite_array(q, "initial q", (n,))
    v = finite_array(v, "initial v", (n,))
    z = finite_array(z, "initial z", (r,))
    with np.errstate(over="ignore", invalid="ignore"):
        q_coi, v_coi = float(q @ closure.mass), float(v @ closure.mass)
        mass_sum = float(np.sum(closure.mass))
    if (not np.isfinite([q_coi, v_coi, mass_sum]).all()
            or max(abs(q_coi), abs(v_coi)) > 1e-10):
        error = ValueError("initial state must satisfy the finite original mass COI rule")
        error.partial_record = {"q_mass_coi": q_coi, "v_mass_coi": v_coi, "mass_sum": mass_sum}
        raise error
    clock = finite_array(clock, "clock")
    if (clock.ndim != 1 or not len(clock) or clock[0] != 0
            or np.any(np.diff(clock) <= 0)
            or isinstance(max_step, (bool, np.bool_)) or not np.isfinite(max_step)
            or not 0 < max_step <= 0.01):
        raise ValueError("invalid original integration clock/step")
    if linear and len(clock) != 11:
        raise ValueError("PL history must have exactly eleven recorded points")
    power = finite_array(power, "power", (len(clock), n) if linear else (n,))
    with np.errstate(over="ignore", invalid="ignore"):
        imbalance = np.sum(power, axis=-1) if linear else np.sum(power)
    if not np.isfinite(imbalance).all() or np.max(np.abs(imbalance)) > 1e-10:
        raise ValueError("input power must satisfy the original force balance rule")
    equilibrium = finite_array(equilibrium, "equilibrium", (n,))
    return np.concatenate((q, v, z)), clock, power, equilibrium


def linear_stage_power(left, right, numerator, count):
    """Use recorded endpoints exactly; interpolate only inside the interval."""
    if numerator == 0:
        return left
    if numerator == count:
        return right
    return left + (numerator / count) * (right - left)


def rollout_fixed_input(closure, q, v, z, clock, power, equilibrium, *,
                        max_step, linear, preflight_dispatch: Callable,
                        rhs_dispatch: Callable, power_dispatch: Callable | None = None,
                        update_sink: Callable | None = None):
    """Closed-loop RK4 with PL observed history or the old constant future.

    All stages use model-generated states. The constant branch deliberately
    bypasses interpolation arithmetic, preserving the original RK4 order.
    The caller records attempted/returned RHS calls before durable closure.
    """
    if linear and power_dispatch is None:
        raise ValueError("PL stage input evaluations require their own accounting")
    preflight_values = {}

    def preflight():
        value = path_preflight(closure, q, v, z, clock, power, equilibrium, max_step, linear=linear)
        preflight_values["returned"] = value
        return value

    try:
        state, clock, power, equilibrium = preflight_dispatch(preflight)
    except BaseException as error:
        error.input_consistency_partial_record = {"phase": "preflight", **preflight_values}
        if not hasattr(error, "partial_record"):
            error.partial_record = error.input_consistency_partial_record
        raise
    output = [state.copy()]
    input_schedule = []
    updates = 0
    current = {"phase": "before_first_interval"}
    try:
        for j, interval in enumerate(np.diff(clock)):
            count = max(1, int(np.ceil(interval / max_step - 1e-12)))
            h = interval / count
            for k in range(count):
                stages = {}
                current = {"interval_index": j, "substep_index": k,
                           "substep_count": count, "substep_size": h,
                           "substep_start_state": state.copy(), "stages": stages}

                def field(value, numerator, stage, *, j=j, k=k, count=count,
                          stages=stages, current=current):
                    current["stage"] = stage
                    current["phase"] = "input_evaluation" if linear else "rhs"
                    stages[stage] = {"state": value.copy()}
                    if not np.isfinite(value).all():
                        raise FloatingPointError("nonfinite RK4 stage state")

                    def evaluate_input():
                        witness = {"interval_index": j, "substep_index": k, "substep_count": count,
                                   "numerator": numerator, "stage_code": int(stage[1]),
                                   "returned_power": None, "used_power": None}
                        input_schedule.append(witness)
                        result = linear_stage_power(power[j], power[j + 1], numerator, count)
                        witness["returned_power"] = result.copy()
                        stages[stage]["power"] = result.copy()
                        return result

                    u = power_dispatch(evaluate_input) if linear else power
                    if linear:
                        input_schedule[-1]["used_power"] = u.copy()
                    stages[stage]["power"] = u.copy()
                    if not np.isfinite(u).all():
                        raise FloatingPointError("nonfinite PL stage power")
                    current["phase"] = "rhs"

                    def evaluate_rhs():
                        result = closure.rhs(value, u, equilibrium)
                        stages[stage]["returned_rhs"] = result.copy()
                        return result

                    result = rhs_dispatch(evaluate_rhs)
                    stages[stage].setdefault("returned_rhs", result.copy())
                    if not np.isfinite(result).all():
                        raise FloatingPointError("nonfinite RK4 RHS return")
                    return result

                k1 = field(state, k, "k1")
                k2 = field(state + h * k1 / 2, k + .5, "k2")
                k3 = field(state + h * k2 / 2, k + .5, "k3")
                k4 = field(state + h * k3, k + 1, "k4")
                current["phase"] = "state_update"
                state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                state[:10] = coi_project(state[:10], closure.mass)
                state[10:20] = coi_project(state[10:20], closure.mass)
                if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e12:
                    raise FloatingPointError("divergent input-consistency rollout")
                updates += 1
                if update_sink is not None:
                    current["phase"] = "update_sink"
                    update_sink(updates)
            output.append(state.copy())
    except BaseException as error:
        partial = {"output_prefix": np.stack(output), "last_state": state.copy(),
                   "input_schedule_prefix": input_schedule,
                   "completed_updates": updates, "current": current}
        error.input_consistency_partial_record = partial
        if not hasattr(error, "partial_record"):
            error.partial_record = partial
        raise
    schedule = {"index_fields": ["interval_index", "substep_index", "substep_count", "numerator", "stage_code"],
                "stage_codes": {"k1": 1, "k2": 2, "k3": 3, "k4": 4},
                "indices": np.asarray([[row[key] for key in ("interval_index", "substep_index", "substep_count",
                                                               "numerator", "stage_code")]
                                       for row in input_schedule], dtype=np.float64).reshape(-1, 5),
                "returned_power": np.asarray([row["returned_power"] for row in input_schedule],
                                             dtype=np.float64).reshape(-1, 10),
                "used_power": np.asarray([row["used_power"] for row in input_schedule],
                                         dtype=np.float64).reshape(-1, 10)}
    return {"path": np.stack(output), "rk4_updates": updates, "input_schedule": schedule}


def reset_visible_origin(closure, history_path, observed_endpoint, equilibrium, *, energy_dispatch=None):
    """Keep the model's own hidden endpoint; expose the visible reset energy."""
    path = finite_array(history_path, "closed-loop history", (11, 22))
    observed = finite_array(observed_endpoint, "observed endpoint", (20,))
    state = np.r_[observed, path[-1, 20:]]
    record = {"before": path[-1].copy(), "origin": state,
              "reset_is_observation_injection_not_dissipative_dynamics": True}
    dispatch = (lambda function: function()) if energy_dispatch is None else energy_dispatch

    def evaluate_energy(name, value):
        result = closure.energy(value, equilibrium)
        record[name] = result
        return result

    try:
        energy_before = dispatch(lambda: evaluate_energy("energy_before", path[-1]))
        energy_after = dispatch(lambda: evaluate_energy("energy_after", state))
        record["reset_energy_change"] = energy_after - energy_before
        if not np.isfinite([energy_before, energy_after, record["reset_energy_change"]]).all():
            raise FloatingPointError("nonfinite visible-reset energy/change")
        return record
    except BaseException as error:
        error.input_consistency_partial_record = record
        if not hasattr(error, "partial_record"):
            error.partial_record = record
        raise
