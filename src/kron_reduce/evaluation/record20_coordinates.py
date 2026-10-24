"""Array-only input and coordinate checks for the registered A12 audit."""
from __future__ import annotations

import numpy as np

from kron_reduce.physics.coordinates import coi_project, coi_project_power


def inspect_record20_arrays(*, t, delta, omega, origin_pm, inertia,
                            equilibrium_delta, equilibrium_pm, omega_s):
    """Keep diagnostic future arrays separate from the three-point input.

    No identifier, length, eligibility flag, label or truth statistic enters
    the returned input mapping. An unusable input is represented by None.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (t, delta, omega, inertia, equilibrium_delta, equilibrium_pm)]
    t, delta, omega, inertia, eq_delta, eq_pm = arrays
    # A missing registered Pm selection remains absent, never a fabricated
    # vector. Visible-array diagnostics can still be checked independently.
    origin_pm = None if origin_pm is None else np.asarray(origin_pm, dtype=np.float64)
    n = t.size
    if (t.ndim != 1 or delta.shape != (n, 10) or omega.shape != (n, 10)
            or any(x.shape != (10,) for x in (inertia, eq_delta, eq_pm))
            or (origin_pm is not None and origin_pm.shape != (10,))):
        raise ValueError("record20 array shapes differ from the registered contract")
    if (not np.isfinite(omega_s) or abs(float(omega_s) - 120 * np.pi) > 1e-12
            or not all(np.isfinite(x).all() for x in (inertia, eq_delta, eq_pm))
            or np.any(inertia <= 0)):
        raise ValueError("source_static_prior_invalid")
    mass = inertia / omega_s
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eq = coi_project(eq_delta, mass)
        q = coi_project(delta, mass) - eq
        v = coi_project((omega - 1.0) * omega_s, mass)
        power = None if origin_pm is None else coi_project_power(origin_pm - eq_pm, mass)

    def clock_ok(count):
        observed = t[:count]
        return bool(n >= count and t[0] == 0.0 and np.isfinite(observed).all()
                    and np.all(np.diff(observed) > 0)
                    and np.all(np.abs(observed - np.arange(count) * .01) <= 1e-10))

    def visible_ok(count):
        return bool(n >= count and clock_ok(count)
                    and all(np.isfinite(x[:count]).all() for x in (delta, omega, q, v)))

    def residual(x):
        if not np.isfinite(x).all() or not x.size:
            return None
        value = float(np.max(np.abs(np.sum(x * mass, axis=-1))))
        return value if np.isfinite(value) else None

    q_residual, v_residual = residual(q), residual(v)
    power_residual = float(abs(np.sum(power))) if power is not None and np.isfinite(power).all() else None
    visible_coi_ok = all(x is not None and x <= 1e-10 for x in (q_residual, v_residual))
    power_ok = power_residual is not None and power_residual <= 1e-10
    history_ok = bool(visible_ok(3) and origin_pm is not None and np.isfinite(origin_pm).all()
                      and np.isfinite(power).all()
                      and all((r := residual(x[:3])) is not None and r <= 1e-10
                              for x in (q, v))
                      and power_residual is not None and power_residual <= 1e-10)
    future_ok = visible_ok(5)
    full_ok = bool(1 <= n <= 1501 and visible_ok(n) and visible_coi_ok)
    future_coi_ok = all((r := residual(x[:5])) is not None and r <= 1e-10 for x in (q, v))
    supported = history_ok and future_ok and future_coi_ok
    data = None if not history_ok else {
        "time": (t[:3] - t[2]).copy(), "q": q[:3].copy(), "v": v[:3].copy(),
        "power": power.copy(), "mass": mass.copy(),
        "equilibrium_delta": eq.copy(), "equilibrium_pm": eq_pm.copy(),
        "omega_s": np.asarray(omega_s, dtype=np.float64),
    }
    reasons = []
    if not history_ok:
        reasons.append("history_unavailable")
    if not future_ok:
        reasons.append("two_future_points_unavailable")
    if not full_ok:
        reasons.append("full_visible_record_invalid")
    if not visible_coi_ok:
        reasons.append("coi_constraint_invalid")
    if not power_ok:
        reasons.append("origin_power_unavailable_or_invalid")
    transformed = {"q": q, "v": v}
    if power is not None:
        transformed["power"] = power
    return {
        "history_available": history_ok,
        "at_least_two_real_future_points": future_ok,
        "record20_input_supported": supported,
        "full_visible_record_valid": full_ok,
        "mass_weighted_q_abs_max": q_residual,
        "mass_weighted_v_abs_max": v_residual,
        "power_sum_abs": power_residual,
        "reasons": reasons,
    }, data, transformed
