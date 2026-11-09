"""Ball-constrained integral LS candidate; no data access or rollout entry point."""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np


class EnergyBallNumericalError(FloatingPointError):
    """A spent solver attempt with inspectable diagnostics, never a fallback estimate."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {**(diagnostics or {}), "status": "numerical_failure",
                            "message": message, "z": None}


@contextmanager
def _arithmetic(diagnostics=None):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            yield
    except FloatingPointError as error:
        if isinstance(error, EnergyBallNumericalError):
            raise
        raise EnergyBallNumericalError(str(error), diagnostics) from error


def _norm(value):
    return math.hypot(*np.asarray(value).ravel())


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def solve_energy_ball(matrix, right, svd, radius, unconstrained=None):
    """Reuse a supplied thin SVD; rank deficiency remains a research-policy unresolved case.

    The zero-radius multiplier is a limit marker, not a finite KKT certificate.
    Scalar evaluations include endpoint repairs; neither SVD nor a new trajectory is computed.
    """
    if isinstance(radius, (bool, np.bool_)) or not isinstance(
            radius, (int, float, np.integer, np.floating)):
        raise TypeError("radius must be a real scalar")
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    matrix, right = _array(matrix, "matrix"), _array(right, "right")
    if not isinstance(svd, dict) or not {"u", "s", "vh", "rank", "threshold"} <= svd.keys():
        raise ValueError("incomplete thin SVD metadata")
    u, s, vh = (_array(svd[key], key) for key in ("u", "s", "vh"))
    if (matrix.ndim != 2 or s.ndim != 1 or not len(s)
            or matrix.shape[1] != len(s) or matrix.shape[0] < len(s)
            or u.shape != matrix.shape or vh.shape != (len(s), len(s))
            or right.shape != (matrix.shape[0],)):
        raise ValueError("inconsistent thin SVD shapes")
    threshold = svd["threshold"]
    if (isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold)
            or not np.isfinite(threshold) or threshold < 0
            or np.any(s < 0) or np.any(np.diff(s) > 0)):
        raise ValueError("invalid singular spectrum or rank threshold")
    rank = int(np.count_nonzero(s > threshold))
    if type(svd["rank"]) is not int or rank != svd["rank"]:
        raise ValueError("saved numerical rank disagrees with its spectrum and threshold")
    matrix_norm = _norm(matrix)
    with _arithmetic():
        u_error = u.T @ u-np.eye(len(s))
        v_error = vh @ vh.T-np.eye(len(s))
        reconstruction = (u*s)@vh-matrix
    if not all(np.isfinite(value).all() for value in (u_error, v_error, reconstruction, matrix_norm)):
        raise EnergyBallNumericalError("nonfinite SVD validation arithmetic")
    if (np.max(np.abs(u_error)) > 1e-10 or np.max(np.abs(v_error)) > 1e-10
            or _norm(reconstruction) > 1e-10*max(np.finfo(float).tiny, matrix_norm)):
        raise ValueError("supplied factors do not reconstruct an orthonormal thin SVD")
    trace = []
    result = {
        "radius": radius, "rank": rank, "rank_threshold": float(threshold),
        "iterations": 0, "secular_evaluations": 0, "trace": trace,
        "bracket_initial": None, "bracket_final": None,
    }

    def finish(z, status, multiplier):
        z_norm = _norm(z)
        with _arithmetic(result):
            residual = matrix @ z-right
        residual_norm = _norm(residual)
        energy = .5*z_norm*z_norm
        objective = .5*residual_norm*residual_norm
        if (not np.isfinite(z).all() or not math.isfinite(energy)
                or not math.isfinite(residual_norm) or not math.isfinite(objective)):
            raise EnergyBallNumericalError("nonfinite ball-LS output", result)
        if z_norm > radius*(1+1e-12):
            raise EnergyBallNumericalError("ball-LS feasibility check failed", result)
        stationarity, complementarity = None, None
        if multiplier is not None:
            with _arithmetic(result):
                gradient = matrix.T @ residual+multiplier*z
            denominator = matrix_norm*(matrix_norm*z_norm+_norm(right))+multiplier*z_norm
            if not np.isfinite(gradient).all() or not math.isfinite(denominator):
                raise EnergyBallNumericalError("nonfinite KKT check", result)
            gradient_norm = _norm(gradient)
            stationarity = (gradient_norm/denominator if denominator else
                            (0. if gradient_norm == 0 else math.inf))
            complementarity = abs((z_norm/radius)**2-1) if multiplier > 0 else 0.
            if (stationarity > 1e-10 or complementarity > 3e-12
                    or (multiplier > 0 and abs(z_norm/radius-1) > 1e-12)):
                raise EnergyBallNumericalError("ball-LS KKT check failed", result)
        result.update(
            z=z.copy(), status=status, multiplier=multiplier,
            multiplier_kind="zero_radius_limit" if multiplier is None else "finite",
            constraint_active=status in ("zero_baseline", "active"),
            regularization_active=multiplier is not None and multiplier > 0,
            constraint_binding=radius == 0 or abs(z_norm/radius-1) <= 1e-12,
            residual_norm=residual_norm, objective=objective, energy=energy,
            kkt_feasibility_scaled=max(0., z_norm/radius-1) if radius else z_norm,
            boundary_norm_relative_error=abs(z_norm/radius-1) if radius else None,
            kkt_stationarity_scaled=stationarity,
            kkt_complementarity_scaled=complementarity,
        )
        return result

    if radius == 0:
        return finish(np.zeros(len(s)), "zero_baseline", None)
    if rank < len(s):
        result.update(z=None, status="unresolved", multiplier=None,
                      multiplier_kind="rank_unresolved", constraint_active=None,
                      regularization_active=None, constraint_binding=None,
                      residual_norm=None, energy=None, kkt_stationarity_scaled=None,
                      kkt_complementarity_scaled=None, kkt_feasibility_scaled=None,
                      boundary_norm_relative_error=None, objective=None)
        return result
    with _arithmetic(result):
        d = s*s
        g = u.T @ right
        if np.any(d <= 0) or not np.isfinite(g).all():
            raise EnergyBallNumericalError("singular-value squares must be finite and strictly positive", result)
        z_unc = (vh.T @ (g/s) if unconstrained is None
                 else _array(unconstrained, "unconstrained"))
    if z_unc.shape != s.shape or not np.isfinite(z_unc).all():
        raise ValueError("invalid unconstrained solution")
    # A saved representative must solve this same equation, even when the bound is active.
    with _arithmetic(result):
        unc_gradient = matrix.T @ (matrix@z_unc-right)
    unc_scale = matrix_norm*(matrix_norm*_norm(z_unc)+_norm(right))
    if not np.isfinite(unc_gradient).all() or not math.isfinite(unc_scale):
        raise EnergyBallNumericalError("nonfinite unconstrained validation", result)
    if _norm(unc_gradient) > 1e-10*unc_scale:
        raise ValueError("unconstrained representative does not solve the supplied equation")
    if _norm(z_unc) <= radius:
        return finish(z_unc, "inactive", 0.)
    with _arithmetic(result):
        h = (s*g)/radius
    if not np.isfinite(h).all():
        raise EnergyBallNumericalError("scaled secular numerator is not representable", result)

    def evaluate(multiplier, label):
        if result["secular_evaluations"] >= 80:
            raise EnergyBallNumericalError("secular evaluation budget exhausted", result)
        result["secular_evaluations"] += 1
        result["last_attempted_multiplier"] = multiplier
        with _arithmetic(result):
            denominator = d+multiplier
            if not np.isfinite(denominator).all():
                raise EnergyBallNumericalError("nonfinite secular denominator", result)
            w = h/denominator
            norm = _norm(w)
            derivative = -norm*float(np.sum((w/norm)**2/denominator)) if norm else 0.
        if not math.isfinite(norm) or not math.isfinite(derivative):
            raise EnergyBallNumericalError("nonfinite secular evaluation", result)
        trace.append({"label": label, "multiplier": multiplier,
                      "relative_norm": norm, "derivative": derivative})
        return norm, derivative, w

    raw_lower = max(0., float(np.max(np.abs(h)-d)))
    raw_upper = _norm(h)-float(np.min(d))
    if not math.isfinite(raw_upper) or raw_upper <= 0:
        raise EnergyBallNumericalError("active problem has no representable positive upper bracket", result)
    lower = max(0., float(np.nextafter(raw_lower, -math.inf)))
    upper = float(np.nextafter(raw_upper, math.inf))
    lower_norm, _, _ = evaluate(lower, "lower")
    if lower_norm < 1:
        lower = 0.
        lower_norm, _, _ = evaluate(lower, "lower_zero_repair")
    upper_norm, _, upper_w = evaluate(upper, "upper")
    for _ in range(8):
        if upper_norm <= 1:
            break
        upper = float(np.nextafter(upper+max(upper, float(np.max(d))), math.inf))
        upper_norm, _, upper_w = evaluate(upper, "upper_repair")
    if lower_norm < 1 or upper_norm > 1 or lower > upper:
        result["bracket_final"] = [lower, upper]
        raise EnergyBallNumericalError("secular bracket sign check failed", result)
    result["bracket_initial"] = [lower, upper]
    current = lower+(upper-lower)/2
    for iteration in range(65):
        result["bracket_final"] = [lower, upper]
        if 0 <= 1-upper_norm <= 1e-12:
            if upper <= 0:
                raise EnergyBallNumericalError("active constraint needs a positive multiplier", result)
            return finish(vh.T @ (radius*upper_w), "active", upper)
        if iteration == 64 or not lower < current < upper:
            break
        norm, derivative, w = evaluate(current, "root")
        result["iterations"] += 1
        if norm > 1:
            lower = current
        else:
            upper, upper_norm, upper_w = current, norm, w
        newton = current-(norm-1)/derivative if derivative else math.nan
        margin = .1*(upper-lower)
        current = (newton if math.isfinite(newton) and lower+margin < newton < upper-margin
                   else lower+(upper-lower)/2)
    raise EnergyBallNumericalError("ball-LS root did not converge within the fixed budget", result)
