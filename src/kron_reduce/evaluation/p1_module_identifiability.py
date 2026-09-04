"""Fit-only conditional-rank and profile audits for whole P1 modules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.data.p1_causal import P1RoleManifest, validate_p1_samples
from kron_reduce.models.base import RunContext
from kron_reduce.models.p1_calibration import core_fingerprint
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel

_PROFILE_MULTIPLIERS = (0.5, 2.0)
_SIGMA_RATIO_MIN = 1e-6
_CONDITION_MAX = 1e6
_PROFILE_INCREASE_MIN = 0.02


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    header = json.dumps(
        {"dtype": "float64-le", "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _validate_system(matrix: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(vector)
        or matrix.shape[0] < 1
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
        or not np.isfinite(vector).all()
    ):
        raise ValueError("a finite nonempty weak system is required")
    return matrix, vector


def _solve_nonnegative_ridge(
    matrix: np.ndarray, vector: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if isinstance(ridge, (bool, np.bool_)) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("selected ridge must be finite and nonnegative")
    if matrix.ndim != 2 or matrix.shape[0] != len(vector) or matrix.shape[1] < 1:
        raise ValueError("nonnegative weak solve requires at least one parameter")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scale = np.sqrt(np.mean(matrix**2, axis=0))
    except FloatingPointError as error:
        raise ValueError("nonfinite weak-design column scale") from error
    if not np.isfinite(scale).all():
        raise ValueError("nonfinite weak-design column scale")
    solver_scale = np.where(scale > 1e-14, scale, 1.0)
    normalized = matrix / solver_scale
    gram = normalized.T @ normalized
    normal_rhs = normalized.T @ vector
    regularized = gram + ridge * np.eye(gram.shape[0])
    jitter = np.finfo(float).eps * max(1.0, float(np.trace(regularized)))
    try:
        upper = np.linalg.cholesky(regularized + jitter * np.eye(gram.shape[0])).T
        compressed_rhs = np.linalg.solve(upper.T, normal_rhs)
        result = lsq_linear(
            upper,
            compressed_rhs,
            bounds=(0.0, np.inf),
            max_iter=3000,
            lsq_solver="exact",
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
        raise ValueError("nonnegative weak solve failed") from error
    theta = result.x / solver_scale
    if not result.success or not np.isfinite(theta).all():
        raise ValueError("nonnegative weak solve failed")
    return (
        theta,
        solver_scale,
        {
            "solver_success": True,
            "solver_iterations": int(result.nit),
            "solver_form": "normal_equation_cholesky_nonnegative_least_squares",
        },
    )


def _objective(
    matrix: np.ndarray,
    vector: np.ndarray,
    theta: np.ndarray,
    scale: np.ndarray,
    ridge: float,
) -> tuple[float, float, float]:
    with np.errstate(over="raise", invalid="raise"):
        residual = matrix @ theta - vector
        data = float(residual @ residual)
        penalty = float(ridge * ((theta * scale) @ (theta * scale)))
        total = data + penalty
    if not np.isfinite([data, penalty, total]).all() or min(data, penalty, total) < 0:
        raise FloatingPointError("nonfinite weak profile objective")
    return total, data, penalty


def _checked_layout(
    layout: Mapping[str, slice], module_ids: Sequence[str], n_parameters: int
) -> dict[str, slice]:
    required = {"all", "node_damping", "coupling", *module_ids}
    if set(layout) != required or len(set(module_ids)) != len(module_ids):
        raise ValueError("parameter layout must exactly name every module block")
    checked: dict[str, slice] = {}
    coverage = np.zeros(n_parameters, dtype=int)
    for name, section in layout.items():
        if (
            not isinstance(section, slice)
            or section.step not in (None, 1)
            or type(section.start) is not int
            or type(section.stop) is not int
            or not 0 <= section.start < section.stop <= n_parameters
        ):
            raise ValueError("parameter layout contains an invalid slice")
        checked[name] = slice(section.start, section.stop)
        if name != "all":
            coverage[section] += 1
    if checked["all"] != slice(0, n_parameters) or not np.array_equal(
        coverage, np.ones(n_parameters, dtype=int)
    ):
        raise ValueError("parameter blocks must partition the full weak design")
    if (
        checked["node_damping"].start != 0
        or checked["coupling"].start != checked["node_damping"].stop
    ):
        raise ValueError("backbone parameters must be the leading contiguous blocks")
    return checked


def _spectrum(
    normalized: np.ndarray,
    active_mask: np.ndarray,
    module_mask: np.ndarray,
) -> dict[str, Any]:
    active = np.flatnonzero(active_mask)
    selected = np.flatnonzero(module_mask & active_mask)
    if not len(active) or not len(selected):
        return {
            "global_rank": 0,
            "global_parameter_count": len(active),
            "conditional_rank": 0,
            "module_parameter_count": len(selected),
            "sigma_ratio": 0.0,
            "condition_number": None,
            "condition_unavailable_reason": "inactive_module_parameter",
        }
    full = normalized[:, active]
    try:
        full_singular = np.linalg.svd(full, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("global weak-design SVD failed") from error
    reference = float(full_singular[0]) if len(full_singular) else 0.0
    full_tolerance = max(full.shape) * np.finfo(float).eps * reference
    global_rank = int(np.sum(full_singular > full_tolerance))
    global_minimum = float(full_singular[-1]) if global_rank == full.shape[1] else 0.0

    module = normalized[:, selected]
    nuisance = np.flatnonzero(active_mask & ~module_mask)
    if len(nuisance):
        nuisance_matrix = normalized[:, nuisance]
        try:
            u, singular, _ = np.linalg.svd(nuisance_matrix, full_matrices=False)
        except np.linalg.LinAlgError as error:
            raise ValueError("nuisance weak-design SVD failed") from error
        nuisance_reference = float(singular[0]) if len(singular) else 0.0
        nuisance_tolerance = max(nuisance_matrix.shape) * np.finfo(float).eps * nuisance_reference
        nuisance_rank = int(np.sum(singular > nuisance_tolerance))
        basis = u[:, :nuisance_rank]
        residual = module - basis @ (basis.T @ module)
    else:
        nuisance_rank = 0
        residual = module
    try:
        conditional = np.linalg.svd(residual, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("conditional weak-design SVD failed") from error
    conditional_tolerance = (
        max(residual.shape)
        * np.finfo(float).eps
        * max(reference, float(conditional[0]) if len(conditional) else 0.0)
    )
    conditional_rank = int(np.sum(conditional > conditional_tolerance))
    conditional_minimum = float(conditional[-1]) if conditional_rank == module.shape[1] else 0.0
    minimum = min(global_minimum, conditional_minimum)
    sigma_ratio = minimum / reference if reference > 0 else 0.0
    condition = reference / minimum if minimum > 0 else None
    if not np.isfinite(sigma_ratio) or (condition is not None and not np.isfinite(condition)):
        raise FloatingPointError("nonfinite conditional spectrum")
    return {
        "global_rank": global_rank,
        "global_parameter_count": int(full.shape[1]),
        "global_minimum_singular_value": global_minimum,
        "reference_singular_value": reference,
        "nuisance_rank": nuisance_rank,
        "conditional_rank": conditional_rank,
        "module_parameter_count": int(module.shape[1]),
        "conditional_minimum_singular_value": conditional_minimum,
        "sigma_ratio": float(sigma_ratio),
        "condition_number": float(condition) if condition is not None else None,
        "condition_unavailable_reason": None if condition is not None else "rank_deficient",
    }


def audit_weak_module_blocks(
    matrix: np.ndarray,
    vector: np.ndarray,
    parameter_layout: Mapping[str, slice],
    modules: Sequence[Mapping[str, str]],
    *,
    selected_ridge: float,
    profile_multipliers: tuple[float, float] = _PROFILE_MULTIPLIERS,
    sigma_ratio_min: float = _SIGMA_RATIO_MIN,
    condition_max: float = _CONDITION_MAX,
    profile_increase_min: float = _PROFILE_INCREASE_MIN,
) -> dict[str, Any]:
    """Audit whole parameter blocks without counting ridge rows as observations."""
    matrix, vector = _validate_system(matrix, vector)
    if isinstance(selected_ridge, (bool, np.bool_)):
        raise TypeError("selected ridge must be finite and nonnegative")
    if (
        profile_multipliers != _PROFILE_MULTIPLIERS
        or any(
            isinstance(value, (bool, np.bool_))
            for value in (sigma_ratio_min, condition_max, profile_increase_min)
        )
        or not np.isfinite([sigma_ratio_min, condition_max, profile_increase_min]).all()
        or sigma_ratio_min < 0
        or condition_max <= 0
        or profile_increase_min < 0
    ):
        raise ValueError("unregistered identifiability settings")
    if any(not isinstance(row, Mapping) for row in modules):
        raise ValueError("module descriptors must be mappings")
    module_ids = tuple(row.get("instance_id") for row in modules)
    if any(not isinstance(name, str) or not name for name in module_ids):
        raise ValueError("every module requires a nonempty instance ID")
    if any(
        set(row) != {"instance_id", "family"}
        or not isinstance(row["family"], str)
        or not row["family"]
        for row in modules
    ):
        raise ValueError("module descriptors require exactly instance_id and family")
    layout = _checked_layout(parameter_layout, module_ids, matrix.shape[1])
    theta, scale, solver = _solve_nonnegative_ridge(matrix, vector, selected_ridge)
    baseline_total, baseline_data, baseline_penalty = _objective(
        matrix, vector, theta, scale, selected_ridge
    )
    backbone_stop = layout["coupling"].stop
    active_mask = np.ones(matrix.shape[1], dtype=bool)
    optional = theta[backbone_stop:]
    if len(optional):
        floor = max(float(np.max(optional)), 1.0) * 1e-8
        active_mask[backbone_stop:] = optional > floor
    else:
        floor = 1e-8
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            raw_scale = np.sqrt(np.mean(matrix**2, axis=0))
    except FloatingPointError as error:
        raise ValueError("nonfinite weak-design column scale") from error
    normalized = matrix / np.where(raw_scale > 1e-14, raw_scale, 1.0)
    results = []
    for descriptor in modules:
        name = str(descriptor["instance_id"])
        section = layout[name]
        module_mask = np.zeros(matrix.shape[1], dtype=bool)
        module_mask[section] = True
        all_active = bool(np.all(active_mask[section]))
        visible_columns = bool(np.all(raw_scale[section] > 1e-14))
        # A nuisance coefficient being driven to zero by the nonnegative solve
        # cannot make another declared block easier to identify.  Keep every
        # structural design column in the rank audit; use ``active_mask`` only
        # to deny support to a target block with screened parameters.
        spectrum = _spectrum(
            normalized,
            np.ones(matrix.shape[1], dtype=bool),
            module_mask,
        )
        profiles = []
        profile_valid = baseline_total > 0 and all_active and visible_columns
        for multiplier in profile_multipliers:
            fixed = multiplier * theta[section]
            nuisance = np.flatnonzero(~module_mask)
            try:
                nuisance_theta, _, nuisance_solver = _solve_nonnegative_ridge(
                    matrix[:, nuisance],
                    vector - matrix[:, section] @ fixed,
                    selected_ridge,
                )
                candidate = np.zeros_like(theta)
                candidate[section] = fixed
                candidate[nuisance] = nuisance_theta
                total, data, penalty = _objective(matrix, vector, candidate, scale, selected_ridge)
                rounding = (
                    128 * np.finfo(float).eps * max(matrix.shape) * max(1.0, baseline_total, total)
                )
                if total + rounding < baseline_total:
                    raise ValueError("profile objective is below its full optimum")
                increase = max(0.0, total - baseline_total) / baseline_total
                if not np.isfinite(increase):
                    raise FloatingPointError("nonfinite relative profile increase")
                profiles.append(
                    {
                        "multiplier": multiplier,
                        "status": "completed",
                        "objective": total,
                        "data_objective": data,
                        "ridge_penalty": penalty,
                        "relative_increase": float(increase),
                        "rounding_tolerance": rounding,
                        "nuisance_solver": nuisance_solver,
                    }
                )
            except (FloatingPointError, ValueError) as error:
                profile_valid = False
                profiles.append(
                    {
                        "multiplier": multiplier,
                        "status": "failed",
                        "failure": str(error),
                    }
                )
        profile_increase = (
            min(float(row["relative_increase"]) for row in profiles)
            if profile_valid and all(row["status"] == "completed" for row in profiles)
            else None
        )
        condition = spectrum["condition_number"]
        rank_gate = bool(
            all_active
            and visible_columns
            and spectrum["sigma_ratio"] >= sigma_ratio_min
            and condition is not None
            and condition <= condition_max
        )
        profile_gate = bool(
            profile_increase is not None and profile_increase > profile_increase_min
        )
        results.append(
            {
                **descriptor,
                "parameter_slice": [section.start, section.stop],
                "weak_parameter_norm": float(np.linalg.norm(theta[section])),
                "all_module_columns_active": all_active,
                "all_module_columns_visible": visible_columns,
                "active_screen_floor": floor,
                **spectrum,
                "profile_multipliers": list(profile_multipliers),
                "profile_rows": profiles,
                "profile_loss_increase": profile_increase,
                "sigma_ratio_min": sigma_ratio_min,
                "condition_max": condition_max,
                "profile_increase_min": profile_increase_min,
                "conditional_rank_gate_passed": rank_gate,
                "profile_gate_passed": profile_gate,
                "parameter_identifiability_passed": rank_gate and profile_gate,
                "scientific_acceptance": False,
            }
        )
    return {
        "schema": "kron_reduce.p1_weak_module_identifiability.v1",
        "matrix_shape": list(matrix.shape),
        "matrix_sha256": _array_sha256(matrix),
        "vector_sha256": _array_sha256(vector),
        "ridge_rows_counted_as_observations": False,
        "selected_ridge": float(selected_ridge),
        "baseline_objective": baseline_total,
        "baseline_data_objective": baseline_data,
        "baseline_ridge_penalty": baseline_penalty,
        "baseline_solver": solver,
        "active_parameter_count": int(np.sum(active_mask)),
        "screened_parameter_count": int(np.sum(~active_mask)),
        "module_results": results,
        "scientific_acceptance": False,
    }


def audit_saved_model_identifiability(
    model: PrunableEnergyModel,
    fit_samples: Sequence[dict[str, Any]],
    manifest: P1RoleManifest,
    context: RunContext,
    *,
    selected_ridge: float,
) -> dict[str, Any]:
    """Bind the pure weak-system audit to one saved model and complete fit role."""
    validate_p1_samples(fit_samples, manifest, "fit")
    required_steps = max(len(sample["training_context"]["targets"]["t"]) for sample in fit_samples)
    if (
        type(context.seed) is not int
        or context.seed != 324
        or context.max_cases < len(fit_samples)
        or context.fit_max_steps < required_steps
        or context.horizon_steps < required_steps
        or tuple(context.integral_windows) != (1, 5, 20, 40, 80)
        or context.weak_max_windows_per_case_horizon != 8
        or context.max_internal_step != 0.01
    ):
        raise ValueError("fit context differs from the frozen complete P1 weak design")
    model._require_fitted()
    recorded_ridge = model.diagnostics.get("selected_ridge")
    if (
        not isinstance(recorded_ridge, (int, float))
        or isinstance(recorded_ridge, bool)
        or float(recorded_ridge) != float(selected_ridge)
    ):
        raise ValueError("selected ridge does not match the saved core")
    before = core_fingerprint(model)
    matrix, vector, system = model.build_weak_system(fit_samples, context)
    n_nodes, n_edges = model._require_fitted()
    layout = model._parameter_layout(n_nodes, n_edges)
    modules = [
        {"instance_id": row.instance_id, "family": row.family} for row in model.active_instances
    ]
    audit = audit_weak_module_blocks(
        matrix,
        vector,
        layout,
        modules,
        selected_ridge=selected_ridge,
    )
    if core_fingerprint(model) != before:
        raise RuntimeError("identifiability audit mutated its saved model")
    return {
        **audit,
        "role": "fit",
        "role_manifest_sha256": manifest.source_sha256,
        "fit_event_ids": list(manifest.ids("fit")),
        "model_fingerprint": before,
        "fit_context": asdict(context),
        "weak_system": system,
        "selection_data_reopened": False,
        "internal_audit_data_used": False,
        "scientific_acceptance": False,
    }
