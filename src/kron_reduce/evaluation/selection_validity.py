from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from kron_reduce.models.m0 import M0Model, _pm_equilibrium, _trapz


def _cell(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["amplitude_level"]), str(row["richness_level"])


def source_excitation_overlap_audit(
    rows: Iterable[dict[str, Any]], *, minimum_fault_line_clusters: int = 5
) -> dict[str, Any]:
    """Audit whether source and excitation effects can be separated.

    Only Q-A training trajectories are admissible.  A cell is directly
    comparable when every observed source occurs in it.  Every excitation cell
    represented in the design must contain the predeclared number of
    independent fault-line clusters per source.  Treating a source-exclusive
    high-excitation cell as harmless would recreate the confounding that this
    gate is intended to detect.
    """

    eligible = [
        row
        for row in rows
        if str(row.get("role")) == "train"
        and str(row.get("quality_grade")) == "Q-A"
        and bool(row.get("structure_selection_eligible", False))
    ]
    sources = sorted({str(row["source_dataset"]) for row in eligible})
    counts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    events: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in eligible:
        amplitude, richness = _cell(row)
        key = (str(row["source_dataset"]), amplitude, richness)
        counts[key].add(str(row["fault_line"]))
        events[key] += 1

    cells = sorted({_cell(row) for row in eligible})
    table = []
    comparable = []
    for amplitude, richness in cells:
        present = [
            source
            for source in sources
            if counts[(source, amplitude, richness)]
        ]
        if len(present) == len(sources) and sources:
            comparable.append((amplitude, richness))
        for source in sources:
            table.append(
                {
                    "source_dataset": source,
                    "amplitude_level": amplitude,
                    "richness_level": richness,
                    "n_events": events[(source, amplitude, richness)],
                    "n_fault_line_clusters": len(
                        counts[(source, amplitude, richness)]
                    ),
                    "comparable_cell": len(present) == len(sources),
                    "minimum_clusters_met": len(
                        counts[(source, amplitude, richness)]
                    )
                    >= minimum_fault_line_clusters,
                }
            )

    failing = [
        (source, amplitude, richness)
        for amplitude, richness in cells
        for source in sources
        if len(counts[(source, amplitude, richness)])
        < minimum_fault_line_clusters
    ]
    passed = bool(cells) and bool(comparable) and not failing
    return {
        "status": "PASS" if passed else "SOURCE_EXCITATION_CONFOUNDED",
        "minimum_fault_line_clusters": int(minimum_fault_line_clusters),
        "sources": sources,
        "n_eligible_events": len(eligible),
        "n_eligible_fault_line_clusters": len(
            {str(row["fault_line"]) for row in eligible}
        ),
        "comparable_cells": [
            {"amplitude_level": amplitude, "richness_level": richness}
            for amplitude, richness in comparable
        ],
        "failing_source_cells": [
            {
                "source_dataset": source,
                "amplitude_level": amplitude,
                "richness_level": richness,
                "n_fault_line_clusters": len(
                    counts[(source, amplitude, richness)]
                ),
            }
            for source, amplitude, richness in failing
        ],
        "table": table,
        "claim_allowed": passed,
        "interpretation": (
            "source-balanced excitation comparison is identifiable"
            if passed
            else "source and excitation cannot be separated by the available Q-A fault-line clusters"
        ),
    }


def _balanced_group_assignment(
    rows: Sequence[dict[str, Any]], n_folds: int
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("at least two folds are required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["fault_line"])].append(row)
    if len(groups) < n_folds:
        raise ValueError("fewer fault-line clusters than requested folds")

    strata = sorted(
        {
            (
                str(row["source_dataset"]),
                str(row["amplitude_level"]),
                str(row["richness_level"]),
            )
            for row in rows
        }
    )
    stratum_index = {value: index for index, value in enumerate(strata)}
    summaries: dict[str, tuple[int, np.ndarray]] = {}
    for line, group in groups.items():
        histogram = np.zeros(len(strata), dtype=float)
        for row in group:
            key = (
                str(row["source_dataset"]),
                str(row["amplitude_level"]),
                str(row["richness_level"]),
            )
            histogram[stratum_index[key]] += 1.0
        summaries[line] = (len(group), histogram)

    totals = np.zeros(n_folds, dtype=float)
    histograms = np.zeros((n_folds, len(strata)), dtype=float)
    target_total = len(rows) / n_folds
    target_histogram = np.sum(
        np.stack([value[1] for value in summaries.values()]), axis=0
    ) / n_folds
    assignment: dict[str, int] = {}
    ordered = sorted(
        summaries,
        key=lambda line: (
            -summaries[line][0],
            -float(np.max(summaries[line][1])),
            line,
        ),
    )
    for line in ordered:
        size, histogram = summaries[line]
        scores = []
        for fold in range(n_folds):
            trial_totals = totals.copy()
            trial_histograms = histograms.copy()
            trial_totals[fold] += size
            trial_histograms[fold] += histogram
            size_loss = np.sum(((trial_totals - target_total) / max(target_total, 1.0)) ** 2)
            scale = np.maximum(target_histogram, 1.0)
            stratum_loss = np.sum(((trial_histograms - target_histogram) / scale) ** 2)
            scores.append((float(size_loss + stratum_loss), totals[fold], fold))
        selected = min(scores)[2]
        assignment[line] = selected
        totals[selected] += size
        histograms[selected] += histogram
    return assignment


def nested_fault_line_folds(
    rows: Sequence[dict[str, Any]], *, outer_folds: int = 5, inner_folds: int = 4
) -> list[dict[str, Any]]:
    """Create deterministic nested group folds using Q-A train rows only."""

    eligible = [
        row
        for row in rows
        if str(row.get("role")) == "train"
        and str(row.get("quality_grade")) == "Q-A"
        and bool(row.get("structure_selection_eligible", False))
    ]
    outer = _balanced_group_assignment(eligible, outer_folds)
    output = []
    for outer_fold in range(outer_folds):
        development = [
            row for row in eligible if outer[str(row["fault_line"])] != outer_fold
        ]
        inner = _balanced_group_assignment(development, inner_folds)
        for row in eligible:
            line = str(row["fault_line"])
            is_outer = outer[line] == outer_fold
            output.append(
                {
                    "event_id": str(row["event_id"]),
                    "fault_line": line,
                    "source_dataset": str(row["source_dataset"]),
                    "amplitude_level": str(row["amplitude_level"]),
                    "richness_level": str(row["richness_level"]),
                    "outer_fold": outer_fold,
                    "outer_role": "evaluation" if is_outer else "development",
                    "inner_fold": None if is_outer else inner[line],
                }
            )
    return output


def equal_fault_line_paired_inference(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_draws: int = 5000,
    confidence: float = 0.95,
    random_sign_draws: int = 10000,
    seed: int = 324,
) -> dict[str, Any]:
    """Cluster-equal paired inference for comparator minus reference losses."""

    by_line: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        difference = float(row["comparator_minus_reference"])
        if np.isfinite(difference):
            by_line[str(row["fault_line"])].append(difference)
    cluster_means = np.asarray(
        [np.mean(by_line[line]) for line in sorted(by_line)], dtype=float
    )
    if len(cluster_means) < 2:
        return {
            "status": "INSUFFICIENT_INFORMATION",
            "n_fault_line_clusters": len(cluster_means),
        }
    rng = np.random.default_rng(seed)
    sampled = cluster_means[
        rng.integers(0, len(cluster_means), size=(bootstrap_draws, len(cluster_means)))
    ].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    observed = float(np.mean(cluster_means))
    if len(cluster_means) <= 15:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=len(cluster_means))))
        null = np.mean(signs * cluster_means[None, :], axis=1)
        sign_method = "exact"
        p_one_sided = float(np.sum(null >= observed) / len(null))
    else:
        signs = rng.choice(
            (-1.0, 1.0), size=(random_sign_draws, len(cluster_means))
        )
        null = np.mean(signs * cluster_means[None, :], axis=1)
        sign_method = "monte_carlo"
        p_one_sided = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return {
        "status": "completed",
        "estimand": "equal_fault_line_mean_of_within_line_event_mean_difference",
        "mean_difference": observed,
        "ci_lower": float(np.quantile(sampled, tail)),
        "ci_upper": float(np.quantile(sampled, 1.0 - tail)),
        "confidence": float(confidence),
        "bootstrap_draws": int(bootstrap_draws),
        "n_fault_line_clusters": len(cluster_means),
        "sign_flip_method": sign_method,
        "sign_flip_draws": len(null),
        "one_sided_p_value": p_one_sided,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values."""

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return adjusted


def complexity_record(
    *,
    spatial_coordinates: int,
    active_dynamic_states: int,
    active_parameters: int,
    active_edges: int,
) -> dict[str, int]:
    if min(
        spatial_coordinates,
        active_dynamic_states,
        active_parameters,
        active_edges,
    ) < 0:
        raise ValueError("complexity terms must be nonnegative")
    return {
        "spatial_coordinates": int(spatial_coordinates),
        "base_state_count": int(2 * spatial_coordinates),
        "active_dynamic_states": int(active_dynamic_states),
        "total_state_count": int(2 * spatial_coordinates + active_dynamic_states),
        "active_parameters": int(active_parameters),
        "active_edges": int(active_edges),
    }


def complexity_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["total_state_count"]),
        int(record["active_parameters"]),
        int(record["active_edges"]),
        int(record["spatial_coordinates"]),
    )


def structure_selection_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    """The sole admissible ID set for order, partition and support discovery."""

    return {
        str(row["event_id"])
        for row in rows
        if str(row.get("role")) == "train"
        and str(row.get("quality_grade")) == "Q-A"
        and bool(row.get("structure_selection_eligible", False))
    }


def _joint_inertia_windows(
    samples: Sequence[dict[str, Any]],
    *,
    windows: tuple[int, ...],
    max_windows_per_case_horizon: int,
) -> tuple[list[dict[str, np.ndarray]], int, int]:
    blocks: list[dict[str, np.ndarray]] = []
    n_nodes = int(np.asarray(samples[0]["model_inputs"]["static"]["mass"]).size)
    n_edges = int(
        np.asarray(samples[0]["model_inputs"]["static"]["incidence"]).shape[1]
    )
    for sample in samples:
        target = sample["training_context"]["targets"]
        time = np.asarray(target["t"], dtype=float)
        delta = np.asarray(target["delta"], dtype=float)
        omega = np.asarray(target["omega"], dtype=float)
        pm = np.asarray(sample["training_context"]["Pm_observed"], dtype=float)
        static = sample["model_inputs"]["static"]
        incidence = np.asarray(static["incidence"], dtype=float)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
        delta_pm = pm - _pm_equilibrium(static)
        eta = delta @ incidence
        eta_eq = incidence.T @ equilibrium
        sine = np.sin(eta) - np.sin(eta_eq)
        widths = [value for value in windows if 0 < value < len(time)]
        for width in widths:
            starts = np.arange(len(time) - width)
            if len(starts) > max_windows_per_case_horizon:
                starts = np.unique(
                    np.linspace(
                        0,
                        len(starts) - 1,
                        max_windows_per_case_horizon,
                    )
                    .round()
                    .astype(int)
                )
            weight = 1.0 / np.sqrt(max(len(widths) * len(starts) * n_nodes, 1))
            for start in starts:
                stop = int(start + width)
                segment = slice(int(start), stop + 1)
                blocks.append(
                    {
                        "delta_omega": weight * (omega[stop] - omega[int(start)]),
                        "pm": weight * _trapz(delta_pm[segment], time[segment]),
                        "omega": weight * _trapz(omega[segment], time[segment]),
                        "sine": weight * _trapz(sine[segment], time[segment]),
                        "incidence": incidence,
                    }
                )
    return blocks, n_nodes, n_edges


def _finite_difference_jacobian(
    function: Any, point: np.ndarray
) -> np.ndarray:
    """Central-difference Jacobian used only for post-fit identifiability.

    The rank audit must not count ridge rows as observations.  Keeping this
    small helper here avoids relying on SciPy's private numerical-difference
    API and makes the audited matrix explicit.
    """

    value = np.asarray(function(point), dtype=float)
    jacobian = np.empty((value.size, point.size), dtype=float)
    for index in range(point.size):
        step = np.sqrt(np.finfo(float).eps) * max(abs(point[index]), 1.0)
        upper = point.copy()
        lower = point.copy()
        upper[index] += step
        lower[index] = max(lower[index] - step, np.finfo(float).eps)
        denominator = upper[index] - lower[index]
        jacobian[:, index] = (
            np.asarray(function(upper), dtype=float)
            - np.asarray(function(lower), dtype=float)
        ) / denominator
    return jacobian


def profile_joint_inertia(
    samples: Sequence[dict[str, Any]],
    *,
    windows: tuple[int, ...] = (1, 5, 20),
    max_windows_per_case_horizon: int = 8,
    ridge: float = 1e-6,
    maximum_condition_number: float = 1e8,
) -> dict[str, Any]:
    """Profile positive effective inertia, damping and coupling in COI coordinates.

    This is an identifiability audit.  It does not mutate samples or promote a
    learned mass into rollout.  The residual includes the mass-dependent COI
    projection exactly for each weak window.
    """

    if not samples:
        raise ValueError("joint inertia profiling requires samples")
    blocks, n_nodes, _n_edges = _joint_inertia_windows(
        samples,
        windows=windows,
        max_windows_per_case_horizon=max_windows_per_case_horizon,
    )
    if not blocks:
        raise ValueError("no joint inertia windows")
    prior_mass = np.asarray(
        samples[0]["model_inputs"]["static"]["mass"], dtype=float
    ).reshape(-1)
    matrix, vector, _ = M0Model.build_weak_system(
        list(samples),
        windows=windows,
        horizon_steps=max(windows) + 1,
        max_fit_steps=max(windows) + 2,
        max_windows_per_case_horizon=max_windows_per_case_horizon,
    )
    initial_dk, _ = M0Model._solve_nonnegative_ridge(matrix, vector, ridge)
    initial = np.maximum(
        np.concatenate((prior_mass, initial_dk)),
        np.sqrt(np.finfo(float).eps),
    )

    def data_residual(theta: np.ndarray) -> np.ndarray:
        mass = theta[:n_nodes]
        damping = theta[n_nodes : 2 * n_nodes]
        coupling = theta[2 * n_nodes :]
        total_mass = float(np.sum(mass))
        values = []
        for block in blocks:
            raw = (
                block["pm"]
                - damping * block["omega"]
                - block["incidence"] @ (coupling * block["sine"])
            )
            projected = raw - mass * float(np.sum(raw)) / total_mass
            values.append(mass * block["delta_omega"] - projected)
        return np.concatenate(values)

    def residual(theta: np.ndarray) -> np.ndarray:
        regularization = np.sqrt(ridge) * np.concatenate(
            (theta[n_nodes : 2 * n_nodes], theta[2 * n_nodes :])
        )
        return np.concatenate((data_residual(theta), regularization))

    result = least_squares(
        residual,
        initial,
        bounds=(np.finfo(float).eps, np.inf),
        x_scale="jac",
        max_nfev=3000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    jacobian = _finite_difference_jacobian(data_residual, result.x)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / max(singular[-1], np.finfo(float).eps))
    n_parameters = len(result.x)
    identified = bool(
        result.success
        and rank == n_parameters
        and np.isfinite(condition)
        and condition < maximum_condition_number
    )
    mass = result.x[:n_nodes]
    damping = result.x[n_nodes : 2 * n_nodes]
    coupling = result.x[2 * n_nodes :]
    return {
        "status": "IDENTIFIABLE" if identified else "PARAMETER_NOT_IDENTIFIABLE",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "n_weak_windows": len(blocks),
        "n_parameters": n_parameters,
        "effective_rank": rank,
        "condition_number": condition,
        "maximum_condition_number": float(maximum_condition_number),
        "weak_residual_rmse": float(
            np.sqrt(np.mean(data_residual(result.x) ** 2))
        ),
        "effective_mass": mass.tolist(),
        "effective_damping": damping.tolist(),
        "effective_coupling": coupling.tolist(),
        "prior_mass": prior_mass.tolist(),
        "relative_mass_change": float(
            np.linalg.norm(mass - prior_mass)
            / max(np.linalg.norm(prior_mass), np.finfo(float).eps)
        ),
        "claim_boundary": (
            "coordinate-conditioned effective inertia audit; no machine inertia recovery claim"
        ),
    }
