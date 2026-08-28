"""Fast, data-level decision for using one clustered representation.

The selector consumes one trajectory corpus and returns one decision.  It
does not stratify trajectories into excitation regimes and it never routes
different events to different models.  A cheap integral weak-form B2 fit is
used only to expose the synchronising operator needed by the screen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from kron_reduce.models.base import RunContext
from kron_reduce.models.m0 import M0Model
from kron_reduce.physics.aggregation import (
    cluster_aggregation_matrices,
    reconstruct_cluster_state,
    validate_partition,
)
from kron_reduce.physics.coherency import (
    almost_equitable_defect,
    generalized_laplacian_spectrum,
    node_pairs,
    normalized_dynamic_pair_costs,
    normalized_forcing_pair_costs,
    synchronizing_laplacian,
)
from kron_reduce.physics.graph_quotient import (
    conditional_predictive_information_gain,
    input_heterogeneity,
    intra_cluster_energy,
)
from kron_reduce.physics.multiresolution import aggregate_model_sample

Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ClusterThresholds:
    information: float = 0.02
    input: float = 0.02
    mode: float = 0.05
    spectral_gap_ratio: float = 0.25
    state: float = 0.02
    coupling: float = 0.05
    closure: float = 0.02
    bootstrap_support: float = 0.80
    confidence: float = 0.95

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not np.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("cluster thresholds must be finite and positive")
        if self.bootstrap_support > 1.0 or self.confidence >= 1.0:
            raise ValueError("support and confidence must be smaller than one")


@dataclass(frozen=True)
class ClusterDecision:
    use_clustering: bool
    n_original: int
    n_clusters: int
    partition: Partition
    score: float
    components: dict[str, float]
    bootstrap_support: float
    selected_rank: int
    candidate_scores: tuple[dict[str, Any], ...]
    spectrum: tuple[float, ...]
    roles: dict[str, tuple[str, ...]]
    method: str = "single_corpus_cluster_loss_v1"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["partition"] = [list(group) for group in self.partition]
        result["candidate_scores"] = [dict(row) for row in self.candidate_scores]
        return result


@dataclass(frozen=True)
class ClusterAuditResult:
    """Post-decision evidence computed without changing the frozen decision."""

    seed: int
    audit_groups: tuple[str, ...]
    candidate_audits: tuple[dict[str, Any], ...]
    baseline_audits: tuple[dict[str, Any], ...]
    rollout_horizons_s: tuple[float, ...]
    rollout_budget: float
    decision_unchanged: bool
    method: str = "frozen_cluster_decision_audit_v1"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_audits"] = [dict(row) for row in self.candidate_audits]
        result["baseline_audits"] = [dict(row) for row in self.baseline_audits]
        return result


def _group(sample: dict[str, Any]) -> str:
    metadata = sample.get("training_context", {}).get("metadata", {})
    value = metadata.get("fault_line") or metadata.get("independent_group")
    return str(value if value not in {None, ""} else sample["event_id"])


def data_level_group_split(
    samples: Sequence[dict[str, Any]],
    *,
    seed: int = 324,
    fit_fraction: float = 0.60,
    selection_fraction: float = 0.20,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, ...]]]:
    """Split independent event groups without using excitation strata."""

    if not samples:
        raise ValueError("data-level selection requires nonempty samples")
    if fit_fraction <= 0 or selection_fraction <= 0 or fit_fraction + selection_fraction >= 1:
        raise ValueError("fit and selection fractions must leave an audit split")
    groups = sorted({_group(sample) for sample in samples})
    if len(groups) < 3:
        raise ValueError("at least three independent trajectory groups are required")
    rng = np.random.default_rng(int(seed))
    ordered = np.asarray(groups, dtype=object)[rng.permutation(len(groups))].tolist()
    n_fit = min(max(1, round(float(fit_fraction) * len(groups))), len(groups) - 2)
    n_selection = min(
        max(1, round(float(selection_fraction) * len(groups))),
        len(groups) - n_fit - 1,
    )
    roles = {
        "fit": tuple(sorted(str(value) for value in ordered[:n_fit])),
        "selection": tuple(
            sorted(str(value) for value in ordered[n_fit : n_fit + n_selection])
        ),
        "audit": tuple(sorted(str(value) for value in ordered[n_fit + n_selection :])),
    }
    output = {
        role: [sample for sample in samples if _group(sample) in set(role_groups)]
        for role, role_groups in roles.items()
    }
    return output, roles


def _common_static(samples: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = samples[0]["model_inputs"]["static"]
    mass = np.asarray(first["mass"], dtype=float).reshape(-1)
    incidence = np.asarray(first["incidence"], dtype=float)
    equilibrium = np.asarray(first["equilibrium_delta"], dtype=float).reshape(-1)
    for sample in samples[1:]:
        static = sample["model_inputs"]["static"]
        if not (
            np.allclose(static["mass"], mass)
            and np.allclose(static["incidence"], incidence)
            and np.allclose(static["equilibrium_delta"], equilibrium)
        ):
            raise ValueError("cluster selection requires a common coordinate system")
    return mass, incidence, equilibrium


def _fit_b2(
    fit: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    *,
    seed: int,
    max_steps: int,
) -> M0Model:
    horizon = min(max_steps, min(len(sample["training_context"]["targets"]["t"]) for sample in fit))
    context = RunContext(
        seed=int(seed),
        horizon_steps=max(2, int(horizon)),
        max_cases=len(fit),
        ridge_values=(1e-10, 1e-8, 1e-6, 1e-4),
        integral_windows=(1, 5, 20),
        fit_max_steps=int(horizon),
        weak_max_windows_per_case_horizon=8,
        max_internal_step=0.01,
    )
    def causalized(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for sample in values:
            copied = deepcopy(sample)
            copied["training_context"]["Pm_observed"] = np.asarray(
                copied["model_inputs"]["exogenous"]["Pm"], dtype=float
            ).copy()
            output.append(copied)
        return output

    fit_causal = causalized(fit)
    selection_causal = causalized(selection or fit)
    model = M0Model(max_internal_step=0.01)
    matrix, vector, diagnostics = model.build_weak_system(
        fit_causal,
        windows=context.integral_windows,
        max_cases=context.max_cases,
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
        max_windows_per_case_horizon=context.weak_max_windows_per_case_horizon,
    )
    validation_matrix, validation_vector, _ = model.build_weak_system(
        selection_causal,
        windows=context.integral_windows,
        max_cases=len(selection_causal),
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
        max_windows_per_case_horizon=context.weak_max_windows_per_case_horizon,
    )
    n_nodes = len(fit_causal[0]["model_inputs"]["static"]["mass"])
    candidates = []
    for ridge in context.ridge_values:
        theta, solve = model._solve_nonnegative_ridge(matrix, vector, float(ridge))
        validation_loss = float(
            np.sqrt(np.mean((validation_matrix @ theta - validation_vector) ** 2))
        )
        candidates.append((validation_loss, float(ridge), theta, solve))
    loss, ridge, theta, solve = min(candidates, key=lambda row: (row[0], row[1]))
    model.damping = theta[:n_nodes]
    model.coupling = theta[n_nodes:]
    model.selected_ridge = ridge
    model.diagnostics = {
        **diagnostics,
        **solve,
        "selected_ridge": ridge,
        "selection_weak_residual_rmse": loss,
        "long_rollout_used": False,
        "purpose": "clusterability_screen_only",
    }
    return model


def _mass_energy(values: np.ndarray, mass: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sum(array**2 * mass.reshape((1,) * (array.ndim - 1) + (-1,))))


def _project(values: np.ndarray, projector: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) @ np.asarray(projector, dtype=float).T


def _delay_blocks(values: np.ndarray, delays: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    use = min(max(1, int(delays)), max(1, len(array) - 1))
    if len(array) <= use:
        return array.reshape(1, -1)
    return np.stack([array[index : index + use].reshape(-1) for index in range(len(array) - use + 1)])


def _active_spatial_modes(
    samples: Sequence[dict[str, Any]], mass: np.ndarray, *, max_steps: int
) -> np.ndarray:
    root = np.sqrt(mass)
    blocks = []
    for sample in samples:
        target = sample["training_context"]["targets"]
        use = min(int(max_steps), len(target["t"]))
        delta = np.asarray(target["delta"][:use], dtype=float)
        omega = np.asarray(target["omega"][:use], dtype=float)
        delta = delta - delta[[0]]
        delta_scale = max(float(np.std(delta)), 1e-12)
        omega_scale = max(float(np.std(omega)), 1e-12)
        blocks.extend((delta / delta_scale * root, omega / omega_scale * root))
    _, singular, vt = np.linalg.svd(np.concatenate(blocks, axis=0), full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        return np.empty((len(mass), 0))
    keep = np.flatnonzero(singular >= 0.10 * singular[0])
    keep = keep[: max(1, len(mass) - 1)]
    return vt[keep].T / root[:, None]


def _soft_components(
    samples: Sequence[dict[str, Any]],
    mass: np.ndarray,
    projector: np.ndarray,
    active_modes: np.ndarray,
    gap_ratio: float,
    *,
    max_steps: int,
    delays: int,
) -> dict[str, float]:
    total_information = hidden_information = 0.0
    total_input = hidden_input = 0.0
    root = np.sqrt(mass)
    for sample in samples:
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        use = min(int(max_steps), len(target["t"]))
        delta = np.asarray(target["delta"][:use], dtype=float)
        omega = np.asarray(target["omega"][:use], dtype=float)
        delta = delta - delta[[0]]
        scale_d = max(float(np.std(delta)), 1e-12)
        scale_w = max(float(np.std(omega)), 1e-12)
        state = np.concatenate((delta / scale_d, omega / scale_w), axis=1)
        retained = np.concatenate(
            (_project(delta, projector) / scale_d, _project(omega, projector) / scale_w),
            axis=1,
        )
        weights = np.concatenate((root, root))
        full_h = _delay_blocks(state * weights[None, :], delays)
        hidden_h = _delay_blocks((state - retained) * weights[None, :], delays)
        total_information += float(np.sum(full_h**2))
        hidden_information += float(np.sum(hidden_h**2))

        pm = np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:use], dtype=float)
        equilibrium_pm = np.asarray(static["Pm_equilibrium"], dtype=float).reshape(-1)
        acceleration = (pm - equilibrium_pm[None, :]) / mass[None, :]
        total_input += _mass_energy(acceleration, mass)
        hidden_input += _mass_energy(acceleration - _project(acceleration, projector), mass)
    return {
        "information_loss": hidden_information / max(total_information, np.finfo(float).eps),
        "input_loss": hidden_input / max(total_input, np.finfo(float).eps),
        "mode_loss": _soft_mode_loss(active_modes, mass, projector),
        "spectral_gap_ratio": float(gap_ratio),
    }


def _soft_mode_loss(modes: np.ndarray, mass: np.ndarray, projector: np.ndarray) -> float:
    if modes.size == 0:
        return 0.0
    residual = modes - projector @ modes
    numerator = np.sum(residual**2 * mass[:, None], axis=0)
    denominator = np.sum(modes**2 * mass[:, None], axis=0)
    return float(np.max(numerator / np.maximum(denominator, np.finfo(float).eps)))


def _soft_score(components: dict[str, float], thresholds: ClusterThresholds) -> float:
    return float(
        max(
            components["information_loss"] / thresholds.information,
            components["input_loss"] / thresholds.input,
            components["mode_loss"] / thresholds.mode,
            components["spectral_gap_ratio"] / thresholds.spectral_gap_ratio,
        )
    )


def _normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    positive = array[array > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return array / max(scale, np.finfo(float).eps)


def _partition_from_distance(distance: np.ndarray, n_clusters: int) -> Partition:
    n_nodes = len(distance)
    clusters: list[tuple[int, ...]] = [(index,) for index in range(n_nodes)]
    while len(clusters) > int(n_clusters):
        candidates = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                cross = distance[np.ix_(clusters[left], clusters[right])]
                candidates.append(
                    (float(np.mean(cross)), clusters[left], clusters[right], left, right)
                )
        _, group_left, group_right, left, right = min(
            candidates, key=lambda row: (row[0], row[1], row[2])
        )
        merged = tuple(sorted((*group_left, *group_right)))
        clusters = [
            group for index, group in enumerate(clusters) if index not in {left, right}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda group: (group[0], len(group), group))
    return tuple(clusters)


def _hard_partition(
    samples: Sequence[dict[str, Any]],
    mass: np.ndarray,
    eigenvectors: np.ndarray,
    n_clusters: int,
    *,
    max_steps: int,
) -> Partition:
    n_nodes = len(mass)
    modal_distance = np.zeros((n_nodes, n_nodes), dtype=float)
    embedding = eigenvectors[:, :n_clusters] * np.sqrt(mass)[:, None]
    dynamic_rows = []
    input_rows = []
    for sample in samples:
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        use = min(int(max_steps), len(target["t"]))
        dynamic_rows.append(
            normalized_dynamic_pair_costs(
                np.asarray(target["delta"][:use]), np.asarray(target["omega"][:use]), mass
            )
        )
        input_rows.append(
            normalized_forcing_pair_costs(
                np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:use]),
                np.asarray(static["Pm_equilibrium"]),
                mass,
            )
        )
    dynamic = np.median(np.stack(dynamic_rows), axis=0)
    forcing = np.median(np.stack(input_rows), axis=0)
    for index, (left, right) in enumerate(node_pairs(n_nodes)):
        modal_distance[left, right] = modal_distance[right, left] = float(
            np.sum((embedding[left] - embedding[right]) ** 2)
        )
    dynamic_matrix = np.zeros_like(modal_distance)
    forcing_matrix = np.zeros_like(modal_distance)
    for value, (left, right) in zip(dynamic, node_pairs(n_nodes), strict=True):
        dynamic_matrix[left, right] = dynamic_matrix[right, left] = float(value)
    for value, (left, right) in zip(forcing, node_pairs(n_nodes), strict=True):
        forcing_matrix[left, right] = forcing_matrix[right, left] = float(value)
    distance = (
        _normalized(modal_distance)
        + _normalized(dynamic_matrix)
        + _normalized(forcing_matrix)
    ) / 3.0
    return _partition_from_distance(distance, int(n_clusters))


def _paths(
    samples: Sequence[dict[str, Any]], *, max_steps: int
) -> list[dict[str, Any]]:
    output = []
    for sample in samples:
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        use = min(int(max_steps), len(target["t"]))
        output.append(
            {
                "event_id": str(sample["event_id"]),
                "fault_line": _group(sample),
                "delta": np.asarray(target["delta"][:use], dtype=float),
                "omega": np.asarray(target["omega"][:use], dtype=float),
                "u": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:use], dtype=float)
                - np.asarray(static["Pm_equilibrium"], dtype=float)[None, :],
            }
        )
    return output


def _hard_components(
    fit: Sequence[dict[str, Any]],
    selection: Sequence[dict[str, Any]],
    mass: np.ndarray,
    laplacian: np.ndarray,
    partition: Partition,
    active_modes: np.ndarray,
    *,
    max_steps: int,
    seed: int,
    cpig_bootstrap_draws: int = 100,
) -> dict[str, float]:
    fit_paths = _paths(fit, max_steps=max_steps)
    selection_paths = _paths(selection or fit, max_steps=max_steps)
    energy = intra_cluster_energy(selection_paths, mass, partition)
    time = np.asarray(fit[0]["training_context"]["targets"]["t"], dtype=float)
    dt = float(np.median(np.diff(time[: min(len(time), max_steps)])))
    horizon_steps = tuple(
        sorted(
            {
                min(max(1, round(seconds / dt)), max(1, max_steps // 2))
                for seconds in (0.05, 0.2, 1.0)
            }
        )
    )
    cpig_rows = conditional_predictive_information_gain(
        fit_paths,
        selection_paths,
        mass,
        partition,
        horizons=horizon_steps,
        delays=2,
        rank_cap=min(12, 2 * len(mass)),
        max_rows_per_path=32,
        bootstrap_draws=max(1, int(cpig_bootstrap_draws)),
        seed=int(seed),
    )
    cpig = max(
        [max(0.0, float(row["cpig"])) for row in cpig_rows if row["status"] == "completed"],
        default=float("inf"),
    )
    return {
        "state_loss": max(float(energy["delta"]), float(energy["omega"])),
        "input_loss": float(input_heterogeneity(selection_paths, mass, partition)),
        "coupling_loss": float(almost_equitable_defect(laplacian, mass, partition)),
        "closure_gain": float(cpig),
        "mode_loss": _soft_mode_loss(
            active_modes,
            mass,
            cluster_aggregation_matrices(mass, partition)[2],
        ),
    }


def _hard_score(components: dict[str, float], thresholds: ClusterThresholds) -> float:
    return float(
        max(
            components["state_loss"] / thresholds.state,
            components["input_loss"] / thresholds.input,
            components["coupling_loss"] / thresholds.coupling,
            components["closure_gain"] / thresholds.closure,
            components["mode_loss"] / thresholds.mode,
        )
    )


def _rollout_prefix(
    model: M0Model,
    sample: dict[str, Any],
    maximum_horizon_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(sample["model_inputs"]["exogenous"]["t"], dtype=float)
    stop = min(len(time), max(2, int(np.searchsorted(time, maximum_horizon_s, side="right"))))
    use_time = time[:stop]
    static = {
        key: np.asarray(value, dtype=float)
        for key, value in sample["model_inputs"]["static"].items()
    }
    pm = np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:stop], dtype=float)
    delta, omega = model.rollout(
        static,
        use_time,
        {"Pm": pm, "interpolation": "zoh"},
    )
    if not (np.isfinite(delta).all() and np.isfinite(omega).all()):
        raise RuntimeError(f"non-finite B2 rollout for {sample['event_id']}")
    return use_time, np.asarray(delta, dtype=float), np.asarray(omega, dtype=float)


def _full_rollout_cache(
    model: M0Model,
    samples: Sequence[dict[str, Any]],
    maximum_horizon_s: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | Exception]:
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | Exception] = {}
    for sample in samples:
        try:
            cache[str(sample["event_id"])] = _rollout_prefix(
                model, sample, maximum_horizon_s
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            cache[str(sample["event_id"])] = error
    return cache


def _rollout_penalties(
    fit: Sequence[dict[str, Any]],
    selection: Sequence[dict[str, Any]],
    audit: Sequence[dict[str, Any]],
    full_rollouts: Mapping[
        str, tuple[np.ndarray, np.ndarray, np.ndarray] | Exception
    ],
    mass: np.ndarray,
    partition: Partition,
    *,
    seed: int,
    max_steps: int,
    horizons_s: tuple[float, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]], int, str | None]:
    keys = tuple(f"{value:g}s" for value in horizons_s)
    failed = 0
    error_text: str | None = None
    totals = {
        key: {
            "delta_cluster_sse": 0.0,
            "delta_full_sse": 0.0,
            "delta_truth_sst": 0.0,
            "omega_cluster_sse": 0.0,
            "omega_full_sse": 0.0,
            "omega_truth_sst": 0.0,
        }
        for key in keys
    }
    try:
        reduced_fit = [
            aggregate_model_sample(sample, partition, f"cluster_audit_{len(partition)}")
            for sample in fit
        ]
        reduced_selection = [
            aggregate_model_sample(sample, partition, f"cluster_audit_{len(partition)}")
            for sample in selection
        ]
        reduced_audit = {
            str(sample["event_id"]): aggregate_model_sample(
                sample, partition, f"cluster_audit_{len(partition)}"
            )
            for sample in audit
        }
        reduced_model = _fit_b2(
            reduced_fit,
            reduced_selection,
            seed=int(seed),
            max_steps=int(max_steps),
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return (
            {key: float("inf") for key in keys},
            {key: {"delta": float("inf"), "omega": float("inf")} for key in keys},
            len(audit),
            str(error),
        )

    for sample in audit:
        event_id = str(sample["event_id"])
        full = full_rollouts[event_id]
        if isinstance(full, Exception):
            failed += 1
            error_text = str(full)
            continue
        try:
            time, full_delta, full_omega = full
            _, reduced_delta, reduced_omega = _rollout_prefix(
                reduced_model, reduced_audit[event_id], max(horizons_s)
            )
            cluster_delta = reconstruct_cluster_state(reduced_delta, mass, partition)
            cluster_omega = reconstruct_cluster_state(reduced_omega, mass, partition)
            target = sample["training_context"]["targets"]
            true_delta = np.asarray(target["delta"][: len(time)], dtype=float)
            true_omega = np.asarray(target["omega"][: len(time)], dtype=float)
            for horizon, key in zip(horizons_s, keys, strict=True):
                stop = min(
                    len(time),
                    max(2, int(np.searchsorted(time, horizon, side="right"))),
                )
                true_delta_increment = true_delta[:stop] - true_delta[[0]]
                true_omega_increment = true_omega[:stop] - true_omega[[0]]
                totals[key]["delta_cluster_sse"] += _mass_energy(
                    cluster_delta[:stop] - true_delta[:stop], mass
                )
                totals[key]["delta_full_sse"] += _mass_energy(
                    full_delta[:stop] - true_delta[:stop], mass
                )
                totals[key]["delta_truth_sst"] += _mass_energy(
                    true_delta_increment, mass
                )
                totals[key]["omega_cluster_sse"] += _mass_energy(
                    cluster_omega[:stop] - true_omega[:stop], mass
                )
                totals[key]["omega_full_sse"] += _mass_energy(
                    full_omega[:stop] - true_omega[:stop], mass
                )
                totals[key]["omega_truth_sst"] += _mass_energy(
                    true_omega_increment, mass
                )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failed += 1
            error_text = str(error)

    components: dict[str, dict[str, float]] = {}
    losses: dict[str, float] = {}
    if failed:
        return (
            {key: float("inf") for key in keys},
            {key: {"delta": float("inf"), "omega": float("inf")} for key in keys},
            failed,
            error_text,
        )
    for key in keys:
        row = totals[key]
        delta_loss = max(
            0.0, row["delta_cluster_sse"] - row["delta_full_sse"]
        ) / max(row["delta_truth_sst"], np.finfo(float).eps)
        omega_loss = max(
            0.0, row["omega_cluster_sse"] - row["omega_full_sse"]
        ) / max(row["omega_truth_sst"], np.finfo(float).eps)
        components[key] = {"delta": float(delta_loss), "omega": float(omega_loss)}
        losses[key] = float(max(delta_loss, omega_loss))
    return losses, components, failed, error_text


def audit_cluster_decision(
    samples: Sequence[dict[str, Any]],
    decision: ClusterDecision,
    *,
    thresholds: ClusterThresholds | None = None,
    seed: int = 324,
    max_steps: int = 1001,
    rollout_horizons_s: Sequence[float] = (0.2, 1.0, 5.0),
    rollout_budget: float = 0.02,
    cpig_bootstrap_draws: int = 100,
    baseline_partitions: Mapping[str, Partition] | None = None,
) -> ClusterAuditResult:
    """Audit a frozen decision on held-out groups without changing its route.

    The synchronising spectrum and every candidate partition are reconstructed
    from the decision's fit/selection roles and checked against the stored
    spectrum before audit-group metrics are evaluated.
    """

    materialized = list(samples)
    thresholds = thresholds or ClusterThresholds()
    horizons = tuple(sorted({float(value) for value in rollout_horizons_s}))
    if not horizons or any(not np.isfinite(value) or value <= 0.0 for value in horizons):
        raise ValueError("rollout horizons must be finite and positive")
    if not np.isfinite(rollout_budget) or rollout_budget <= 0.0:
        raise ValueError("rollout budget must be finite and positive")
    if int(cpig_bootstrap_draws) < 1:
        raise ValueError("CPIG bootstrap draws must be positive")
    before = deepcopy(decision.to_dict())
    split, roles = data_level_group_split(materialized, seed=int(seed))
    if roles != decision.roles:
        raise ValueError("audit seed/roles do not match the frozen cluster decision")
    fit, selection, audit = split["fit"], split["selection"], split["audit"]
    if not audit:
        raise ValueError("frozen cluster audit requires nonempty audit groups")
    mass, incidence, equilibrium = _common_static(materialized)
    if len(mass) != decision.n_original:
        raise ValueError("decision node count does not match the supplied corpus")
    baseline = _fit_b2(fit, selection, seed=int(seed), max_steps=int(max_steps))
    laplacian, _ = synchronizing_laplacian(
        incidence, np.asarray(baseline.coupling), equilibrium
    )
    eigenvalues, eigenvectors = generalized_laplacian_spectrum(laplacian, mass)
    if len(decision.spectrum) != len(eigenvalues) or not np.allclose(
        eigenvalues,
        np.asarray(decision.spectrum),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("frozen synchronising spectrum could not be reproduced")
    audit_modes = _active_spatial_modes(audit, mass, max_steps=int(max_steps))
    full_rollouts = _full_rollout_cache(baseline, audit, max(horizons))
    decision_rows = {
        int(row["n_clusters"]): row for row in decision.candidate_scores
    }

    def evaluate(
        name: str,
        kind: str,
        partition: Partition,
        soft_row: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        canonical = validate_partition(partition, len(mass))
        error: str | None = None
        try:
            components = _hard_components(
                fit,
                audit,
                mass,
                laplacian,
                canonical,
                audit_modes,
                max_steps=int(max_steps),
                seed=int(seed),
                cpig_bootstrap_draws=int(cpig_bootstrap_draws),
            )
            hard_score = _hard_score(components, thresholds)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as caught:
            components = {
                "state_loss": float("inf"),
                "input_loss": float("inf"),
                "coupling_loss": float("inf"),
                "closure_gain": float("inf"),
                "mode_loss": float("inf"),
            }
            hard_score = float("inf")
            error = str(caught)
        rollout, rollout_components, failures, rollout_error = _rollout_penalties(
            fit,
            selection,
            audit,
            full_rollouts,
            mass,
            canonical,
            seed=int(seed),
            max_steps=int(max_steps),
            horizons_s=horizons,
        )
        error = error or rollout_error
        hard_pass = bool(np.isfinite(hard_score) and hard_score <= 1.0)
        rollout_pass = bool(
            failures == 0
            and all(np.isfinite(value) and value <= rollout_budget for value in rollout.values())
        )
        result = {
            "name": str(name),
            "kind": str(kind),
            "n_clusters": len(canonical),
            "partition": [list(group) for group in canonical],
            "hard_components": components,
            "hard_score": float(hard_score),
            "hard_pass": hard_pass,
            "rollout_loss": rollout,
            "rollout_components": rollout_components,
            "maximum_rollout_loss": max(rollout.values(), default=float("inf")),
            "rollout_failure_count": int(failures),
            "rollout_pass": rollout_pass,
            "passes_all": bool(hard_pass and rollout_pass),
            "audit_groups": list(roles["audit"]),
            "error": error,
        }
        if soft_row is not None:
            result.update(
                {
                    "point_score": float(soft_row["point_score"]),
                    "bootstrap_upper": float(soft_row["bootstrap_upper"]),
                    "bootstrap_support": float(soft_row["bootstrap_support"]),
                    "soft_pass": bool(soft_row["soft_pass"]),
                    "soft_components": dict(soft_row["soft_components"]),
                }
            )
        return result

    candidate_audits = []
    for n_clusters in range(2, len(mass)):
        partition = _hard_partition(
            fit,
            mass,
            eigenvectors,
            n_clusters,
            max_steps=int(max_steps),
        )
        stored = decision_rows.get(n_clusters)
        if stored is None:
            raise RuntimeError(f"frozen decision omitted candidate rank {n_clusters}")
        stored_partition = stored.get("partition")
        if stored_partition is not None and tuple(
            tuple(int(node) for node in group) for group in stored_partition
        ) != partition:
            raise RuntimeError(f"candidate partition {n_clusters} changed after freeze")
        candidate_audits.append(
            evaluate(f"candidate_k{n_clusters}", "selector_candidate", partition, stored)
        )

    baseline_audits = []
    for name, partition in (baseline_partitions or {}).items():
        baseline_audits.append(evaluate(str(name), "reference_baseline", partition, None))

    unchanged = before == decision.to_dict()
    if not unchanged:
        raise RuntimeError("cluster decision changed during audit")
    return ClusterAuditResult(
        seed=int(seed),
        audit_groups=roles["audit"],
        candidate_audits=tuple(candidate_audits),
        baseline_audits=tuple(baseline_audits),
        rollout_horizons_s=horizons,
        rollout_budget=float(rollout_budget),
        decision_unchanged=unchanged,
    )


def decide_clustering(
    samples: Sequence[dict[str, Any]],
    *,
    thresholds: ClusterThresholds | None = None,
    seed: int = 324,
    bootstrap_draws: int = 200,
    max_steps: int = 1001,
    delays: int = 12,
) -> ClusterDecision:
    """Return one clustering decision for the entire supplied corpus."""

    materialized = list(samples)
    thresholds = thresholds or ClusterThresholds()
    split, roles = data_level_group_split(materialized, seed=int(seed))
    fit = split["fit"]
    selection = split["selection"]
    mass, incidence, equilibrium = _common_static(materialized)
    n_nodes = len(mass)
    identity: Partition = tuple((index,) for index in range(n_nodes))
    if n_nodes < 3:
        return ClusterDecision(
            False, n_nodes, n_nodes, identity, float("inf"), {}, 0.0, n_nodes, (), (), roles
        )
    baseline = _fit_b2(fit, selection, seed=int(seed), max_steps=int(max_steps))
    try:
        laplacian, _ = synchronizing_laplacian(
            incidence, np.asarray(baseline.coupling), equilibrium
        )
        eigenvalues, eigenvectors = generalized_laplacian_spectrum(laplacian, mass)
    except ValueError:
        return ClusterDecision(
            False,
            n_nodes,
            n_nodes,
            identity,
            float("inf"),
            {"coupling_operator_invalid": 1.0},
            0.0,
            n_nodes,
            (),
            (),
            roles,
        )
    active_modes = _active_spatial_modes(fit, mass, max_steps=int(max_steps))
    point_components: dict[int, dict[str, float]] = {}
    point_scores: dict[int, float] = {}
    for k in range(2, n_nodes):
        membership = eigenvectors[:, :k]
        projector = membership @ (membership.T * mass[None, :])
        gap = max(0.0, float(eigenvalues[k - 1])) / max(
            max(0.0, float(eigenvalues[k])), np.finfo(float).eps
        )
        components = _soft_components(
            fit,
            mass,
            projector,
            active_modes,
            gap,
            max_steps=int(max_steps),
            delays=int(delays),
        )
        point_components[k] = components
        point_scores[k] = _soft_score(components, thresholds)

    groups = sorted({_group(sample) for sample in fit})
    grouped = {group: [sample for sample in fit if _group(sample) == group] for group in groups}
    rng = np.random.default_rng(int(seed))
    draws: dict[int, list[float]] = {k: [] for k in point_scores}
    if bootstrap_draws > 0 and groups:
        for _ in range(int(bootstrap_draws)):
            chosen = rng.integers(0, len(groups), size=len(groups))
            boot = [sample for index in chosen for sample in grouped[groups[int(index)]]]
            boot_modes = _active_spatial_modes(boot, mass, max_steps=int(max_steps))
            for k in point_scores:
                membership = eigenvectors[:, :k]
                projector = membership @ (membership.T * mass[None, :])
                components = _soft_components(
                    boot,
                    mass,
                    projector,
                    boot_modes,
                    point_components[k]["spectral_gap_ratio"],
                    max_steps=int(max_steps),
                    delays=int(delays),
                )
                draws[k].append(_soft_score(components, thresholds))
    candidates = []
    passing_ranks = []
    for k in sorted(point_scores):
        values = np.asarray(draws[k] or [point_scores[k]], dtype=float)
        upper = float(np.quantile(values, thresholds.confidence))
        support = float(np.mean(values <= 1.0))
        passed = bool(upper <= 1.0 and support >= thresholds.bootstrap_support)
        candidates.append(
            {
                "n_clusters": int(k),
                "point_score": float(point_scores[k]),
                "bootstrap_upper": upper,
                "bootstrap_support": support,
                "soft_components": dict(point_components[k]),
                "soft_pass": passed,
            }
        )
        if passed:
            passing_ranks.append(k)

    for k in passing_ranks:
        partition = _hard_partition(fit, mass, eigenvectors, k, max_steps=int(max_steps))
        components = _hard_components(
            fit,
            selection,
            mass,
            laplacian,
            partition,
            active_modes,
            max_steps=int(max_steps),
            seed=int(seed),
        )
        score = _hard_score(components, thresholds)
        row = next(item for item in candidates if item["n_clusters"] == k)
        row["hard_components"] = components
        row["hard_score"] = score
        row["hard_pass"] = bool(score <= 1.0)
        row["partition"] = [list(group) for group in partition]
        if score <= 1.0:
            return ClusterDecision(
                True,
                n_nodes,
                k,
                partition,
                score,
                components,
                float(row["bootstrap_support"]),
                k,
                tuple(candidates),
                tuple(float(value) for value in eigenvalues),
                roles,
            )

    best = min(candidates, key=lambda row: row["bootstrap_upper"], default=None)
    return ClusterDecision(
        False,
        n_nodes,
        n_nodes,
        identity,
        float(best["bootstrap_upper"]) if best else float("inf"),
        dict(best["soft_components"]) if best else {},
        float(best["bootstrap_support"]) if best else 0.0,
        n_nodes,
        tuple(candidates),
        tuple(float(value) for value in eigenvalues),
        roles,
    )
