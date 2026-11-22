from __future__ import annotations

import csv
import hashlib
import html
import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import coherence

from kron_reduce.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation import (
    dominant_frequency,
    exact_dmd_modes,
    match_dmd_modes,
    spectral_overlap,
    spectral_signature,
)
from kron_reduce.models import GraphModalDiscoveryM0Model, M0Model, RunContext
from kron_reduce.models.transient_emf_m0 import TransientEmfM0Model
from kron_reduce.physics import (
    CLUSTERED_7_PARTITION,
    aggregate_state,
    cluster_aggregation_matrices,
    coi_project,
    coi_project_power,
    complete_incidence,
    project_cluster_consensus,
    reconstruct_cluster_state,
)
from project_harness.archive_storage import archive_path

HIGHORDER_SOURCE = "ieee39_highorder_dataset"
DENSE_SOURCE = "ieee39_near_critical_unstable_v1_dense"
LEGACY_7_PARTITION = ((0, 7), (1, 2), (3,), (4,), (5, 6), (8,), (9,))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_gmd_checkpoint(root: Path, relative: str) -> Path:
    checkpoint = root / relative
    diagnostics = root / "results/diagnostics"
    # Resolve archived artifacts without changing hashed historical configurations.
    if not checkpoint.is_file() and checkpoint.is_relative_to(diagnostics):
        archived_relative = (
            "results/backup/m0_gmd_horizon_balanced_backbone/"
            + checkpoint.relative_to(diagnostics).as_posix()
        )
        archived = archive_path(root, archived_relative)
        if archived.is_file():
            return archived
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in materialized for field in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    def coerce(value: str) -> Any:
        if value == "":
            return None
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return float(value)
        except ValueError:
            return value

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            {key: coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)
        ]


def _source(sample: dict[str, Any]) -> str:
    return str(sample["training_context"]["metadata"]["source_dataset"])


def _uniform_path(
    time_grid: np.ndarray,
    values: np.ndarray,
    *,
    maximum_time: float,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    time_grid = np.asarray(time_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    stop = min(float(time_grid[-1]), float(maximum_time))
    uniform_time = np.arange(0.0, stop + 0.5 * dt, dt)
    uniform = np.column_stack(
        [
            np.interp(uniform_time, time_grid, values[:, channel])
            for channel in range(values.shape[1])
        ]
    )
    return uniform_time, uniform


def _weighted_energy(values: np.ndarray, mass: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mass = np.asarray(mass, dtype=float)
    return float(np.sum(values * values * mass[None, :]))


def _frequency_quantile(frequency: np.ndarray, power: np.ndarray, quantile: float) -> float:
    if not len(frequency) or float(np.sum(power)) <= 0.0:
        return float("nan")
    normalized = power / np.sum(power)
    return float(
        frequency[min(int(np.searchsorted(np.cumsum(normalized), quantile)), len(frequency) - 1)]
    )


def _high_frequency_share(
    frequency: np.ndarray, power: np.ndarray, cutoff_hz: float = 1.0
) -> float:
    total = float(np.sum(power))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(power[frequency >= cutoff_hz]) / total)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float) - float(np.mean(left))
    right = np.asarray(right, dtype=float) - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > np.finfo(float).eps else 0.0


def _band_coherence(left: np.ndarray, right: np.ndarray, dt: float = 0.01) -> float:
    nperseg = min(256, len(left))
    if nperseg < 16:
        return float("nan")
    frequency, value = coherence(left, right, fs=1.0 / dt, nperseg=nperseg)
    band = (frequency >= 0.1) & (frequency <= 2.0)
    return float(np.mean(value[band])) if np.any(band) else float("nan")


def _random_pair_sets(draws: int, seed: int) -> list[tuple[tuple[int, int], ...]]:
    rng = np.random.default_rng(seed)
    output: set[tuple[tuple[int, int], ...]] = set()
    maximum = 3150
    while len(output) < min(draws, maximum):
        permutation = rng.permutation(10)
        pairs = tuple(
            sorted(
                tuple(sorted((int(permutation[2 * j]), int(permutation[2 * j + 1]))))
                for j in range(3)
            )
        )
        output.add(pairs)
    return sorted(output)


def _design_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    column_scale = np.sqrt(np.mean(matrix * matrix, axis=0))
    active = column_scale > np.finfo(float).eps
    normalized = matrix[:, active] / column_scale[active]
    singular = np.linalg.svd(normalized, compute_uv=False)
    column_norm = np.linalg.norm(normalized, axis=0)
    gram = normalized.T @ normalized
    correlation = gram / np.maximum(
        column_norm[:, None] * column_norm[None, :], np.finfo(float).eps
    )
    np.fill_diagonal(correlation, 0.0)
    return {
        "n_rows": int(matrix.shape[0]),
        "n_parameters": int(matrix.shape[1]),
        "active_columns": int(np.sum(active)),
        "effective_rank": int(np.linalg.matrix_rank(normalized)),
        "condition_number": float(singular[0] / max(singular[-1], np.finfo(float).eps)),
        "stable_rank": float(
            np.sum(singular * singular) / max(singular[0] ** 2, np.finfo(float).eps)
        ),
        "mutual_coherence": float(np.max(np.abs(correlation))),
    }


def _fit_m0_from_design(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    context: RunContext,
) -> tuple[M0Model, dict[str, Any]]:
    matrix, vector, weak = M0Model.build_weak_system(
        train,
        windows=context.integral_windows,
        max_cases=context.max_cases,
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
    )
    design = _design_metrics(matrix)
    model = M0Model(max_internal_step=context.max_internal_step)
    model.validation_scales = M0Model._training_scales(train, len(train))
    candidates: list[dict[str, Any]] = []
    n_nodes = int(np.asarray(train[0]["model_inputs"]["static"]["mass"]).size)
    for ridge in context.ridge_values:
        theta, solver = M0Model._solve_nonnegative_ridge(matrix, vector, ridge)
        model.damping = theta[:n_nodes]
        model.coupling = theta[n_nodes:]
        validation = model.validation_summary(val, context)
        candidates.append(
            {
                "ridge": float(ridge),
                "theta": theta.copy(),
                "solver": solver,
                "validation": validation,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["validation"]["standardized_loss"],
            row["validation"]["raw_mean_rmse"],
            row["ridge"],
        ),
    )
    theta = np.asarray(selected["theta"])
    model.damping = theta[:n_nodes]
    model.coupling = theta[n_nodes:]
    model.selected_ridge = float(selected["ridge"])
    model.diagnostics = {
        **weak,
        **design,
        **selected["solver"],
        "selected_ridge": model.selected_ridge,
        "validation": selected["validation"],
        "epochs": 0,
        "optimizer": "single_bounded_integral_ridge",
    }
    return model, {
        **weak,
        **design,
        "selected_ridge": model.selected_ridge,
        "weak_residual_rmse": selected["solver"]["weak_residual_rmse"],
        "validation_mean_rmse": selected["validation"]["raw_mean_rmse"],
        "minimum_damping": float(np.min(model.damping)),
        "median_damping": float(np.median(model.damping)),
        "maximum_damping": float(np.max(model.damping)),
    }


def _trajectory_errors(
    truth_delta: np.ndarray,
    truth_omega: np.ndarray,
    predicted_delta: np.ndarray,
    predicted_omega: np.ndarray,
    incidence: np.ndarray,
    time_grid: np.ndarray,
) -> dict[str, float]:
    omega_error = predicted_omega - truth_omega
    edge_error = (predicted_delta - truth_delta) @ incidence
    truth_peak_index = int(np.argmax(np.max(np.abs(truth_omega), axis=1)))
    pred_peak_index = int(np.argmax(np.max(np.abs(predicted_omega), axis=1)))
    quarter = max(2, len(time_grid) // 4)
    truth_decay = float(
        np.sqrt(np.mean(truth_omega[-quarter:] ** 2))
        / max(np.sqrt(np.mean(truth_omega[:quarter] ** 2)), 1e-12)
    )
    pred_decay = float(
        np.sqrt(np.mean(predicted_omega[-quarter:] ** 2))
        / max(np.sqrt(np.mean(predicted_omega[:quarter] ** 2)), 1e-12)
    )
    return {
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.concatenate(((predicted_delta - truth_delta).ravel(), omega_error.ravel()))
                    ** 2
                )
            )
        ),
        "omega_rmse": float(np.sqrt(np.mean(omega_error**2))),
        "edge_angle_rmse": float(np.sqrt(np.mean(edge_error**2))),
        "truth_peak_abs_omega": float(np.max(np.abs(truth_omega))),
        "predicted_peak_abs_omega": float(np.max(np.abs(predicted_omega))),
        "peak_time_error_s": abs(float(time_grid[pred_peak_index] - time_grid[truth_peak_index])),
        "decay_envelope_absolute_error": abs(pred_decay - truth_decay),
    }


def _gmd_variant(model: GraphModalDiscoveryM0Model, variant: str) -> Any:
    if variant == "backbone":
        return model.backbone
    static = np.asarray(model.static_coefficients).copy()
    memory = np.asarray(model.memory_coefficients).copy()
    if variant == "static_only":
        memory[:] = 0.0
    elif variant == "memory_only":
        static[:] = 0.0
    elif variant == "linear_only":
        nonlinear = [
            index
            for index, name in enumerate(model.feature_names)
            if name not in {"delta_sin", "edge_speed"}
        ]
        static[nonlinear] = 0.0
        memory[:, nonlinear] = 0.0
    elif variant != "full":
        raise ValueError(f"unknown GMD variant: {variant}")
    return GraphModalDiscoveryM0Model(
        backbone=model.backbone,
        modal_basis=np.asarray(model.modal_basis).copy(),
        taus=np.asarray(model.taus).copy(),
        static_coefficients=static,
        memory_coefficients=memory,
        feature_names=model.feature_names,
        max_internal_step=model.max_internal_step,
    )


def _sum_clusters(values: np.ndarray, partition: tuple[tuple[int, ...], ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.stack([np.sum(values[..., list(group)], axis=-1) for group in partition], axis=-1)


def _aggregate_adapted_sample(
    sample: dict[str, Any], partition: tuple[tuple[int, ...], ...], name: str
) -> dict[str, Any]:
    """Create an alternate seven-node view without rereading canonical data."""
    static10 = sample["model_inputs"]["static"]
    target10 = sample["training_context"]["targets"]
    mass10 = np.asarray(static10["mass"], dtype=float)
    mass7 = np.asarray([np.sum(mass10[list(group)]) for group in partition])
    incidence7, edges7 = complete_incidence(len(partition))
    static7 = {
        "delta0": coi_project(aggregate_state(static10["delta0"], mass10, partition), mass7),
        "omega0": coi_project(aggregate_state(static10["omega0"], mass10, partition), mass7),
        "Pm_equilibrium": coi_project_power(
            _sum_clusters(static10["Pm_equilibrium"], partition), mass7
        ),
        "Pm_clear": coi_project_power(_sum_clusters(static10["Pm_clear"], partition), mass7),
        "equilibrium_delta": coi_project(
            aggregate_state(static10["equilibrium_delta"], mass10, partition), mass7
        ),
        "mass": mass7,
        "incidence": incidence7,
    }
    static7["Pm0"] = static7["Pm_clear"].copy()
    pm_observed = coi_project_power(
        _sum_clusters(sample["training_context"]["Pm_observed"], partition), mass7
    )
    exogenous_pm = coi_project_power(
        _sum_clusters(sample["model_inputs"]["exogenous"]["Pm"], partition), mass7
    )
    return {
        "event_id": sample["event_id"],
        "model_inputs": {
            "static": static7,
            "exogenous": {
                "t": np.asarray(sample["model_inputs"]["exogenous"]["t"]).copy(),
                "Pm": exogenous_pm,
            },
        },
        "training_context": {
            "targets": {
                "t": np.asarray(target10["t"]).copy(),
                "delta": coi_project(aggregate_state(target10["delta"], mass10, partition), mass7),
                "omega": coi_project(aggregate_state(target10["omega"], mass10, partition), mass7),
            },
            "Pm_observed": pm_observed,
            "Pm_equilibrium": static7["Pm_equilibrium"].copy(),
            "label": sample["training_context"]["label"],
            "metadata": dict(sample["training_context"]["metadata"]),
            "decision_index": sample["training_context"]["decision_index"],
            "claim_type": sample["training_context"]["claim_type"],
            "representation": name,
            "edges": edges7,
        },
    }


def _discover_train_partition(
    train_samples: list[dict[str, Any]], maximum_time: float
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, Any]]]:
    """Select three disjoint pairs by source-balanced train-only trajectory distance."""
    pair_values: dict[tuple[int, int], dict[str, list[float]]] = {
        (left, right): defaultdict(list) for left in range(10) for right in range(left + 1, 10)
    }
    for sample in train_samples:
        target = sample["training_context"]["targets"]
        _, delta = _uniform_path(
            np.asarray(target["t"]), np.asarray(target["delta"]), maximum_time=maximum_time
        )
        _, omega = _uniform_path(
            np.asarray(target["t"]), np.asarray(target["omega"]), maximum_time=maximum_time
        )
        delta = delta - delta[[0]]
        delta_scale = max(float(np.sqrt(np.mean(delta**2))), 1e-12)
        omega_scale = max(float(np.sqrt(np.mean(omega**2))), 1e-12)
        source = _source(sample)
        for pair in pair_values:
            left, right = pair
            distance = 0.5 * (
                float(np.sqrt(np.mean((delta[:, left] - delta[:, right]) ** 2))) / delta_scale
                + float(np.sqrt(np.mean((omega[:, left] - omega[:, right]) ** 2))) / omega_scale
            )
            pair_values[pair][source].append(distance)
    pair_score: dict[tuple[int, int], float] = {}
    score_rows: list[dict[str, Any]] = []
    for pair, by_source in pair_values.items():
        source_medians = {source: float(np.median(values)) for source, values in by_source.items()}
        pair_score[pair] = float(np.mean(list(source_medians.values())))
        score_rows.append(
            {
                "pair": f"{pair[0]}-{pair[1]}",
                "source_balanced_distance": pair_score[pair],
                **{f"{source}_median_distance": value for source, value in source_medians.items()},
            }
        )
    pairs = sorted(pair_score)
    candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    for first_index, first in enumerate(pairs):
        for second_index in range(first_index + 1, len(pairs)):
            second = pairs[second_index]
            if set(first) & set(second):
                continue
            for third in pairs[second_index + 1 :]:
                if set(first) & set(third) or set(second) & set(third):
                    continue
                selected = tuple(sorted((first, second, third)))
                candidates.append((sum(pair_score[pair] for pair in selected), selected))
    _, selected_pairs = min(candidates, key=lambda item: (item[0], item[1]))
    paired_nodes = {node for pair in selected_pairs for node in pair}
    partition = tuple(selected_pairs) + tuple(
        (node,) for node in range(10) if node not in paired_nodes
    )
    for row in score_rows:
        pair = tuple(int(value) for value in str(row["pair"]).split("-"))
        row["selected"] = pair in selected_pairs
    return partition, score_rows


def _projection_and_coherency_audit(
    samples: list[dict[str, Any]],
    *,
    role: str,
    partition_name: str,
    partition: tuple[tuple[int, ...], ...],
    random_pair_sets: list[tuple[tuple[int, int], ...]],
    maximum_time: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projection_rows: list[dict[str, Any]] = []
    coherency_rows: list[dict[str, Any]] = []
    for sample in samples:
        target = sample["training_context"]["targets"]
        _, omega = _uniform_path(
            np.asarray(target["t"]), np.asarray(target["omega"]), maximum_time=maximum_time
        )
        _, delta = _uniform_path(
            np.asarray(target["t"]), np.asarray(target["delta"]), maximum_time=maximum_time
        )
        mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=float)
        source = _source(sample)
        for signal_name, values in (
            ("omega_absolute", omega),
            ("omega_increment", omega - omega[[0]]),
            ("delta_increment", delta - delta[[0]]),
            ("delta_absolute", delta),
        ):
            low = project_cluster_consensus(values, mass, partition)
            high = values - low
            total_energy = _weighted_energy(values, mass)
            retained = _weighted_energy(low, mass) / max(total_energy, np.finfo(float).eps)
            low_frequency, low_power = spectral_signature(
                low, 0.01, channel_weights=mass, max_frequency_hz=5.0
            )
            high_frequency, high_power = spectral_signature(
                high, 0.01, channel_weights=mass, max_frequency_hz=5.0
            )
            projection_rows.append(
                {
                    "role": role,
                    "partition": partition_name,
                    "event_id": sample["event_id"],
                    "source_dataset": source,
                    "signal": signal_name,
                    "retained_mass_energy_fraction": retained,
                    "discarded_mass_energy_fraction": _weighted_energy(high, mass)
                    / max(total_energy, np.finfo(float).eps),
                    "low_component_median_frequency_hz": _frequency_quantile(
                        low_frequency, low_power, 0.5
                    ),
                    "high_component_median_frequency_hz": _frequency_quantile(
                        high_frequency, high_power, 0.5
                    ),
                    "low_component_above_1hz_share": _high_frequency_share(
                        low_frequency, low_power
                    ),
                    "high_component_above_1hz_share": _high_frequency_share(
                        high_frequency, high_power
                    ),
                }
            )

        omega_increment = omega - omega[[0]]
        correlation_matrix = np.eye(10)
        coherence_matrix = np.eye(10)
        relative_rms = np.zeros((10, 10))
        global_scale = max(float(np.sqrt(np.mean(omega_increment**2))), 1e-12)
        for left in range(10):
            for right in range(left + 1, 10):
                correlation_matrix[left, right] = correlation_matrix[right, left] = (
                    _safe_correlation(omega_increment[:, left], omega_increment[:, right])
                )
                coherence_matrix[left, right] = coherence_matrix[right, left] = _band_coherence(
                    omega_increment[:, left], omega_increment[:, right]
                )
                relative_rms[left, right] = relative_rms[right, left] = float(
                    np.sqrt(np.mean((omega_increment[:, left] - omega_increment[:, right]) ** 2))
                    / global_scale
                )
        fixed_pairs = tuple(tuple(group) for group in partition if len(group) == 2)
        if len(fixed_pairs) != 3:
            raise ValueError("seven-node comparator must contain exactly three pairs")
        fixed_correlation = float(
            np.mean([correlation_matrix[left, right] for left, right in fixed_pairs])
        )
        fixed_coherence = float(
            np.mean([coherence_matrix[left, right] for left, right in fixed_pairs])
        )
        fixed_relative_rms = float(
            np.mean([relative_rms[left, right] for left, right in fixed_pairs])
        )
        random_correlation = np.asarray(
            [
                np.mean([correlation_matrix[left, right] for left, right in pair_set])
                for pair_set in random_pair_sets
            ]
        )
        random_coherence = np.asarray(
            [
                np.mean([coherence_matrix[left, right] for left, right in pair_set])
                for pair_set in random_pair_sets
            ]
        )
        random_relative_rms = np.asarray(
            [
                np.mean([relative_rms[left, right] for left, right in pair_set])
                for pair_set in random_pair_sets
            ]
        )
        coherency_rows.append(
            {
                "role": role,
                "partition": partition_name,
                "event_id": sample["event_id"],
                "source_dataset": source,
                "fixed_pair_mean_correlation": fixed_correlation,
                "fixed_pair_mean_band_coherence": fixed_coherence,
                "fixed_pair_relative_rms": fixed_relative_rms,
                "correlation_random_partition_percentile": float(
                    np.mean(random_correlation <= fixed_correlation)
                ),
                "coherence_random_partition_percentile": float(
                    np.mean(random_coherence <= fixed_coherence)
                ),
                "relative_rms_random_partition_percentile": float(
                    np.mean(random_relative_rms >= fixed_relative_rms)
                ),
            }
        )
    return projection_rows, coherency_rows


def _fit_comparison_models(
    samples: dict[str, dict[str, list[dict[str, Any]]]],
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> tuple[dict[tuple[str, str, str], Any], list[dict[str, Any]]]:
    models: dict[tuple[str, str, str], Any] = {}
    design_rows: list[dict[str, Any]] = []
    for cohort in ("combined", "highorder", "dense"):
        for representation in samples:
            train_all = samples[representation]["train"]
            val_all = samples[representation]["val"]
            source_filter = {
                "combined": None,
                "highorder": HIGHORDER_SOURCE,
                "dense": DENSE_SOURCE,
            }[cohort]
            train = (
                train_all
                if source_filter is None
                else [row for row in train_all if _source(row) == source_filter]
            )
            val = (
                val_all
                if source_filter is None
                else [row for row in val_all if _source(row) == source_filter]
            )
            context = RunContext(
                seed=int(config["seed"]),
                epochs=0,
                horizon_steps=int(config["fit_steps"]),
                max_cases=len(train),
                learning_rate=0.0,
                ridge_values=tuple(float(value) for value in config["ridge_values"]),
                integral_windows=tuple(int(value) for value in config["integral_windows"]),
                max_internal_step=float(config["max_internal_step"]),
                fit_max_steps=int(config["fit_steps"]),
            )
            m0, diagnostics = _fit_m0_from_design(train, val, context)
            models[(cohort, representation, "M0")] = m0
            model_path = checkpoint_dir / f"{cohort}-{representation}-M0.npz"
            m0.save(model_path)
            design_rows.append(
                {"cohort": cohort, "representation": representation, "method": "M0", **diagnostics}
            )
            if cohort != "highorder":
                continue
            third_order = TransientEmfM0Model(
                anchor=m0,
                alpha_candidates=tuple(
                    float(value) for value in config["third_order_alpha_candidates"]
                ),
                tau_candidates=tuple(
                    float(value) for value in config["third_order_tau_candidates"]
                ),
                max_internal_step=float(config["max_internal_step"]),
            )
            third_fit = third_order.fit(train, val, context)
            models[(cohort, representation, "M0-3E")] = third_order
            third_order.save(checkpoint_dir / f"{cohort}-{representation}-M0-3E.npz")
            design_rows.append(
                {
                    "cohort": cohort,
                    "representation": representation,
                    "method": "M0-3E",
                    "n_parameters": diagnostics["n_parameters"] + 2,
                    "selected_alpha": third_order.alpha,
                    "selected_tau": third_order.tau,
                    "validation_mean_rmse": third_fit.details["selected_validation"][
                        "raw_mean_rmse"
                    ],
                    "validation_standardized_loss": third_fit.best_validation_loss,
                    "minimum_damping": float(np.min(third_order.damping)),
                    "median_damping": float(np.median(third_order.damping)),
                    "maximum_damping": float(np.max(third_order.damping)),
                }
            )
    return models, design_rows


def _parameter_consistency(models: dict[tuple[str, str, str], Any]) -> list[dict[str, Any]]:
    """Compare independently fitted effective M0 parameters across source families."""
    rows: list[dict[str, Any]] = []
    representations = sorted({key[1] for key in models if key[2] == "M0"})
    for representation in representations:
        highorder = models[("highorder", representation, "M0")]
        dense = models[("dense", representation, "M0")]
        for parameter in ("damping", "coupling"):
            left = np.asarray(getattr(highorder, parameter), dtype=float)
            right = np.asarray(getattr(dense, parameter), dtype=float)
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            cosine = (
                float(left @ right / denominator)
                if denominator > np.finfo(float).eps
                else float("nan")
            )
            reference = max(
                0.5 * float(np.linalg.norm(left) + np.linalg.norm(right)), np.finfo(float).eps
            )
            rows.append(
                {
                    "representation": representation,
                    "parameter": parameter,
                    "highorder_dense_cosine": cosine,
                    "symmetric_relative_l2_difference": float(
                        np.linalg.norm(left - right) / reference
                    ),
                    "highorder_median": float(np.median(left)),
                    "dense_median": float(np.median(right)),
                    "highorder_minimum": float(np.min(left)),
                    "dense_minimum": float(np.min(right)),
                    "highorder_maximum": float(np.max(left)),
                    "dense_maximum": float(np.max(right)),
                }
            )
    return rows


def _comparison_rollouts(
    models: dict[tuple[str, str, str], Any],
    samples: dict[str, dict[str, list[dict[str, Any]]]],
    partitions: dict[str, tuple[tuple[int, ...], ...]],
    horizons_s: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val10 = {sample["event_id"]: sample for sample in samples["generator_10"]["val"]}
    val7 = {
        name: {sample["event_id"]: sample for sample in samples[name]["val"]} for name in partitions
    }
    for (cohort, representation, method), model in models.items():
        representation_samples = samples[representation]["val"]
        if cohort == "highorder":
            representation_samples = [
                sample for sample in representation_samples if _source(sample) == HIGHORDER_SOURCE
            ]
        elif cohort == "dense":
            representation_samples = [
                sample for sample in representation_samples if _source(sample) == DENSE_SOURCE
            ]
        for sample in representation_samples:
            target = sample["training_context"]["targets"]
            maximum_steps = min(len(target["t"]), round(max(horizons_s) / 0.01) + 1)
            static = sample["model_inputs"]["static"]
            exogenous = sample["model_inputs"]["exogenous"]
            try:
                predicted_delta, predicted_omega = model.rollout(
                    static,
                    np.asarray(target["t"][:maximum_steps]),
                    {"Pm": np.asarray(exogenous["Pm"][:maximum_steps]), "interpolation": "linear"},
                )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
                rows.append(
                    {
                        "cohort": cohort,
                        "representation": representation,
                        "method": method,
                        "event_id": sample["event_id"],
                        "source_dataset": _source(sample),
                        "status": "failed",
                        "error": repr(error),
                    }
                )
                continue
            for horizon_s in horizons_s:
                steps = round(horizon_s / 0.01) + 1
                if maximum_steps < steps:
                    continue
                native = _trajectory_errors(
                    np.asarray(target["delta"][:steps]),
                    np.asarray(target["omega"][:steps]),
                    predicted_delta[:steps],
                    predicted_omega[:steps],
                    np.asarray(static["incidence"]),
                    np.asarray(target["t"][:steps]),
                )
                base_row = {
                    "cohort": cohort,
                    "representation": representation,
                    "method": method,
                    "event_id": sample["event_id"],
                    "source_dataset": _source(sample),
                    "horizon_s": horizon_s,
                    "status": "completed",
                    **{f"native_{key}": value for key, value in native.items()},
                }
                if representation == "generator_10":
                    mass10 = np.asarray(static["mass"])
                    for partition_name, partition in partitions.items():
                        truth7_sample = val7[partition_name][sample["event_id"]]
                        truth7 = truth7_sample["training_context"]["targets"]
                        delta7 = aggregate_state(predicted_delta[:steps], mass10, partition)
                        omega7 = aggregate_state(predicted_omega[:steps], mass10, partition)
                        interarea = _trajectory_errors(
                            np.asarray(truth7["delta"][:steps]),
                            np.asarray(truth7["omega"][:steps]),
                            delta7,
                            omega7,
                            np.asarray(truth7_sample["model_inputs"]["static"]["incidence"]),
                            np.asarray(truth7["t"][:steps]),
                        )
                        rows.append(
                            {
                                **base_row,
                                "evaluation_partition": partition_name,
                                **{f"interarea7_{key}": value for key, value in interarea.items()},
                            }
                        )
                else:
                    truth10 = val10[sample["event_id"]]["training_context"]["targets"]
                    static10 = val10[sample["event_id"]]["model_inputs"]["static"]
                    partition = partitions[representation]
                    delta10 = reconstruct_cluster_state(
                        predicted_delta[:steps], static10["mass"], partition
                    )
                    omega10 = reconstruct_cluster_state(
                        predicted_omega[:steps], static10["mass"], partition
                    )
                    reconstructed = _trajectory_errors(
                        np.asarray(truth10["delta"][:steps]),
                        np.asarray(truth10["omega"][:steps]),
                        delta10,
                        omega10,
                        np.asarray(static10["incidence"]),
                        np.asarray(truth10["t"][:steps]),
                    )
                    rows.append(
                        {
                            **base_row,
                            "evaluation_partition": representation,
                            **{
                                f"reconstructed10_{key}": value
                                for key, value in reconstructed.items()
                            },
                        }
                    )
    return rows


def _gmd_rollout_audit(
    model: GraphModalDiscoveryM0Model,
    validation_samples: list[dict[str, Any]],
    horizons_s: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory_rows: list[dict[str, Any]] = []
    modal_rows: list[dict[str, Any]] = []
    variants = {
        name: _gmd_variant(model, name)
        for name in ("backbone", "static_only", "memory_only", "linear_only", "full")
    }
    for sample in validation_samples:
        target = sample["training_context"]["targets"]
        maximum_steps = min(len(target["t"]), round(max(horizons_s) / 0.01) + 1)
        static = sample["model_inputs"]["static"]
        exogenous = sample["model_inputs"]["exogenous"]
        for variant_name, variant in variants.items():
            try:
                predicted_delta, predicted_omega = variant.rollout(
                    static,
                    np.asarray(target["t"][:maximum_steps]),
                    {"Pm": np.asarray(exogenous["Pm"][:maximum_steps]), "interpolation": "linear"},
                )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
                trajectory_rows.append(
                    {
                        "event_id": sample["event_id"],
                        "source_dataset": _source(sample),
                        "variant": variant_name,
                        "status": "failed",
                        "error": repr(error),
                    }
                )
                continue
            for horizon_s in horizons_s:
                steps = round(horizon_s / 0.01) + 1
                if maximum_steps < steps:
                    continue
                _, truth_omega = _uniform_path(
                    np.asarray(target["t"][:steps]),
                    np.asarray(target["omega"][:steps]),
                    maximum_time=horizon_s,
                )
                _, predicted_omega_uniform = _uniform_path(
                    np.asarray(target["t"][:steps]), predicted_omega[:steps], maximum_time=horizon_s
                )
                frequency, truth_power = spectral_signature(
                    truth_omega,
                    0.01,
                    channel_weights=np.asarray(static["mass"]),
                    max_frequency_hz=5.0,
                )
                _, predicted_power = spectral_signature(
                    predicted_omega_uniform,
                    0.01,
                    channel_weights=np.asarray(static["mass"]),
                    max_frequency_hz=5.0,
                )
                errors = _trajectory_errors(
                    np.asarray(target["delta"][:steps]),
                    np.asarray(target["omega"][:steps]),
                    predicted_delta[:steps],
                    predicted_omega[:steps],
                    np.asarray(static["incidence"]),
                    np.asarray(target["t"][:steps]),
                )
                trajectory_rows.append(
                    {
                        "event_id": sample["event_id"],
                        "source_dataset": _source(sample),
                        "variant": variant_name,
                        "horizon_s": horizon_s,
                        "status": "completed",
                        "spectral_overlap": spectral_overlap(truth_power, predicted_power),
                        "truth_dominant_frequency_hz": dominant_frequency(frequency, truth_power),
                        "predicted_dominant_frequency_hz": dominant_frequency(
                            frequency, predicted_power
                        ),
                        "dominant_frequency_error_hz": abs(
                            dominant_frequency(frequency, predicted_power)
                            - dominant_frequency(frequency, truth_power)
                        ),
                        **errors,
                    }
                )
            if maximum_steps < 501:
                continue
            _, truth_omega = _uniform_path(
                np.asarray(target["t"][:maximum_steps]),
                np.asarray(target["omega"][:maximum_steps]),
                maximum_time=max(horizons_s),
            )
            _, predicted_omega_uniform = _uniform_path(
                np.asarray(target["t"][:maximum_steps]),
                predicted_omega[:maximum_steps],
                maximum_time=max(horizons_s),
            )
            truth_modes = exact_dmd_modes(truth_omega, 0.01, rank=6, max_frequency_hz=3.0)
            predicted_modes = exact_dmd_modes(
                predicted_omega_uniform, 0.01, rank=6, max_frequency_hz=3.0
            )
            matches = match_dmd_modes(truth_modes, predicted_modes)
            if not matches:
                modal_rows.append(
                    {
                        "event_id": sample["event_id"],
                        "source_dataset": _source(sample),
                        "variant": variant_name,
                        "status": "no_matched_mode",
                    }
                )
            for rank, match in enumerate(
                sorted(matches, key=lambda row: row["reference_strength"], reverse=True), start=1
            ):
                modal_rows.append(
                    {
                        "event_id": sample["event_id"],
                        "source_dataset": _source(sample),
                        "variant": variant_name,
                        "status": "completed",
                        "matched_mode_rank": rank,
                        **match,
                    }
                )
    return trajectory_rows, modal_rows


def _closure_force_components(
    model: GraphModalDiscoveryM0Model,
    delta: np.ndarray,
    omega: np.ndarray,
    time_grid: np.ndarray,
    static: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    basis = np.asarray(model.modal_basis)
    static_coefficients = np.asarray(model.static_coefficients)
    memory_coefficients = np.asarray(model.memory_coefficients)
    taus = np.asarray(model.taus)
    mass = np.asarray(static["mass"])
    drivers = np.stack(
        [
            model._modal_drivers(delta[index], omega[index], static)
            for index in range(len(time_grid))
        ]
    )
    z = model._first_order_hold_memory(drivers, time_grid, taus)

    def force_for(
        feature_indices: list[int], *, include_static: bool, include_memory: bool
    ) -> np.ndarray:
        amplitude = np.zeros((len(time_grid), basis.shape[1]))
        if include_static:
            amplitude += np.sum(
                static_coefficients[None, feature_indices, :] * drivers[:, feature_indices, :],
                axis=1,
            )
        if include_memory and len(taus):
            amplitude += np.sum(
                memory_coefficients[None, :, feature_indices, :]
                * (z[:, :, feature_indices, :] - drivers[:, None, feature_indices, :]),
                axis=(1, 2),
            )
        return np.stack([coi_project_power(-basis @ row, mass) for row in amplitude])

    all_features = list(range(len(model.feature_names)))
    components = {
        "full": force_for(all_features, include_static=True, include_memory=True),
        "static": force_for(all_features, include_static=True, include_memory=False),
        "memory": force_for(all_features, include_static=False, include_memory=True),
    }
    for index, name in enumerate(model.feature_names):
        components[f"feature:{name}"] = force_for([index], include_static=True, include_memory=True)
    return components


def _residual_structure_audit(
    model: GraphModalDiscoveryM0Model,
    validation_samples: list[dict[str, Any]],
    maximum_time: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    maximum_steps = round(maximum_time / 0.01) + 1
    for sample in validation_samples:
        target = sample["training_context"]["targets"]
        steps = min(len(target["t"]), maximum_steps)
        if steps < 101:
            continue
        prepared = model._prepare_raw_paths([sample], horizon_steps=steps, max_fit_steps=steps)[0]
        time_grid = np.asarray(prepared["t"])
        delta = np.asarray(prepared["delta"])
        omega = np.asarray(prepared["omega"])
        true_residual = np.asarray(prepared["force_residual"])
        components = _closure_force_components(model, delta, omega, time_grid, prepared["static"])
        omega_mid = 0.5 * (omega[:-1] + omega[1:])
        dt = np.diff(time_grid)
        true_power = np.sum(omega_mid * true_residual, axis=1)
        true_work = float(np.sum(true_power * dt))
        for component_name, force_path in components.items():
            predicted = 0.5 * (force_path[:-1] + force_path[1:])
            denominator = float(np.linalg.norm(true_residual) * np.linalg.norm(predicted))
            predicted_power = np.sum(omega_mid * predicted, axis=1)
            power_correlation = _safe_correlation(true_power, predicted_power)
            predicted_work = float(np.sum(predicted_power * dt))
            rows.append(
                {
                    "event_id": sample["event_id"],
                    "source_dataset": _source(sample),
                    "component": component_name,
                    "force_cosine_alignment": float(np.sum(true_residual * predicted) / denominator)
                    if denominator > 0
                    else 0.0,
                    "residual_explained_energy": float(
                        1.0
                        - np.sum((true_residual - predicted) ** 2)
                        / max(np.sum(true_residual**2), np.finfo(float).eps)
                    ),
                    "predicted_to_true_force_rms_ratio": float(
                        np.sqrt(np.mean(predicted**2))
                        / max(np.sqrt(np.mean(true_residual**2)), np.finfo(float).eps)
                    ),
                    "residual_power_correlation": power_correlation,
                    "true_cumulative_residual_work": true_work,
                    "predicted_cumulative_residual_work": predicted_work,
                    "normalized_cumulative_work_error": abs(predicted_work - true_work)
                    / max(
                        abs(true_work),
                        np.sqrt(np.mean(true_power**2)) * float(time_grid[-1]),
                        1e-12,
                    ),
                    "true_negative_power_fraction": float(np.mean(true_power < 0.0)),
                    "predicted_negative_power_fraction": float(np.mean(predicted_power < 0.0)),
                }
            )
    return rows


def _group_mean(rows: list[dict[str, Any]], field: str, **filters: Any) -> float:
    values = [
        float(row[field])
        for row in rows
        if all(row.get(key) == value for key, value in filters.items())
        and row.get(field) not in (None, "")
        and np.isfinite(float(row[field]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _group_median(rows: list[dict[str, Any]], field: str, **filters: Any) -> float:
    values = [
        float(row[field])
        for row in rows
        if all(row.get(key) == value for key, value in filters.items())
        and row.get(field) not in (None, "")
        and np.isfinite(float(row[field]))
    ]
    return float(np.median(values)) if values else float("nan")


def _claim_assessment(
    config: dict[str, Any],
    partitions: dict[str, tuple[tuple[int, ...], ...]],
    projection_rows: list[dict[str, Any]],
    coherency_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    gmd_rows: list[dict[str, Any]],
    modal_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    spectral_overlap_value = _group_median(
        gmd_rows, "spectral_overlap", variant="full", horizon_s=5.0, status="completed"
    )
    dominant_error = _group_median(
        gmd_rows, "dominant_frequency_error_hz", variant="full", horizon_s=5.0, status="completed"
    )
    modal_mac = _group_median(
        [row for row in modal_rows if row.get("matched_mode_rank") == 1],
        "mac",
        variant="full",
        status="completed",
    )
    modal_damping_error = _group_median(
        [row for row in modal_rows if row.get("matched_mode_rank") == 1],
        "absolute_damping_ratio_error",
        variant="full",
        status="completed",
    )
    residual_alignment = _group_median(residual_rows, "force_cosine_alignment", component="full")
    residual_explained = _group_median(residual_rows, "residual_explained_energy", component="full")
    gmd_failures = sum(
        row.get("variant") == "full" and row.get("status") == "failed" for row in gmd_rows
    )
    oscillation_pass = bool(
        spectral_overlap_value >= float(config["minimum_spectral_overlap"])
        and dominant_error <= float(config["maximum_dominant_frequency_error_hz"])
        and (
            not np.isfinite(modal_mac)
            or modal_mac >= float(config["minimum_modal_assurance_criterion"])
        )
        and gmd_failures == 0
    )
    residual_pass = bool(residual_alignment > 0.0 and residual_explained > 0.0)

    design_index = {
        (row["cohort"], row["representation"], row["method"]): row for row in design_rows
    }

    partition_assessments: dict[str, dict[str, Any]] = {}
    for name, partition in partitions.items():
        energy_omega_absolute = _group_median(
            projection_rows,
            "retained_mass_energy_fraction",
            role="val",
            partition=name,
            signal="omega_absolute",
        )
        energy_omega_increment = _group_median(
            projection_rows,
            "retained_mass_energy_fraction",
            role="val",
            partition=name,
            signal="omega_increment",
        )
        energy_delta = _group_median(
            projection_rows,
            "retained_mass_energy_fraction",
            role="val",
            partition=name,
            signal="delta_increment",
        )
        percentile = _group_median(
            coherency_rows,
            "correlation_random_partition_percentile",
            role="val",
            partition=name,
        )
        low_frequency = _group_median(
            projection_rows,
            "low_component_median_frequency_hz",
            role="val",
            partition=name,
            signal="omega_absolute",
        )
        high_frequency = _group_median(
            projection_rows,
            "high_component_median_frequency_hz",
            role="val",
            partition=name,
            signal="omega_absolute",
        )
        low_above_1hz = _group_median(
            projection_rows,
            "low_component_above_1hz_share",
            role="val",
            partition=name,
            signal="omega_absolute",
        )
        high_above_1hz = _group_median(
            projection_rows,
            "high_component_above_1hz_share",
            role="val",
            partition=name,
            signal="omega_absolute",
        )
        conditioning_ratios = {
            cohort: float(
                design_index[(cohort, name, "M0")]["condition_number"]
                / design_index[(cohort, "generator_10", "M0")]["condition_number"]
            )
            for cohort in ("combined", "highorder", "dense")
        }
        native_m03e = _group_mean(
            comparison_rows,
            "native_rmse",
            cohort="highorder",
            representation=name,
            evaluation_partition=name,
            method="M0-3E",
            horizon_s=1.0,
            status="completed",
        )
        projected_m03e = _group_mean(
            comparison_rows,
            "interarea7_rmse",
            cohort="highorder",
            representation="generator_10",
            evaluation_partition=name,
            method="M0-3E",
            horizon_s=1.0,
            status="completed",
        )
        selected_alpha = float(design_index[("highorder", name, "M0-3E")]["selected_alpha"])
        gates = {
            "projection_energy": bool(
                energy_omega_absolute >= float(config["minimum_dynamic_energy_retention"])
                and energy_delta >= float(config["minimum_dynamic_energy_retention"])
            ),
            "fixed_partition_coherency": bool(
                percentile >= float(config["minimum_random_partition_percentile"])
            ),
            "empirical_temporal_scale_separation": bool(
                high_frequency > low_frequency and high_above_1hz > low_above_1hz
            ),
            "conditioning": bool(all(value < 1.0 for value in conditioning_ratios.values())),
            "higher_order_predictive": bool(
                np.isfinite(native_m03e)
                and np.isfinite(projected_m03e)
                and native_m03e <= projected_m03e
            ),
            "active_third_order_state": bool(selected_alpha > 0.0),
        }
        passed = sum(gates.values())
        status = (
            "SUPPORTED" if passed == len(gates) else ("CONDITIONAL" if passed >= 4 else "REJECTED")
        )
        if not gates["active_third_order_state"] and status == "SUPPORTED":
            status = "CONDITIONAL"
        partition_assessments[name] = {
            "status": status,
            "partition": [list(group) for group in partition],
            "gates": gates,
            "passed_gate_count": passed,
            "validation_omega_absolute_energy_retention": energy_omega_absolute,
            "validation_omega_increment_energy_retention": energy_omega_increment,
            "validation_delta_increment_energy_retention": energy_delta,
            "validation_correlation_random_partition_percentile": percentile,
            "low_component_median_frequency_hz": low_frequency,
            "high_component_median_frequency_hz": high_frequency,
            "low_component_above_1hz_share": low_above_1hz,
            "high_component_above_1hz_share": high_above_1hz,
            "conditioning_ratio_to_10node": conditioning_ratios,
            "highorder_M0_3E_native_mean_rmse": native_m03e,
            "highorder_M0_3E_10node_projected_mean_rmse": projected_m03e,
            "highorder_native_to_projected_rmse_ratio": float(native_m03e / projected_m03e)
            if np.isfinite(native_m03e) and projected_m03e > 0.0
            else float("nan"),
            "selected_third_order_alpha": selected_alpha,
            "selected_third_order_tau": float(
                design_index[("highorder", name, "M0-3E")]["selected_tau"]
            ),
        }

    ranking = sorted(
        partition_assessments,
        key=lambda name: (
            -partition_assessments[name]["passed_gate_count"],
            partition_assessments[name]["highorder_native_to_projected_rmse_ratio"],
            -partition_assessments[name]["validation_correlation_random_partition_percentile"],
            name,
        ),
    )
    best_partition = ranking[0]
    overall_cluster_status = (
        "SUPPORTED"
        if any(row["status"] == "SUPPORTED" for row in partition_assessments.values())
        else (
            "CONDITIONAL"
            if any(row["status"] == "CONDITIONAL" for row in partition_assessments.values())
            else "REJECTED"
        )
    )

    source_specific_highorder = _group_mean(
        comparison_rows,
        "native_rmse",
        cohort="highorder",
        representation="generator_10",
        source_dataset=HIGHORDER_SOURCE,
        method="M0",
        horizon_s=1.0,
        status="completed",
    )
    combined_on_highorder = _group_mean(
        comparison_rows,
        "native_rmse",
        cohort="combined",
        representation="generator_10",
        source_dataset=HIGHORDER_SOURCE,
        method="M0",
        horizon_s=1.0,
        status="completed",
    )
    source_specific_dense = _group_mean(
        comparison_rows,
        "native_rmse",
        cohort="dense",
        representation="generator_10",
        source_dataset=DENSE_SOURCE,
        method="M0",
        horizon_s=1.0,
        status="completed",
    )
    combined_on_dense = _group_mean(
        comparison_rows,
        "native_rmse",
        cohort="combined",
        representation="generator_10",
        source_dataset=DENSE_SOURCE,
        method="M0",
        horizon_s=1.0,
        status="completed",
    )
    generator_parameter_rows = [
        row for row in parameter_rows if row["representation"] == "generator_10"
    ]
    data_regime_separation = bool(
        source_specific_highorder < combined_on_highorder
        or source_specific_dense < combined_on_dense
        or any(
            float(row["symmetric_relative_l2_difference"]) > 0.25
            for row in generator_parameter_rows
        )
    )

    physical_status = (
        "SUPPORTED"
        if oscillation_pass and residual_pass
        else ("CONDITIONAL" if oscillation_pass or residual_pass else "REJECTED")
    )
    return {
        "current_equation_has_effective_physical_meaning": {
            "status": physical_status,
            "scope": "effective reduced response under observed Pm; not machine-parameter or topology recovery",
            "oscillation_preservation_gate": oscillation_pass,
            "residual_structure_gate": residual_pass,
            "median_5s_spectral_overlap": spectral_overlap_value,
            "median_5s_dominant_frequency_error_hz": dominant_error,
            "median_primary_mode_MAC": modal_mac,
            "median_primary_mode_absolute_damping_ratio_error": modal_damping_error,
            "median_force_cosine_alignment": residual_alignment,
            "median_residual_explained_energy": residual_explained,
            "full_rollout_failures": gmd_failures,
        },
        "seven_node_frontend_supported": {
            "status": overall_cluster_status,
            "scope": "mass-orthogonal cluster-consensus projection with train-only partition discovery and frozen validation",
            "best_conditional_candidate": best_partition,
            "candidate_is_not_test_selected_or_promoted": True,
            "partition_assessments": partition_assessments,
        },
        "partition_lineage": {
            "status": "CORRECTED",
            "legacy_and_current_partitions_are_different": partitions["clustered_7_legacy"]
            != partitions["clustered_7_current"],
            "reason": "the historical higher-order experiment and current canonical adapter use different seven-node groupings",
        },
        "mixed_source_parameter_consistency": {
            "status": "CONDITIONAL" if data_regime_separation else "SUPPORTED",
            "source_regime_separation_detected": data_regime_separation,
            "highorder_specific_M0_mean_rmse": source_specific_highorder,
            "combined_M0_on_highorder_mean_rmse": combined_on_highorder,
            "dense_specific_M0_mean_rmse": source_specific_dense,
            "combined_M0_on_dense_mean_rmse": combined_on_dense,
            "parameter_comparison": generator_parameter_rows,
            "interpretation": "effective-parameter inconsistency is evidence of distinct response regimes, not proof that simulator physical parameters changed",
        },
        "strict_graph_low_pass_claim": {
            "status": "REJECTED",
            "reason": "no graph-shift operator and transfer-function attenuation ratio were specified; temporal scale separation is not a strict graph-frequency certificate",
        },
        "clustering_algorithm_novelty": {
            "status": "REJECTED",
            "reason": "slow coherency and inertial aggregation are established; novelty must attach to the conditioning-aware weak-form discovery pipeline and its audits",
        },
        "machine_parameter_or_physical_topology_recovery": {
            "status": "REJECTED",
            "reason": "the reduced complete-graph coefficients and residual closure are effective parameters; the current data do not identify the original network or machine D uniquely",
        },
    }


def _make_figures(
    result_dir: Path,
    gmd_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    coherency_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure_dir = result_dir / "figures"
    figure_dir.mkdir()
    colors = {
        "backbone": "#777777",
        "static_only": "#56B4E9",
        "memory_only": "#E69F00",
        "linear_only": "#009E73",
        "full": "#D55E00",
    }
    partition_colors = {
        "clustered_7_current": "#0072B2",
        "clustered_7_legacy": "#D55E00",
        "clustered_7_train_discovered": "#009E73",
    }
    partition_labels = {
        "clustered_7_current": "current",
        "clustered_7_legacy": "legacy",
        "clustered_7_train_discovered": "train-discovered",
    }
    variants = list(colors)
    horizons = (1.0, 5.0, 10.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    ax = axes[0, 0]
    for variant in variants:
        means = [
            _group_mean(gmd_rows, "rmse", variant=variant, horizon_s=horizon, status="completed")
            for horizon in horizons
        ]
        ax.plot(
            horizons,
            means,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=variant.replace("_", " "),
            color=colors[variant],
        )
    ax.set(
        xlabel="Reintegration horizon (s)",
        ylabel="Mean trajectory RMSE",
        title="Closed-loop accuracy",
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(-0.16, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[0, 1]
    x = np.arange(len(variants))
    overlap = [
        _group_median(
            gmd_rows, "spectral_overlap", variant=variant, horizon_s=5.0, status="completed"
        )
        for variant in variants
    ]
    ax.bar(x, overlap, color=[colors[value] for value in variants], width=0.72)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [value.replace("_", " ") for value in variants], rotation=25, ha="right")
    ax.set(
        ylabel="Median spectral overlap", ylim=(0, 1.05), title="Oscillation-content preservation"
    )
    ax.text(-0.16, 1.05, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1, 0]
    partition_names = list(partition_colors)
    x = np.arange(len(partition_names))
    metrics = (
        (
            "omega energy",
            [
                _group_median(
                    projection_rows,
                    "retained_mass_energy_fraction",
                    role="val",
                    partition=name,
                    signal="omega_absolute",
                )
                for name in partition_names
            ],
            "#56B4E9",
        ),
        (
            "delta-increment energy",
            [
                _group_median(
                    projection_rows,
                    "retained_mass_energy_fraction",
                    role="val",
                    partition=name,
                    signal="delta_increment",
                )
                for name in partition_names
            ],
            "#E69F00",
        ),
        (
            "coherency percentile",
            [
                _group_median(
                    coherency_rows,
                    "correlation_random_partition_percentile",
                    role="val",
                    partition=name,
                )
                for name in partition_names
            ],
            "#009E73",
        ),
    )
    for offset, (label, values, color) in zip((-0.24, 0.0, 0.24), metrics, strict=True):
        ax.bar(x + offset, values, width=0.22, color=color, label=label)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [partition_labels[name] for name in partition_names], rotation=18, ha="right")
    ax.set(ylabel="Fraction / percentile", ylim=(0, 1.05), title="Partition evidence on validation")
    ax.legend(frameon=False, fontsize=6.5)
    ax.text(-0.16, 1.05, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1, 1]
    design = [row for row in design_rows if row["method"] == "M0"]
    representation_style = {
        "generator_10": ("o", "#777777", "10-node"),
        **{name: ("s", partition_colors[name], partition_labels[name]) for name in partition_names},
    }
    for representation, (marker, color, label) in representation_style.items():
        selected = [row for row in design if row["representation"] == representation]
        ax.scatter(
            [row["n_parameters"] for row in selected],
            [row["condition_number"] for row in selected],
            marker=marker,
            color=color,
            s=32,
            label=label,
        )
    ax.set_yscale("log")
    ax.set(
        xlabel="Weak-form parameter count",
        ylabel="Normalized condition number",
        title="Complexity and conditioning",
    )
    ax.legend(frameon=False, fontsize=7)
    ax.text(-0.16, 1.05, "d", transform=ax.transAxes, fontweight="bold", fontsize=10)

    outputs: list[Path] = []
    for suffix, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        path = figure_dir / f"fig1_physical_cluster_audit.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    values7 = [
        _group_mean(
            comparison_rows,
            "native_rmse",
            cohort="highorder",
            representation=name,
            evaluation_partition=name,
            method="M0-3E",
            horizon_s=1.0,
            status="completed",
        )
        for name in partition_names
    ]
    values10 = [
        _group_mean(
            comparison_rows,
            "interarea7_rmse",
            cohort="highorder",
            representation="generator_10",
            evaluation_partition=name,
            method="M0-3E",
            horizon_s=1.0,
            status="completed",
        )
        for name in partition_names
    ]
    x = np.arange(len(partition_names))
    ax.bar(x - 0.18, values10, 0.36, color="#777777", label="10-node model projected")
    ax.bar(
        x + 0.18,
        values7,
        0.36,
        color=[partition_colors[name] for name in partition_names],
        label="native 7-node model",
    )
    ax.set_xticks(x, [partition_labels[name] for name in partition_names], rotation=18, ha="right")
    ax.set(ylabel="High-order-source mean RMSE", title="Matched inter-area prediction (1 s)")
    ax.legend(frameon=False, fontsize=7)
    ax.text(-0.15, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
    for suffix, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        path = figure_dir / f"fig2_highorder_cluster_comparison.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _format(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _write_report_v1(
    result_dir: Path,
    *,
    run_id: str,
    claims: dict[str, Any],
    projection_audit: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    gmd_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
) -> None:
    physical = claims["current_equation_has_effective_physical_meaning"]
    cluster = claims["seven_node_frontend_supported"]
    gmd_table = []
    for horizon in (1.0, 5.0, 10.0):
        gmd_table.append(
            {
                "horizon": horizon,
                "backbone_rmse": _group_mean(
                    gmd_rows, "rmse", variant="backbone", horizon_s=horizon, status="completed"
                ),
                "full_rmse": _group_mean(
                    gmd_rows, "rmse", variant="full", horizon_s=horizon, status="completed"
                ),
                "full_overlap": _group_median(
                    gmd_rows,
                    "spectral_overlap",
                    variant="full",
                    horizon_s=horizon,
                    status="completed",
                ),
                "full_frequency_error": _group_median(
                    gmd_rows,
                    "dominant_frequency_error_hz",
                    variant="full",
                    horizon_s=horizon,
                    status="completed",
                ),
            }
        )
    residual_components = sorted(
        {row["component"] for row in residual_rows},
        key=lambda component: (
            -_group_median(residual_rows, "predicted_to_true_force_rms_ratio", component=component)
        ),
    )
    source_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in dataset_rows:
        source_counts[(str(row["role"]), str(row["source_dataset"]))] += 1
    design_index = {
        (row["cohort"], row["representation"], row["method"]): row for row in design_rows
    }

    lines = [
        "# 冻结方程物理意义与七节点聚合审计",
        "",
        f"**运行标识：** `{run_id}`  ",
        "**证据范围：** F1、seed 324、train/validation only、observed $P_m(t)$ 条件重积分；test 未打开。",
        "",
        "## 摘要判定",
        "",
        f"1. **现有 M0-GMD 方程的有效物理意义：{physical['status']}。** 它的可辩护含义是：在冻结参数、observed $P_m(t)$ 条件下，方程是否保存验证轨迹的振荡频带、主频和有效模态形状，并且残差闭合是否与真实力残差同向。该判定不等于恢复了真实机械阻尼、原始网络拓扑或唯一的机器参数。",
        f"2. **七节点前端：{cluster['status']}。** 严格成立的是质量正交 cluster-consensus 投影；“低通”仅在当前数据的时间频率分离、相干性、条件数和跨区预测证据范围内作为经验描述。",
        "3. **严格图低通与聚类算法新颖性：均拒绝。** 慢相干和惯量聚合已有成熟理论；本工作的创新空间应表述为‘面向弱形式图结构发现的、COI 保持且可审计的条件化前端’，而不是发明七聚类或一般低通滤波。",
        "",
        "## 1. 数据与审计边界",
        "",
        f"训练/验证样本来自两个固定 source。训练 high-order/dense 数量分别为 {source_counts.get(('train', HIGHORDER_SOURCE), 0)}/{source_counts.get(('train', DENSE_SOURCE), 0)}；验证分别为 {source_counts.get(('val', HIGHORDER_SOURCE), 0)}/{source_counts.get(('val', DENSE_SOURCE), 0)}。所有用于拟合的轨迹至少有 101 个真实点；长时模态审计只在具备相应真实时长的验证轨迹上执行。标签只允许冻结后的描述性分层，本次计算没有读取标签字段。",
        "",
        f"聚合矩阵满足 $\\|P^2-P\\|_\\infty={_format(projection_audit['idempotence_error'])}$、$\\|P^TM-MP\\|_\\infty={_format(projection_audit['mass_self_adjoint_error'])}$ 和 $\\|AR-I\\|_\\infty={_format(projection_audit['left_inverse_error'])}$。因此能量分解是代数恒等式，而不是由数据拟合出来的现象。",
        "",
        "## 2. 方程是否保存系统振荡",
        "",
        "采用三组互补指标。第一组是工程模型验证常用的轨迹形状量：RMSE、峰值、峰值时刻与衰减包络。第二组是频谱重叠和主频误差。第三组用 exact DMD 给出窗口内的有效振荡极点，并以频率、阻尼比和模态保证准则（MAC）配对。由于 observed $P_m(t)$ 使系统非自治，DMD 数值是 response descriptor，不宣称为全系统严格特征值。",
        "",
        "| 视界 | M0-3E backbone mean RMSE | M0-GMD full mean RMSE | full 频谱重叠中位数 | full 主频误差中位数/Hz |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in gmd_table:
        lines.append(
            f"| {row['horizon']:.0f} s | {_format(row['backbone_rmse'])} | {_format(row['full_rmse'])} | {_format(row['full_overlap'])} | {_format(row['full_frequency_error'])} |"
        )
    lines.extend(
        [
            "",
            f"5 s 频谱重叠门={'通过' if physical['oscillation_preservation_gate'] else '未通过'}；中位主模态 MAC={_format(physical['median_primary_mode_MAC'])}。这里最重要的对照不是单独看 full 曲线，而是同时看 backbone、static-only、memory-only 和 linear-only：如果 full 只降低点误差却损坏频带或产生长时失败，就不能称为保留物理振荡。",
            "",
            "## 3. 残差项揭示了什么结构",
            "",
            "对真实验证路径先由冻结 M0-3E 计算基线力，再用 $m\\Delta\\omega/\\Delta t$ 构造总力，从二者之差得到力空间残差。GMD 闭合在同一真实路径上计算，避免把闭环状态误差混入残差结构判断。零总力、共同角平移不变和正时间常数是结构保证；方向余弦、解释能量和残差功率相关才是数据证据。",
            "",
            f"full 闭合的力方向余弦中位数为 {_format(physical['median_force_cosine_alignment'])}，残差解释能量中位数为 {_format(physical['median_residual_explained_energy'])}。残差结构门={'通过' if physical['residual_structure_gate'] else '未通过'}。残差功率可正可负，因此当前 GMD 只能解释为遗漏的储能、控制与非线性耦合的有效组合，不能直接称为额外阻尼。",
            "",
            "按预测力 RMS/真实残差 RMS 排序的分量：",
            "",
            "| 分量 | 力方向余弦中位数 | 残差解释能量中位数 | 功率相关中位数 | 力 RMS 比 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for component in residual_components:
        lines.append(
            f"| {component} | {_format(_group_median(residual_rows, 'force_cosine_alignment', component=component))} | {_format(_group_median(residual_rows, 'residual_explained_energy', component=component))} | {_format(_group_median(residual_rows, 'residual_power_correlation', component=component))} | {_format(_group_median(residual_rows, 'predicted_to_true_force_rms_ratio', component=component))} |"
        )
    lines.extend(
        [
            "",
            "## 4. 七节点是否仍由当前数据支持",
            "",
            f"验证集速度增量和角度增量的质量能量保留中位数分别为 {_format(cluster['validation_omega_dynamic_energy_retention'])} 与 {_format(cluster['validation_delta_dynamic_energy_retention'])}；固定三个机组对在同型随机三对分组中的相关性分位数中位数为 {_format(cluster['validation_fixed_partition_correlation_percentile'])}。",
            f"速度补空间的中位频率为 {_format(cluster['high_component_median_frequency_hz'])} Hz，cluster-consensus 分量为 {_format(cluster['low_component_median_frequency_hz'])} Hz；1 Hz 以上能量占比分别为 {_format(cluster['high_component_above_1hz_share'])} 和 {_format(cluster['low_component_above_1hz_share'])}。这决定了是否可以使用‘经验低通’措辞。",
            "",
            "弱形式设计矩阵的比较：",
            "",
            "| 数据队列 | 表示 | 参数数 | 有效秩 | 条件数 | stable rank | mutual coherence |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort in ("combined", "highorder"):
        for representation in ("generator_10", "clustered_7"):
            row = design_index[(cohort, representation, "M0")]
            lines.append(
                f"| {cohort} | {representation} | {row['n_parameters']} | {row['effective_rank']} | {_format(row['condition_number'])} | {_format(row['stable_rank'])} | {_format(row['mutual_coherence'])} |"
            )
    lines.extend(
        [
            "",
            f"在 high-order 验证集的匹配七节点指标上，原生七节点 M0-3E 的 1 s mean RMSE={_format(cluster['highorder_M0_3E_native_7node_mean_rmse'])}，十节点 M0-3E 预测后再投影的 mean RMSE={_format(cluster['highorder_M0_3E_10node_projected_to_7_mean_rmse'])}。这项对照把‘表示误差’和‘模型拟合误差’分开：七节点重构十机的误差必须包含不可消除的投影底噪，而跨区七节点指标才是公平比较。",
            "",
            "## 5. 创新性与论文表述",
            "",
            "不建议写成‘提出七节点聚类低通滤波器’。慢相干、惯量聚合以及由相干机组形成动态等值均为已有技术。可辩护的新意是：固定先验分组被写成 COI 保持的质量正交投影；随后用 train-only 弱形式条件数、动态能量分解、随机同型分组分位数和闭环重积分共同审计；最后把 GMD 的结构发现限制在该商空间内。这是一个可证伪的识别前端，而不是无条件降维。",
            "",
            "## 6. 工程指标与文献对应",
            "",
            "NERC MOD-033 要求动态仿真与实际响应比较并建立有意义的差异准则；MOD-027 强调调速/有功频率控制的模型结构与记录响应一致。WECC 指南建议比较曲线形状、幅值、响应速度、上升时间、超调、带宽与延迟，并指出 SSE/欧氏范数有用但不宜被简化为唯一刚性门。Hauer 的 Prony 分析以及后续 mode-meter 工作把频率、阻尼、强度和相对相位作为振荡响应的工程描述量。因此本报告不以单一 RMSE 代替物理验证。",
            "",
            "关键依据：[Hauer 1991](https://doi.org/10.1109/59.119247)、[Trudnowski et al. 2008](https://doi.org/10.1109/TPWRS.2008.919415)、[Hatziargyriou et al. 2021](https://doi.org/10.1109/TPWRS.2020.3041774)、[Chow et al. 1995](https://doi.org/10.1109/59.387903)、[Tyuryukanov et al. 2021](https://doi.org/10.1109/TPWRS.2020.3009628)、[Koochi 2019](https://doi.org/10.1049/iet-gtd.2018.7028)、[low-pass GSP guide](https://arxiv.org/abs/2008.01305)、[SINDy convergence](https://doi.org/10.1137/18M1189828)、[weak-SINDy convergence](https://doi.org/10.1137/22M1526782)、[NERC MOD-027-1](https://www.nerc.com/globalassets/standards/reliability-standards/mod/mod-027-1.pdf)、[NERC MOD-033-1](https://www.nerc.com/globalassets/standards/projects/2010-03/mod_033_1_model_validation_redline_2013_1205.pdf)、[WECC validation requirements](https://www.wecc.org/sites/default/files/documents/meeting/2024/WECC%20Gen%20Fac%20Testing%20and%20Model%20Validation%20Rqmts%20v%204-23-2020.pdf)。",
            "",
            "## 7. 限制与下一步",
            "",
            "- 这是单 fold、单 seed 的 train/validation 证据，足以决定方法是否值得保留，但不足以作五折总体显著性声明。",
            "- observed $P_m(t)$ 是条件上界。完成物理结构确认后，应把同一审计复用于因果 $P_m$ ODE；不能把本结果直接写成在线预测性能。",
            "- DMD/频谱用于有限窗口响应描述。要声称严格小信号特征值，应在线性化自治模型上计算 Jacobian，并与完整系统同工况的 eigenanalysis 配对。",
            "- 七节点固定划分若要成为方法组件，应在训练集上定义可复现的慢相干选择准则，并在验证集冻结；本报告只审计既有划分，没有重新优化它。",
            "",
            "## 产物索引",
            "",
            "- `figures/fig1_physical_cluster_audit.*`：振荡、频谱、聚合能量与条件数证据。",
            "- `figures/fig2_highorder_cluster_comparison.*`：high-order 匹配跨区预测。",
            "- `gmd_rollout_metrics.csv`、`gmd_modal_metrics.csv`、`residual_structure_metrics.csv`：方程物理意义明细。",
            "- `projection_metrics.csv`、`coherency_metrics.csv`、`design_conditioning.csv`、`representation_rollout_metrics.csv`：七节点证据明细。",
            "- `claim_assessment.json`：逐项主张定级；`diagnostic_process_acceptance.json`：过程完整性门。",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    (result_dir / "report.md").write_text(markdown, encoding="utf-8")

    body = html.escape(markdown)
    body = body.replace("\n", "<br>\n")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>方程物理意义与七节点聚合审计</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f4f5f7;color:#202124;line-height:1.7}}
main{{max-width:1040px;margin:28px auto;background:white;padding:44px 54px;box-shadow:0 2px 16px #0001}}
h1{{font-size:28px;border-bottom:3px solid #b33b2e;padding-bottom:12px}} h2{{margin-top:34px;color:#6b211a}}
.status{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}} .card{{padding:16px;border:1px solid #ddd;border-left:5px solid #b33b2e;background:#fafafa}}
img{{max-width:100%;margin:18px 0;border:1px solid #ddd}} pre{{white-space:pre-wrap;font-family:Arial,'Microsoft YaHei',sans-serif}}
a{{color:#8b2f25}} @media(max-width:720px){{main{{margin:0;padding:24px}}.status{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>冻结方程物理意义与七节点聚合审计</h1>
<div class="status"><div class="card"><b>M0-GMD 物理意义</b><br>{html.escape(str(physical["status"]))}</div><div class="card"><b>七节点前端</b><br>{html.escape(str(cluster["status"]))}</div></div>
<img src="figures/fig1_physical_cluster_audit.svg" alt="物理意义与聚合审计主图">
<img src="figures/fig2_highorder_cluster_comparison.svg" alt="高阶模型七节点对照">
<pre>{body}</pre>
</main></body></html>"""
    (result_dir / "report.html").write_text(document, encoding="utf-8")


def _write_report(
    result_dir: Path,
    *,
    run_id: str,
    claims: dict[str, Any],
    partitions: dict[str, tuple[tuple[int, ...], ...]],
    projection_audit: dict[str, dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    coherency_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    gmd_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
) -> None:
    """Write the final Chinese, paper-oriented audit report."""
    physical = claims["current_equation_has_effective_physical_meaning"]
    cluster = claims["seven_node_frontend_supported"]
    partition_claims = cluster["partition_assessments"]
    source_claim = claims["mixed_source_parameter_consistency"]
    source_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in dataset_rows:
        source_counts[(str(row["role"]), str(row["source_dataset"]))] += 1
    design_index = {
        (row["cohort"], row["representation"], row["method"]): row for row in design_rows
    }

    lines = [
        "# 冻结方程物理意义、数据同质性与七节点聚合复核报告",
        "",
        f"**运行标识：** `{run_id}`  ",
        "**证据范围：** F1、seed 324、train/validation only、observed $P_m(t)$ 条件重积分；test 未读取。  ",
        "**研究定位：** 正式论文前的关键节点审计，不把诊断结果外推为五折统计结论。",
        "",
        "## 摘要判定",
        "",
        (
            f"1. **当前 M0-GMD 方程的有效物理意义：{physical['status']}。** 残差结构门为 "
            f"{'通过' if physical['residual_structure_gate'] else '未通过'}，5 s 振荡保持门为 "
            f"{'通过' if physical['oscillation_preservation_gate'] else '未通过'}。因此它目前可解释为 observed-$P_m$ 条件下的有效闭合，不能解释成真实机组阻尼或网络拓扑的唯一恢复。"
        ),
        f"2. **七节点前端：{cluster['status']}。** 三种分区已在同一数据、同一弱形式和同一验证协议下重算。历史 higher-order 的七节点分组与当前 canonical 分组不同；此前把两者视为同一方案是不成立的。",
        f"3. **混合数据源参数一致性：{source_claim['status']}。** 当前 canonical 数据含 high-order 与 dense 两个响应来源。源特定拟合和有效参数差异可识别响应机制是否兼容，但缺少原始仿真器参数快照，因此不能据此断言物理参数确实变化。",
        "4. **严格‘图低通滤波器’主张拒绝。** 质量正交投影可以被称为经验时间尺度分离或 slow-coherency 前端；没有图移位算子和频率响应衰减比时，不能称为严格 graph low-pass filter。",
        "",
        "## 1. 第一性原理：方程应当保留什么",
        "",
        "对受迫二阶摆动系统，逐点 RMSE 不是充分的物理验证。一个可接受的约化方程至少应同时满足：",
        "",
        r"- **运动学与守恒结构：** $\dot\delta=\omega$、内部边力零总和、共同角平移不变、COI 漂移受控；",
        "- **时域响应：** 峰值、峰值时刻、衰减包络、上升/延迟及长视界无发散；",
        "- **振荡内容：** 主频、阻尼比、频谱重叠和模态形状 MAC；",
        "- **残差可解释性：** 在真实路径上，闭合项应与未解释力同向并解释正的残差能量，而不能只靠闭环误差抵消；",
        "- **跨轨迹稳定性：** 相同系数在独立故障轨迹上保持有效，且按数据源分层后不出现系统性反转。",
        "",
        r"本审计在真实验证路径上构造 $r=m\,\Delta\omega/\Delta t-f_{M0}$，再比较 GMD 各结构分量与 $r$ 的力方向余弦、解释能量及功率相关。闭环重积分另行比较 backbone、static-only、memory-only、linear-only 和 full，避免把状态漂移误当成残差结构。",
        "",
        "## 2. 方程的工程验证结果",
        "",
        "| 视界 | 验证轨迹数 | backbone mean RMSE | full mean RMSE | full 频谱重叠中位数 | full 主频误差中位数/Hz | full 失败数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in (1.0, 5.0, 10.0):
        lines.append(
            f"| {horizon:.0f} s | {sum(row.get('variant') == 'full' and row.get('horizon_s') == horizon and row.get('status') == 'completed' for row in gmd_rows)} "
            f"| {_format(_group_mean(gmd_rows, 'rmse', variant='backbone', horizon_s=horizon, status='completed'))} "
            f"| {_format(_group_mean(gmd_rows, 'rmse', variant='full', horizon_s=horizon, status='completed'))} "
            f"| {_format(_group_median(gmd_rows, 'spectral_overlap', variant='full', horizon_s=horizon, status='completed'))} "
            f"| {_format(_group_median(gmd_rows, 'dominant_frequency_error_hz', variant='full', horizon_s=horizon, status='completed'))} "
            f"| {sum(row.get('variant') == 'full' and row.get('horizon_s') == horizon and row.get('status') == 'failed' for row in gmd_rows)} |"
        )
    lines.extend(
        [
            "",
            (
                f"5 s 综合判据为：频谱重叠 {_format(physical['median_5s_spectral_overlap'])}，主频误差 "
                f"{_format(physical['median_5s_dominant_frequency_error_hz'])} Hz，第一匹配模态 MAC "
                f"{_format(physical['median_primary_mode_MAC'])}，第一匹配模态阻尼比绝对误差中位数 "
                f"{_format(physical['median_primary_mode_absolute_damping_ratio_error'])}。阈值是本研究的预注册诊断阈值，并非电网规范的通用强制限值。"
            ),
            "",
            "### 2.1 残差结构",
            "",
            "| 数据源 | 分量 | 力方向余弦中位数 | 残差解释能量中位数 | 残差功率相关中位数 | 预测/真实力 RMS |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    components = sorted({str(row["component"]) for row in residual_rows})
    for source in (HIGHORDER_SOURCE, DENSE_SOURCE):
        for component in components:
            lines.append(
                f"| {source} | {component} "
                f"| {_format(_group_median(residual_rows, 'force_cosine_alignment', source_dataset=source, component=component))} "
                f"| {_format(_group_median(residual_rows, 'residual_explained_energy', source_dataset=source, component=component))} "
                f"| {_format(_group_median(residual_rows, 'residual_power_correlation', source_dataset=source, component=component))} "
                f"| {_format(_group_median(residual_rows, 'predicted_to_true_force_rms_ratio', source_dataset=source, component=component))} |"
            )
    lines.extend(
        [
            "",
            "正的方向余弦和解释能量说明闭合项捕获了某一类系统性遗漏力；但功率可正可负，且不同数据源强弱明显不同时，它更像储能、控制与非线性耦合的有效组合，而不是可直接命名的物理阻尼。",
            "",
            "## 3. 七节点分区谱系与低通假设",
            "",
            "历史 higher-order 代码使用 `[[0,7],[1,2],[3],[4],[5,6],[8],[9]]`；当前 canonical 常量使用 `[[0,2],[1],[3],[4],[5],[6,7],[8,9]]`。本轮另外只用训练集轨迹距离发现三个互不重叠的机组对，验证集只用于冻结后审计。",
            "",
            r"质量加权聚合定义为 $R_{cg}=m_g/\sum_{j\in c}m_j$，复制矩阵为 $A$，投影 $P=AR$。三个分区均满足 $P^2=P$、$RA=I$ 和 $P^TM=MP$，因此这是 COI 保持的质量正交 cluster-consensus 投影。",
            "",
            r"| 分区 | 机组组别 | 状态 | $\omega$绝对能量 | $\Delta\delta$能量 | 相干随机分位数 | 低/高分量中位频率 Hz | >1 Hz 低/高能量 | 通过门数 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, partition in partitions.items():
        row = partition_claims[name]
        groups = " ".join("(" + ",".join(str(node) for node in group) + ")" for group in partition)
        lines.append(
            f"| {name} | {groups} | {row['status']} "
            f"| {_format(row['validation_omega_absolute_energy_retention'])} "
            f"| {_format(row['validation_delta_increment_energy_retention'])} "
            f"| {_format(row['validation_correlation_random_partition_percentile'])} "
            f"| {_format(row['low_component_median_frequency_hz'])}/{_format(row['high_component_median_frequency_hz'])} "
            f"| {_format(row['low_component_above_1hz_share'])}/{_format(row['high_component_above_1hz_share'])} "
            f"| {row['passed_gate_count']}/6 |"
        )
    lines.extend(
        [
            "",
            "经验低通只能在以下可证伪含义下使用：cluster-consensus 分量保留主要质量能量，而补空间具有更高的中位频率及更高的高频能量占比。若这两个频率排序条件不成立，则‘低通’叙述必须删除，即使参数数目和条件数有所改善。",
            "",
            "### 3.1 弱形式可辨识性",
            "",
            "| 数据队列 | 表示 | 参数数 | 有效秩 | 条件数 | stable rank | mutual coherence |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort in ("combined", "highorder", "dense"):
        for representation in ("generator_10", *partitions):
            row = design_index[(cohort, representation, "M0")]
            lines.append(
                f"| {cohort} | {representation} | {row['n_parameters']} | {row['effective_rank']} "
                f"| {_format(row['condition_number'])} | {_format(row['stable_rank'])} | {_format(row['mutual_coherence'])} |"
            )
    lines.extend(
        [
            "",
            "### 3.2 higher-order 复核",
            "",
            "| 分区 | 原生 7 节点 M0-3E RMSE | 10 节点 M0-3E 后投影 RMSE | 选中 alpha | 选中 tau/s | 真三阶状态激活 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, row in partition_claims.items():
        lines.append(
            f"| {name} | {_format(row['highorder_M0_3E_native_mean_rmse'])} "
            f"| {_format(row['highorder_M0_3E_10node_projected_mean_rmse'])} "
            f"| {_format(row['selected_third_order_alpha'])} | {_format(row['selected_third_order_tau'])} "
            f"| {'是' if row['gates']['active_third_order_state'] else '否（退化为 M0）'} |"
        )
    lines.extend(
        [
            "",
            "因此，‘七节点在 higher-order 上更好’必须同时满足两件事：公平的 matched inter-area 指标更优，以及三阶内部状态确实被验证集选择为非零。若 alpha=0，数值上再好也只是二阶 M0 的结果，不能作为 higher-order 证据。",
            "",
            "## 4. 两类数据是否来自同一有效动力学",
            "",
            f"训练集 high-order/dense 样本数为 {source_counts.get(('train', HIGHORDER_SOURCE), 0)}/{source_counts.get(('train', DENSE_SOURCE), 0)}；验证集为 {source_counts.get(('val', HIGHORDER_SOURCE), 0)}/{source_counts.get(('val', DENSE_SOURCE), 0)}。两者共享 canonical 通道定义和 $M$ prior，但 schema 一致不等于隐藏的调速器、励磁器、阻尼或网络参数一致。",
            "",
            "| 指标 | high-order 专用拟合 | combined 在 high-order | dense 专用拟合 | combined 在 dense |",
            "|---|---:|---:|---:|---:|",
            (
                f"| 10 节点 M0、1 s mean RMSE | {_format(source_claim['highorder_specific_M0_mean_rmse'])} "
                f"| {_format(source_claim['combined_M0_on_highorder_mean_rmse'])} "
                f"| {_format(source_claim['dense_specific_M0_mean_rmse'])} "
                f"| {_format(source_claim['combined_M0_on_dense_mean_rmse'])} |"
            ),
            "",
            "| 表示 | 参数 | high-order/dense 余弦 | 对称相对 L2 差 | high-order 中位数 | dense 中位数 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in parameter_rows:
        lines.append(
            f"| {row['representation']} | {row['parameter']} | {_format(row['highorder_dense_cosine'])} "
            f"| {_format(row['symmetric_relative_l2_difference'])} | {_format(row['highorder_median'])} "
            f"| {_format(row['dense_median'])} |"
        )
    lines.extend(
        [
            "",
            "这组检验回答的是‘同一个有效二阶闭合能否同时解释两类轨迹’。若源特定模型明显优于 combined，或有效系数显著偏离，应把数据源作为审计分层，并考虑只训练在可证明同构的高质量样本上；但在取得原仿真配置哈希前，不应写成‘混入了不同真实系统参数’。",
            "",
            "## 5. 工程指标与学术依据",
            "",
            "- 电力系统振荡识别通常报告模态频率、阻尼、幅值/强度与相位；本报告用频谱、DMD 极点和 MAC 形成互补证据。",
            "- 动态模型验证工程指南强调曲线形状、幅值、响应速度、上升时间、超调、带宽和延迟，并要求建立有意义的差异准则；不存在适用于所有装置和工况的单一 RMSE 门限。",
            "- slow coherency 和惯量聚合是成熟理论。本文可以主张的创新不是‘发明七聚类’，而是 train-only 分区、COI 保持投影、弱形式条件数、随机同型分区基线与闭环重积分组成的可审计发现前端。",
            "- 严格图低通需要明确图移位/拉普拉斯算子及其频率响应。当前结果最多支持经验时间尺度分离，不能替代该证书。",
            "",
            "关键文献：[Hauer 1991](https://doi.org/10.1109/59.119247)、[Trudnowski et al. 2008](https://doi.org/10.1109/TPWRS.2008.919415)、[Hatziargyriou et al. 2021](https://doi.org/10.1109/TPWRS.2020.3041774)、[Chow et al. 1995](https://doi.org/10.1109/59.387903)、[Tyuryukanov et al. 2021](https://doi.org/10.1109/TPWRS.2020.3009628)、[Koochi 2019](https://doi.org/10.1049/iet-gtd.2018.7028)、[graph low-pass guide](https://arxiv.org/abs/2008.01305)、[SINDy convergence](https://doi.org/10.1137/18M1189828)、[Weak-SINDy convergence](https://doi.org/10.1137/22M1526782)、[NERC MOD-027-1](https://www.nerc.com/globalassets/standards/reliability-standards/mod/mod-027-1.pdf)、[NERC MOD-033-1](https://www.nerc.com/globalassets/standards/projects/2010-03/mod_033_1_model_validation_redline_2013_1205.pdf)、[WECC model-validation requirements](https://www.wecc.org/sites/default/files/documents/meeting/2024/WECC%20Gen%20Fac%20Testing%20and%20Model%20Validation%20Rqmts%20v%204-23-2020.pdf)。",
            "",
            "## 6. 论文可接受的主张边界",
            "",
            f"- 方程物理意义应写为 **{physical['status']} 的有效约化闭合**；不能写成真实 $D$、Kron 拓扑或机组参数恢复。",
            f"- 七节点应写为 **{cluster['status']} 的可审计降阶前端**，并逐字注明所用分区。当前诊断中相对最好的条件候选为 `{cluster['best_conditional_candidate']}`，但它未通过全部门，既不是 test-selected 最终模型，也没有被晋级。",
            "- 若真三阶状态未激活，应把 M0-3E 结果标成退化二阶对照，不能用来支持 higher-order 机制。",
            "- observed $P_m(t)$ 是条件上界；因果部署必须在相同审计通过后换成 $P_m$ ODE，不能直接声称在线预测性能。",
            "- 单 fold、单 seed 足以否证过强主张和筛选方法方向，但不足以形成五折显著性结论。",
            "",
            "## 7. 产物与可复现性",
            "",
            "- `partition_definitions.json`、`train_partition_pair_scores.csv`：分区谱系和 train-only 发现证据；",
            "- `projection_metrics.csv`、`coherency_metrics.csv`：能量、尺度分离与随机分区基线；",
            "- `design_conditioning.csv`、`parameter_consistency.csv`：弱形式可辨识性及数据源有效参数一致性；",
            "- `representation_rollout_metrics.csv`：同指标 10/7 节点和 M0/M0-3E 闭环比较；",
            "- `gmd_rollout_metrics.csv`、`gmd_modal_metrics.csv`、`residual_structure_metrics.csv`：方程物理意义证据；",
            "- `claim_assessment.json`、`diagnostic_process_acceptance.json`、`artifact_hashes.json`：主张门、过程门与不可变哈希。",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    (result_dir / "report.md").write_text(markdown, encoding="utf-8")
    escaped = html.escape(markdown)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结方程物理意义、数据同质性与七节点聚合复核</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f4f5f7;color:#202124;line-height:1.7}}
main{{max-width:1080px;margin:28px auto;background:white;padding:44px 54px;box-shadow:0 2px 16px #0001}}
h1{{font-size:28px;border-bottom:3px solid #9d2f25;padding-bottom:12px}}
img{{max-width:100%;margin:18px 0;border:1px solid #ddd}} pre{{white-space:pre-wrap;font-family:Arial,'Microsoft YaHei',sans-serif}}
a{{color:#8b2f25}} @media(max-width:720px){{main{{margin:0;padding:24px}}}}
</style></head><body><main>
<h1>冻结方程物理意义、数据同质性与七节点聚合复核</h1>
<img src="figures/fig1_physical_cluster_audit.svg" alt="方程物理意义与分区审计主图">
<img src="figures/fig2_highorder_cluster_comparison.svg" alt="higher-order 数据源七节点比较">
<pre>{escaped}</pre>
</main></body></html>"""
    (result_dir / "report.html").write_text(document, encoding="utf-8")


def _latest_result(root: Path, experiment_name: str, stage: str) -> Path:
    prefix = f"{experiment_name}-{stage}-"
    candidates = sorted(
        path
        for path in (root / "results").glob(f"{prefix}*")
        if path.is_dir() and not (path / "RUN_FAILED.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed result for {experiment_name}/{stage}")
    return candidates[-1]


def _verify_artifact_hashes(diagnostic_dir: Path) -> list[str]:
    expected = json.loads((diagnostic_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    failures = []
    for relative, digest in expected.items():
        path = diagnostic_dir / relative
        if not path.is_file() or _sha256(path) != digest:
            failures.append(relative)
    return failures


def _run_adversarial_audit(
    *, root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic_dir = _latest_result(root, str(config["diagnostic_experiment"]), "diagnostic")
    claims = json.loads((diagnostic_dir / "claim_assessment.json").read_text(encoding="utf-8"))
    process = json.loads(
        (diagnostic_dir / "diagnostic_process_acceptance.json").read_text(encoding="utf-8")
    )
    lineage = json.loads((diagnostic_dir / "lineage.json").read_text(encoding="utf-8"))
    design_rows = _read_csv(diagnostic_dir / "design_conditioning.csv")
    rollout_rows = _read_csv(diagnostic_dir / "representation_rollout_metrics.csv")
    hash_failures = _verify_artifact_hashes(diagnostic_dir)

    checkpoint_validation_consistent = True
    for row in design_rows:
        if row.get("method") != "M0-3E":
            continue
        metadata = (
            diagnostic_dir / "checkpoints" / (f"{row['cohort']}-{row['representation']}-M0-3E.json")
        )
        diagnostics = json.loads(metadata.read_text(encoding="utf-8"))["diagnostics"]
        expected_rmse = float(diagnostics["selected_validation"]["raw_mean_rmse"])
        checkpoint_validation_consistent &= bool(
            np.isclose(float(row["validation_mean_rmse"]), expected_rmse, rtol=0.0, atol=1e-14)
        )

    cluster_claim = claims["seven_node_frontend_supported"]
    partition_claims = cluster_claim["partition_assessments"]
    bounded_claims = bool(
        claims["strict_graph_low_pass_claim"]["status"] == "REJECTED"
        and claims["clustering_algorithm_novelty"]["status"] == "REJECTED"
        and claims["machine_parameter_or_physical_topology_recovery"]["status"] == "REJECTED"
        and claims["current_equation_has_effective_physical_meaning"]["status"] != "SUPPORTED"
        and partition_claims["clustered_7_current"]["status"] == "REJECTED"
        and cluster_claim["status"] != "SUPPORTED"
    )
    gates = {
        "diagnostic_artifact_hashes_match": not hash_failures,
        "diagnostic_process_gate_passed": process.get("status") == "PASS",
        "test_data_unopened": not lineage.get("test_ids_read"),
        "labels_unused": not lineage.get("labels_used_for_fit_selection_or_thresholds"),
        "partition_discovery_train_only": bool(lineage.get("partition_discovery_used_train_only")),
        "legacy_current_lineage_explicit": bool(
            claims["partition_lineage"]["legacy_and_current_partitions_are_different"]
        ),
        "m03e_raw_validation_metric_matches_checkpoint": checkpoint_validation_consistent,
        "all_comparison_rollouts_completed": not any(
            row.get("status") != "completed" for row in rollout_rows
        ),
        "scientific_claims_are_bounded": bounded_claims,
        "conditional_candidate_not_promoted": bool(
            cluster_claim.get("candidate_is_not_test_selected_or_promoted")
        ),
    }
    findings = [
        {
            "severity": "P0",
            "finding": "historical and canonical seven-node partitions differed",
            "disposition": "corrected by explicit current/legacy/train-only comparison",
        },
        {
            "severity": "P1",
            "finding": "current canonical partition fails energy, coherency, scale-separation and predictive gates",
            "disposition": "current partition claim rejected",
        },
        {
            "severity": "P1",
            "finding": "no seven-node candidate beats the matched ten-node projection or retains 95% dynamic energy",
            "disposition": "seven-node frontend remains conditional and unpromoted",
        },
        {
            "severity": "P1",
            "finding": "M0-GMD improves pointwise RMSE but fails the 5 s oscillation-preservation gate",
            "disposition": "equation physical meaning remains conditional",
        },
        {
            "severity": "P1",
            "finding": "high-order and dense source-specific effective parameters are inconsistent",
            "disposition": "reported as response-regime evidence, not proof of changed simulator parameters",
        },
        {
            "severity": "P2",
            "finding": "slow coherency is established and the temporal projection lacks a graph-frequency response certificate",
            "disposition": "algorithm novelty and strict graph-low-pass claims rejected",
        },
    ]
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "status": status,
        "decision": "ACCEPTABLE_WITH_BOUNDED_SCIENTIFIC_CLAIMS" if status == "PASS" else "REJECT",
        "run_id": run_id,
        "diagnostic_result": diagnostic_dir.name,
        "gates": gates,
        "hash_failures": hash_failures,
        "findings": findings,
        "external_review_note": "The configured external review transport was blocked by the execution security boundary; this artifact records the repository-local adversarial audit.",
    }
    _json(result_dir / "adversarial_review.json", review)
    lines = [
        "# 对抗性审查与主张压力测试",
        "",
        f"**判定：{review['decision']}**  ",
        f"**诊断来源：** `{diagnostic_dir.name}`  ",
        "",
        "## 门控",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in gates.items())
    lines.extend(["", "## 发现与处置", ""])
    lines.extend(
        f"- **{row['severity']}** {row['finding']}；处置：{row['disposition']}。"
        for row in findings
    )
    lines.extend(
        [
            "",
            "## 可接收边界",
            "",
            "过程和证据链可接收，但科学主张必须保持条件性：不得把 GMD 称为真实参数恢复，不得把七节点称为已晋级模型，不得把经验时间尺度分离称为严格图低通。",
        ]
    )
    (result_dir / "adversarial_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "PASS":
        raise RuntimeError("adversarial audit rejected the diagnostic")


def _run_acceptance(*, root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    diagnostic_dir = _latest_result(root, str(config["diagnostic_experiment"]), "diagnostic")
    adversarial_dir = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_audit"
    )
    diagnostic = json.loads(
        (diagnostic_dir / "diagnostic_process_acceptance.json").read_text(encoding="utf-8")
    )
    adversarial = json.loads(
        (adversarial_dir / "adversarial_review.json").read_text(encoding="utf-8")
    )
    gates = {
        "diagnostic_process_passed": diagnostic.get("status") == "PASS",
        "diagnostic_hashes_match": not _verify_artifact_hashes(diagnostic_dir),
        "adversarial_review_passed": adversarial.get("status") == "PASS",
        "adversarial_decision_is_bounded": adversarial.get("decision")
        == "ACCEPTABLE_WITH_BOUNDED_SCIENTIFIC_CLAIMS",
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    acceptance = {
        "status": status,
        "decision": "ACCEPT" if status == "PASS" else "REJECT",
        "acceptance_scope": "single-fold single-seed method and evidence package; no five-fold significance or causal-deployment claim",
        "run_id": run_id,
        "diagnostic_result": diagnostic_dir.name,
        "adversarial_result": adversarial_dir.name,
        "gates": gates,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# 最终验收\n\n"
        f"**决定：{acceptance['decision']}**\n\n"
        "接收范围限于单 fold、单 seed 的方法与证据包。物理意义和七节点结论保持条件性；test 仍封闭，observed Pm 仍是条件上界。\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("final acceptance failed")


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if stage == "adversarial_audit":
        _run_adversarial_audit(root=root, config=config, result_dir=result_dir, run_id=run_id)
        return
    if stage == "acceptance":
        _run_acceptance(root=root, config=config, result_dir=result_dir, run_id=run_id)
        return
    if stage != "diagnostic":
        raise RuntimeError(f"unsupported physical-cluster audit stage: {stage}")
    if config.get("test_access") != "forbidden" or config.get("pm_mode") != "observed":
        raise RuntimeError("audit requires forbidden test access and observed-Pm conditional input")

    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold=str(config["fold"]),
    )
    minimum_steps = int(config["minimum_fit_steps"])
    train_ids = [
        event_id
        for event_id in split.train_ids
        if store.metadata(event_id).valid_length >= minimum_steps
    ]
    val_ids = [
        event_id
        for event_id in split.val_ids
        if store.metadata(event_id).valid_length >= minimum_steps
    ]
    if not train_ids or not val_ids:
        raise RuntimeError("no eligible train/validation samples")

    dataset_rows: list[dict[str, Any]] = []
    for role, ids in (("train", train_ids), ("val", val_ids)):
        for event_id in ids:
            metadata = store.metadata(event_id)
            dataset_rows.append(
                {
                    "role": role,
                    "event_id": event_id,
                    "source_dataset": metadata.source_dataset,
                    "valid_length": metadata.valid_length,
                    "available_horizon_s": metadata.available_horizon_s,
                    "quality_status": metadata.quality_status,
                }
            )

    request = ModelDataRequest.initial_state_rollout("observed")
    inertia = store.dataset_metadata.M_prior
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {"generator_10": {}}
    for role, ids in (("train", train_ids), ("val", val_ids)):
        adapted = [
            adapt_event(
                store.load_event(event_id),
                inertia_prior=inertia,
                request=request,
                representation="generator_10",
            )
            for event_id in ids
        ]
        for sample in adapted:
            audit_model_sample(sample)
        samples["generator_10"][role] = adapted

    train_partition, pair_score_rows = _discover_train_partition(
        samples["generator_10"]["train"], float(config["modal_horizon_s"])
    )
    partitions: dict[str, tuple[tuple[int, ...], ...]] = {
        "clustered_7_current": tuple(tuple(group) for group in CLUSTERED_7_PARTITION),
        "clustered_7_legacy": LEGACY_7_PARTITION,
        "clustered_7_train_discovered": train_partition,
    }
    for name, partition in partitions.items():
        samples[name] = {}
        for role in ("train", "val"):
            samples[name][role] = [
                _aggregate_adapted_sample(sample, partition, name)
                for sample in samples["generator_10"][role]
            ]
            for sample in samples[name][role]:
                audit_model_sample(sample)

    canonical_current = adapt_event(
        store.load_event(train_ids[0]),
        inertia_prior=inertia,
        request=request,
        representation="clustered_7",
    )
    derived_current = samples["clustered_7_current"]["train"][0]
    current_adapter_identity_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(canonical_current["training_context"]["targets"][field])
                    - np.asarray(derived_current["training_context"]["targets"][field])
                )
            )
        )
        for field in ("delta", "omega")
    )

    checkpoint = _frozen_gmd_checkpoint(root, str(config["frozen_gmd_checkpoint"]))
    source_run_dir = checkpoint.parents[1]
    selection_marker_path = source_run_dir / "selection_marker.json"
    selection_marker = json.loads(selection_marker_path.read_text(encoding="utf-8"))
    if not selection_marker.get("selection_complete") or selection_marker.get("test_ids_read"):
        raise RuntimeError("frozen GMD checkpoint lacks a train/validation-only selection marker")
    frozen_gmd = GraphModalDiscoveryM0Model.load(checkpoint)
    gmd_physics_audit = frozen_gmd.physics_audit()
    if gmd_physics_audit.get("status") != "pass":
        raise RuntimeError("frozen GMD checkpoint failed its structural physics audit")

    mass10 = np.asarray(samples["generator_10"]["train"][0]["model_inputs"]["static"]["mass"])
    mass_matrix = np.diag(mass10)
    projection_audit: dict[str, dict[str, Any]] = {}
    for name, partition in partitions.items():
        aggregate, reconstruct, projection = cluster_aggregation_matrices(mass10, partition)
        projection_audit[name] = {
            "left_inverse_error": float(np.max(np.abs(aggregate @ reconstruct - np.eye(7)))),
            "idempotence_error": float(np.max(np.abs(projection @ projection - projection))),
            "mass_self_adjoint_error": float(
                np.max(np.abs(projection.T @ mass_matrix - mass_matrix @ projection))
            ),
            "rank": int(np.linalg.matrix_rank(projection)),
            "mass_orthogonal_projection": True,
        }
    projection_audit["clustered_7_current"]["canonical_adapter_identity_error"] = (
        current_adapter_identity_error
    )

    random_pair_sets = _random_pair_sets(int(config["random_partition_draws"]), int(config["seed"]))
    projection_rows: list[dict[str, Any]] = []
    coherency_rows: list[dict[str, Any]] = []
    for name, partition in partitions.items():
        for role in ("train", "val"):
            role_projection, role_coherency = _projection_and_coherency_audit(
                samples["generator_10"][role],
                role=role,
                partition_name=name,
                partition=partition,
                random_pair_sets=random_pair_sets,
                maximum_time=float(config["modal_horizon_s"]),
            )
            projection_rows.extend(role_projection)
            coherency_rows.extend(role_coherency)

    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir()
    comparison_models, design_rows = _fit_comparison_models(samples, config, checkpoint_dir)
    parameter_rows = _parameter_consistency(comparison_models)
    horizons = tuple(float(value) for value in config["rollout_horizons_s"])
    comparison_rows = _comparison_rollouts(comparison_models, samples, partitions, horizons)
    gmd_rows, modal_rows = _gmd_rollout_audit(
        frozen_gmd, samples["clustered_7_current"]["val"], horizons
    )
    residual_rows = _residual_structure_audit(
        frozen_gmd, samples["clustered_7_current"]["val"], float(config["modal_horizon_s"])
    )

    claims = _claim_assessment(
        config,
        partitions,
        projection_rows,
        coherency_rows,
        design_rows,
        comparison_rows,
        gmd_rows,
        modal_rows,
        residual_rows,
        parameter_rows,
    )
    _write_csv(result_dir / "dataset_support.csv", dataset_rows)
    _write_csv(result_dir / "projection_metrics.csv", projection_rows)
    _write_csv(result_dir / "coherency_metrics.csv", coherency_rows)
    _write_csv(result_dir / "design_conditioning.csv", design_rows)
    _write_csv(result_dir / "parameter_consistency.csv", parameter_rows)
    _write_csv(result_dir / "train_partition_pair_scores.csv", pair_score_rows)
    _write_csv(result_dir / "representation_rollout_metrics.csv", comparison_rows)
    _write_csv(result_dir / "gmd_rollout_metrics.csv", gmd_rows)
    _write_csv(result_dir / "gmd_modal_metrics.csv", modal_rows)
    _write_csv(result_dir / "residual_structure_metrics.csv", residual_rows)
    _json(result_dir / "projection_audit.json", projection_audit)
    _json(
        result_dir / "partition_definitions.json",
        {name: [list(group) for group in partition] for name, partition in partitions.items()},
    )
    _json(result_dir / "gmd_structural_physics_audit.json", gmd_physics_audit)
    _json(result_dir / "claim_assessment.json", claims)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "dataset_revision": store.dataset_metadata.dataset_revision,
            "dataset_build_id": store.dataset_metadata.build_id,
            "fold": config["fold"],
            "seed": config["seed"],
            "train_ids_used": train_ids,
            "val_ids_used": val_ids,
            "test_ids_read": [],
            "labels_used_for_fit_selection_or_thresholds": False,
            "observed_Pm_is_conditional_noncausal_upper_bound": True,
            "frozen_gmd_checkpoint": checkpoint.relative_to(root).as_posix(),
            "frozen_gmd_checkpoint_sha256": _sha256(checkpoint),
            "frozen_selection_marker": selection_marker,
            "config_sha256": _sha256(config_path),
            "partition_discovery_used_train_only": True,
            "partition_lineage_amendment": "legacy higher-order and current canonical seven-node definitions compared explicitly",
        },
    )

    _make_figures(
        result_dir, gmd_rows, projection_rows, coherency_rows, design_rows, comparison_rows
    )
    _write_report(
        result_dir,
        run_id=run_id,
        claims=claims,
        partitions=partitions,
        projection_audit=projection_audit,
        dataset_rows=dataset_rows,
        projection_rows=projection_rows,
        coherency_rows=coherency_rows,
        design_rows=design_rows,
        comparison_rows=comparison_rows,
        gmd_rows=gmd_rows,
        residual_rows=residual_rows,
        parameter_rows=parameter_rows,
    )

    process_gates = {
        "test_data_unopened": True,
        "labels_unused_for_fit_selection_or_thresholds": True,
        "frozen_checkpoint_selection_complete": True,
        "projection_identities_pass": bool(
            max(
                max(
                    row["left_inverse_error"],
                    row["idempotence_error"],
                    row["mass_self_adjoint_error"],
                )
                for row in projection_audit.values()
            )
            < 1e-12
        ),
        "canonical_current_adapter_reproduced": current_adapter_identity_error < 1e-12,
        "frozen_gmd_structural_audit_pass": gmd_physics_audit.get("status") == "pass",
        "comparison_models_completed": len(comparison_models) == 16,
        "three_partition_lineages_compared": len(partitions) == 3,
        "source_specific_parameter_audit_present": bool(parameter_rows),
        "projection_rows_present": bool(projection_rows),
        "coherency_rows_present": bool(coherency_rows),
        "gmd_rollout_rows_present": bool(gmd_rows),
        "residual_rows_present": bool(residual_rows),
        "scientific_result_may_be_negative": True,
    }
    process_acceptance = {
        "status": "PASS"
        if all(
            value
            for key, value in process_gates.items()
            if key != "scientific_result_may_be_negative"
        )
        else "FAIL",
        "gates": process_gates,
        "scientific_claims": claims,
    }
    _json(result_dir / "diagnostic_process_acceptance.json", process_acceptance)
    if process_acceptance["status"] != "PASS":
        raise RuntimeError("physical-cluster diagnostic process gate failed")

    artifacts = {
        path.relative_to(result_dir).as_posix(): _sha256(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)
