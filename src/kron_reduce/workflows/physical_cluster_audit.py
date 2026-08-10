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
    project_cluster_consensus,
    reconstruct_cluster_state,
)
from kron_reduce.physics.coordinates import coi_project_power

HIGHORDER_SOURCE = "ieee39_highorder_dataset"
DENSE_SOURCE = "ieee39_near_critical_unstable_v1_dense"
FIXED_PAIRS = ((0, 2), (6, 7), (8, 9))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        [np.interp(uniform_time, time_grid, values[:, channel]) for channel in range(values.shape[1])]
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
    return float(frequency[min(int(np.searchsorted(np.cumsum(normalized), quantile)), len(frequency) - 1)])


def _high_frequency_share(frequency: np.ndarray, power: np.ndarray, cutoff_hz: float = 1.0) -> float:
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
        pairs = tuple(sorted(tuple(sorted((int(permutation[2 * j]), int(permutation[2 * j + 1])))) for j in range(3)))
        output.add(pairs)
    return sorted(output)


def _design_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    column_scale = np.sqrt(np.mean(matrix * matrix, axis=0))
    active = column_scale > np.finfo(float).eps
    normalized = matrix[:, active] / column_scale[active]
    singular = np.linalg.svd(normalized, compute_uv=False)
    column_norm = np.linalg.norm(normalized, axis=0)
    gram = normalized.T @ normalized
    correlation = gram / np.maximum(column_norm[:, None] * column_norm[None, :], np.finfo(float).eps)
    np.fill_diagonal(correlation, 0.0)
    return {
        "n_rows": int(matrix.shape[0]),
        "n_parameters": int(matrix.shape[1]),
        "active_columns": int(np.sum(active)),
        "effective_rank": int(np.linalg.matrix_rank(normalized)),
        "condition_number": float(singular[0] / max(singular[-1], np.finfo(float).eps)),
        "stable_rank": float(np.sum(singular * singular) / max(singular[0] ** 2, np.finfo(float).eps)),
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
            {"ridge": float(ridge), "theta": theta.copy(), "solver": solver, "validation": validation}
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
    truth_decay = float(np.sqrt(np.mean(truth_omega[-quarter:] ** 2)) / max(np.sqrt(np.mean(truth_omega[:quarter] ** 2)), 1e-12))
    pred_decay = float(np.sqrt(np.mean(predicted_omega[-quarter:] ** 2)) / max(np.sqrt(np.mean(predicted_omega[:quarter] ** 2)), 1e-12))
    return {
        "rmse": float(np.sqrt(np.mean(np.concatenate(((predicted_delta - truth_delta).ravel(), omega_error.ravel())) ** 2))),
        "omega_rmse": float(np.sqrt(np.mean(omega_error ** 2))),
        "edge_angle_rmse": float(np.sqrt(np.mean(edge_error ** 2))),
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
        nonlinear = [index for index, name in enumerate(model.feature_names) if name not in {"delta_sin", "edge_speed"}]
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


def _projection_and_coherency_audit(
    samples: list[dict[str, Any]],
    *,
    role: str,
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
            ("omega_increment", omega - omega[[0]]),
            ("delta_increment", delta - delta[[0]]),
            ("delta_absolute", delta),
        ):
            low = project_cluster_consensus(values, mass, CLUSTERED_7_PARTITION)
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
                    "event_id": sample["event_id"],
                    "source_dataset": source,
                    "signal": signal_name,
                    "retained_mass_energy_fraction": retained,
                    "discarded_mass_energy_fraction": _weighted_energy(high, mass)
                    / max(total_energy, np.finfo(float).eps),
                    "low_component_median_frequency_hz": _frequency_quantile(low_frequency, low_power, 0.5),
                    "high_component_median_frequency_hz": _frequency_quantile(high_frequency, high_power, 0.5),
                    "low_component_above_1hz_share": _high_frequency_share(low_frequency, low_power),
                    "high_component_above_1hz_share": _high_frequency_share(high_frequency, high_power),
                }
            )

        omega_increment = omega - omega[[0]]
        correlation_matrix = np.eye(10)
        coherence_matrix = np.eye(10)
        relative_rms = np.zeros((10, 10))
        global_scale = max(float(np.sqrt(np.mean(omega_increment**2))), 1e-12)
        for left in range(10):
            for right in range(left + 1, 10):
                correlation_matrix[left, right] = correlation_matrix[right, left] = _safe_correlation(
                    omega_increment[:, left], omega_increment[:, right]
                )
                coherence_matrix[left, right] = coherence_matrix[right, left] = _band_coherence(
                    omega_increment[:, left], omega_increment[:, right]
                )
                relative_rms[left, right] = relative_rms[right, left] = float(
                    np.sqrt(np.mean((omega_increment[:, left] - omega_increment[:, right]) ** 2))
                    / global_scale
                )
        fixed_correlation = float(np.mean([correlation_matrix[left, right] for left, right in FIXED_PAIRS]))
        fixed_coherence = float(np.mean([coherence_matrix[left, right] for left, right in FIXED_PAIRS]))
        fixed_relative_rms = float(np.mean([relative_rms[left, right] for left, right in FIXED_PAIRS]))
        random_correlation = np.asarray(
            [np.mean([correlation_matrix[left, right] for left, right in pair_set]) for pair_set in random_pair_sets]
        )
        random_coherence = np.asarray(
            [np.mean([coherence_matrix[left, right] for left, right in pair_set]) for pair_set in random_pair_sets]
        )
        random_relative_rms = np.asarray(
            [np.mean([relative_rms[left, right] for left, right in pair_set]) for pair_set in random_pair_sets]
        )
        coherency_rows.append(
            {
                "role": role,
                "event_id": sample["event_id"],
                "source_dataset": source,
                "fixed_pair_mean_correlation": fixed_correlation,
                "fixed_pair_mean_band_coherence": fixed_coherence,
                "fixed_pair_relative_rms": fixed_relative_rms,
                "correlation_random_partition_percentile": float(np.mean(random_correlation <= fixed_correlation)),
                "coherence_random_partition_percentile": float(np.mean(random_coherence <= fixed_coherence)),
                "relative_rms_random_partition_percentile": float(np.mean(random_relative_rms >= fixed_relative_rms)),
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
    for cohort in ("combined", "highorder"):
        for representation in ("generator_10", "clustered_7"):
            train_all = samples[representation]["train"]
            val_all = samples[representation]["val"]
            train = train_all if cohort == "combined" else [row for row in train_all if _source(row) == HIGHORDER_SOURCE]
            val = val_all if cohort == "combined" else [row for row in val_all if _source(row) == HIGHORDER_SOURCE]
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
                alpha_candidates=tuple(float(value) for value in config["third_order_alpha_candidates"]),
                tau_candidates=tuple(float(value) for value in config["third_order_tau_candidates"]),
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
                    "validation_mean_rmse": third_fit.details.get("validation_raw_mean_rmse", third_fit.best_validation_loss),
                    "minimum_damping": float(np.min(third_order.damping)),
                    "median_damping": float(np.median(third_order.damping)),
                    "maximum_damping": float(np.max(third_order.damping)),
                }
            )
    return models, design_rows


def _comparison_rollouts(
    models: dict[tuple[str, str, str], Any],
    samples: dict[str, dict[str, list[dict[str, Any]]]],
    horizons_s: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val10 = {sample["event_id"]: sample for sample in samples["generator_10"]["val"]}
    val7 = {sample["event_id"]: sample for sample in samples["clustered_7"]["val"]}
    for (cohort, representation, method), model in models.items():
        representation_samples = samples[representation]["val"]
        if cohort == "highorder":
            representation_samples = [sample for sample in representation_samples if _source(sample) == HIGHORDER_SOURCE]
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
                row = {
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
                    truth7 = val7[sample["event_id"]]["training_context"]["targets"]
                    mass10 = np.asarray(static["mass"])
                    delta7 = aggregate_state(predicted_delta[:steps], mass10, CLUSTERED_7_PARTITION)
                    omega7 = aggregate_state(predicted_omega[:steps], mass10, CLUSTERED_7_PARTITION)
                    interarea = _trajectory_errors(
                        np.asarray(truth7["delta"][:steps]),
                        np.asarray(truth7["omega"][:steps]),
                        delta7,
                        omega7,
                        np.asarray(val7[sample["event_id"]]["model_inputs"]["static"]["incidence"]),
                        np.asarray(truth7["t"][:steps]),
                    )
                    row.update({f"interarea7_{key}": value for key, value in interarea.items()})
                else:
                    truth10 = val10[sample["event_id"]]["training_context"]["targets"]
                    static10 = val10[sample["event_id"]]["model_inputs"]["static"]
                    delta10 = reconstruct_cluster_state(predicted_delta[:steps], static10["mass"], CLUSTERED_7_PARTITION)
                    omega10 = reconstruct_cluster_state(predicted_omega[:steps], static10["mass"], CLUSTERED_7_PARTITION)
                    reconstructed = _trajectory_errors(
                        np.asarray(truth10["delta"][:steps]),
                        np.asarray(truth10["omega"][:steps]),
                        delta10,
                        omega10,
                        np.asarray(static10["incidence"]),
                        np.asarray(truth10["t"][:steps]),
                    )
                    row.update({f"reconstructed10_{key}": value for key, value in reconstructed.items()})
                rows.append(row)
    return rows


def _gmd_rollout_audit(
    model: GraphModalDiscoveryM0Model,
    validation_samples: list[dict[str, Any]],
    horizons_s: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory_rows: list[dict[str, Any]] = []
    modal_rows: list[dict[str, Any]] = []
    variants = {name: _gmd_variant(model, name) for name in ("backbone", "static_only", "memory_only", "linear_only", "full")}
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
                    np.asarray(target["t"][:steps]), np.asarray(target["omega"][:steps]), maximum_time=horizon_s
                )
                _, predicted_omega_uniform = _uniform_path(
                    np.asarray(target["t"][:steps]), predicted_omega[:steps], maximum_time=horizon_s
                )
                frequency, truth_power = spectral_signature(
                    truth_omega, 0.01, channel_weights=np.asarray(static["mass"]), max_frequency_hz=5.0
                )
                _, predicted_power = spectral_signature(
                    predicted_omega_uniform, 0.01, channel_weights=np.asarray(static["mass"]), max_frequency_hz=5.0
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
                        "predicted_dominant_frequency_hz": dominant_frequency(frequency, predicted_power),
                        "dominant_frequency_error_hz": abs(
                            dominant_frequency(frequency, predicted_power) - dominant_frequency(frequency, truth_power)
                        ),
                        **errors,
                    }
                )
            if maximum_steps < 501:
                continue
            _, truth_omega = _uniform_path(
                np.asarray(target["t"][:maximum_steps]), np.asarray(target["omega"][:maximum_steps]), maximum_time=max(horizons_s)
            )
            _, predicted_omega_uniform = _uniform_path(
                np.asarray(target["t"][:maximum_steps]), predicted_omega[:maximum_steps], maximum_time=max(horizons_s)
            )
            truth_modes = exact_dmd_modes(truth_omega, 0.01, rank=6, max_frequency_hz=3.0)
            predicted_modes = exact_dmd_modes(predicted_omega_uniform, 0.01, rank=6, max_frequency_hz=3.0)
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
            for rank, match in enumerate(sorted(matches, key=lambda row: row["reference_strength"], reverse=True), start=1):
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
    drivers = np.stack([model._modal_drivers(delta[index], omega[index], static) for index in range(len(time_grid))])
    z = model._first_order_hold_memory(drivers, time_grid, taus)

    def force_for(feature_indices: list[int], *, include_static: bool, include_memory: bool) -> np.ndarray:
        amplitude = np.zeros((len(time_grid), basis.shape[1]))
        if include_static:
            amplitude += np.sum(
                static_coefficients[None, feature_indices, :] * drivers[:, feature_indices, :], axis=1
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
                    "force_cosine_alignment": float(np.sum(true_residual * predicted) / denominator) if denominator > 0 else 0.0,
                    "residual_explained_energy": float(
                        1.0 - np.sum((true_residual - predicted) ** 2) / max(np.sum(true_residual**2), np.finfo(float).eps)
                    ),
                    "predicted_to_true_force_rms_ratio": float(
                        np.sqrt(np.mean(predicted**2)) / max(np.sqrt(np.mean(true_residual**2)), np.finfo(float).eps)
                    ),
                    "residual_power_correlation": power_correlation,
                    "true_cumulative_residual_work": true_work,
                    "predicted_cumulative_residual_work": predicted_work,
                    "normalized_cumulative_work_error": abs(predicted_work - true_work) / max(abs(true_work), np.sqrt(np.mean(true_power**2)) * float(time_grid[-1]), 1e-12),
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
    projection_rows: list[dict[str, Any]],
    coherency_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    gmd_rows: list[dict[str, Any]],
    modal_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
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
    residual_alignment = _group_median(
        residual_rows, "force_cosine_alignment", component="full"
    )
    residual_explained = _group_median(
        residual_rows, "residual_explained_energy", component="full"
    )
    gmd_failures = sum(row.get("variant") == "full" and row.get("status") == "failed" for row in gmd_rows)
    oscillation_pass = bool(
        spectral_overlap_value >= float(config["minimum_spectral_overlap"])
        and dominant_error <= float(config["maximum_dominant_frequency_error_hz"])
        and (not np.isfinite(modal_mac) or modal_mac >= float(config["minimum_modal_assurance_criterion"]))
        and gmd_failures == 0
    )
    residual_pass = bool(residual_alignment > 0.0 and residual_explained > 0.0)

    energy_omega = _group_median(
        projection_rows,
        "retained_mass_energy_fraction",
        role="val",
        signal="omega_increment",
    )
    energy_delta = _group_median(
        projection_rows,
        "retained_mass_energy_fraction",
        role="val",
        signal="delta_increment",
    )
    partition_percentile = _group_median(
        coherency_rows,
        "correlation_random_partition_percentile",
        role="val",
    )
    low_median_frequency = _group_median(
        projection_rows,
        "low_component_median_frequency_hz",
        role="val",
        signal="omega_increment",
    )
    high_median_frequency = _group_median(
        projection_rows,
        "high_component_median_frequency_hz",
        role="val",
        signal="omega_increment",
    )
    low_above_1hz = _group_median(
        projection_rows,
        "low_component_above_1hz_share",
        role="val",
        signal="omega_increment",
    )
    high_above_1hz = _group_median(
        projection_rows,
        "high_component_above_1hz_share",
        role="val",
        signal="omega_increment",
    )
    design_index = {
        (row["cohort"], row["representation"], row["method"]): row for row in design_rows
    }
    conditioning_pass = all(
        design_index[(cohort, "clustered_7", "M0")]["condition_number"]
        < design_index[(cohort, "generator_10", "M0")]["condition_number"]
        for cohort in ("combined", "highorder")
    )
    highorder_m03e_7 = _group_mean(
        comparison_rows,
        "native_rmse",
        cohort="highorder",
        representation="clustered_7",
        method="M0-3E",
        horizon_s=1.0,
        status="completed",
    )
    highorder_m03e_10_interarea = _group_mean(
        comparison_rows,
        "interarea7_rmse",
        cohort="highorder",
        representation="generator_10",
        method="M0-3E",
        horizon_s=1.0,
        status="completed",
    )
    predictive_pass = bool(
        np.isfinite(highorder_m03e_7)
        and np.isfinite(highorder_m03e_10_interarea)
        and highorder_m03e_7 <= highorder_m03e_10_interarea
    )
    projection_pass = bool(
        energy_omega >= float(config["minimum_dynamic_energy_retention"])
        and energy_delta >= float(config["minimum_dynamic_energy_retention"])
    )
    coherency_pass = bool(partition_percentile >= float(config["minimum_random_partition_percentile"]))
    empirical_lowpass_pass = bool(high_median_frequency > low_median_frequency and high_above_1hz > low_above_1hz)

    physical_status = "SUPPORTED" if oscillation_pass and residual_pass else (
        "CONDITIONAL" if oscillation_pass or residual_pass else "REJECTED"
    )
    cluster_status = "SUPPORTED" if all(
        (projection_pass, coherency_pass, empirical_lowpass_pass, conditioning_pass, predictive_pass)
    ) else ("CONDITIONAL" if sum((projection_pass, coherency_pass, empirical_lowpass_pass, conditioning_pass, predictive_pass)) >= 3 else "REJECTED")
    return {
        "current_equation_has_effective_physical_meaning": {
            "status": physical_status,
            "scope": "effective reduced response under observed Pm; not machine-parameter or topology recovery",
            "oscillation_preservation_gate": oscillation_pass,
            "residual_structure_gate": residual_pass,
            "median_5s_spectral_overlap": spectral_overlap_value,
            "median_5s_dominant_frequency_error_hz": dominant_error,
            "median_primary_mode_MAC": modal_mac,
            "median_force_cosine_alignment": residual_alignment,
            "median_residual_explained_energy": residual_explained,
            "full_rollout_failures": gmd_failures,
        },
        "seven_node_frontend_supported": {
            "status": cluster_status,
            "scope": "mass-orthogonal cluster-consensus projection and empirical slow-coherency frontend",
            "projection_energy_gate": projection_pass,
            "fixed_partition_coherency_gate": coherency_pass,
            "empirical_temporal_lowpass_gate": empirical_lowpass_pass,
            "conditioning_gate": conditioning_pass,
            "higher_order_predictive_gate": predictive_pass,
            "validation_omega_dynamic_energy_retention": energy_omega,
            "validation_delta_dynamic_energy_retention": energy_delta,
            "validation_fixed_partition_correlation_percentile": partition_percentile,
            "low_component_median_frequency_hz": low_median_frequency,
            "high_component_median_frequency_hz": high_median_frequency,
            "low_component_above_1hz_share": low_above_1hz,
            "high_component_above_1hz_share": high_above_1hz,
            "highorder_M0_3E_native_7node_mean_rmse": highorder_m03e_7,
            "highorder_M0_3E_10node_projected_to_7_mean_rmse": highorder_m03e_10_interarea,
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
    colors = {"backbone": "#777777", "static_only": "#56B4E9", "memory_only": "#E69F00", "linear_only": "#009E73", "full": "#D55E00"}
    variants = list(colors)
    horizons = (1.0, 5.0, 10.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    ax = axes[0, 0]
    for variant in variants:
        means = [_group_mean(gmd_rows, "rmse", variant=variant, horizon_s=horizon, status="completed") for horizon in horizons]
        ax.plot(horizons, means, marker="o", markersize=3.5, linewidth=1.2, label=variant.replace("_", " "), color=colors[variant])
    ax.set(xlabel="Reintegration horizon (s)", ylabel="Mean trajectory RMSE", title="Closed-loop accuracy")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(-0.16, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[0, 1]
    x = np.arange(len(variants))
    overlap = [_group_median(gmd_rows, "spectral_overlap", variant=variant, horizon_s=5.0, status="completed") for variant in variants]
    ax.bar(x, overlap, color=[colors[value] for value in variants], width=0.72)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [value.replace("_", " ") for value in variants], rotation=25, ha="right")
    ax.set(ylabel="Median spectral overlap", ylim=(0, 1.05), title="Oscillation-content preservation")
    ax.text(-0.16, 1.05, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1, 0]
    energy = [
        float(row["retained_mass_energy_fraction"])
        for row in projection_rows
        if row["role"] == "val" and row["signal"] in {"omega_increment", "delta_increment"}
    ]
    percentile = [float(row["correlation_random_partition_percentile"]) for row in coherency_rows if row["role"] == "val"]
    box = ax.boxplot([energy, percentile], labels=["Dynamic energy\nretained", "Fixed-pair random\npercentile"], patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(box["boxes"], ("#56B4E9", "#009E73"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    ax.set(ylabel="Fraction / percentile", ylim=(0, 1.05), title="Seven-node evidence")
    ax.text(-0.16, 1.05, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1, 1]
    design = [row for row in design_rows if row["method"] == "M0"]
    for representation, marker, color in (("generator_10", "o", "#777777"), ("clustered_7", "s", "#D55E00")):
        selected = [row for row in design if row["representation"] == representation]
        ax.scatter(
            [row["n_parameters"] for row in selected],
            [row["condition_number"] for row in selected],
            marker=marker,
            color=color,
            s=32,
            label=representation.replace("_", " "),
        )
        for row in selected:
            ax.annotate(row["cohort"], (row["n_parameters"], row["condition_number"]), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    ax.set_yscale("log")
    ax.set(xlabel="Weak-form parameter count", ylabel="Normalized condition number", title="Complexity and conditioning")
    ax.legend(frameon=False, fontsize=7)
    ax.text(-0.16, 1.05, "d", transform=ax.transAxes, fontweight="bold", fontsize=10)

    outputs: list[Path] = []
    for suffix, kwargs in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        path = figure_dir / f"fig1_physical_cluster_audit.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    methods = ("M0", "M0-3E")
    values7 = [_group_mean(comparison_rows, "native_rmse", cohort="highorder", representation="clustered_7", method=method, horizon_s=1.0, status="completed") for method in methods]
    values10 = [_group_mean(comparison_rows, "interarea7_rmse", cohort="highorder", representation="generator_10", method=method, horizon_s=1.0, status="completed") for method in methods]
    x = np.arange(len(methods))
    ax.bar(x - 0.18, values10, 0.36, color="#777777", label="10-node model → 7-node metric")
    ax.bar(x + 0.18, values7, 0.36, color="#D55E00", label="native 7-node model")
    ax.set_xticks(x, methods)
    ax.set(ylabel="High-order validation mean RMSE", title="Matched inter-area prediction (1 s)")
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


def _write_report(
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
                "backbone_rmse": _group_mean(gmd_rows, "rmse", variant="backbone", horizon_s=horizon, status="completed"),
                "full_rmse": _group_mean(gmd_rows, "rmse", variant="full", horizon_s=horizon, status="completed"),
                "full_overlap": _group_median(gmd_rows, "spectral_overlap", variant="full", horizon_s=horizon, status="completed"),
                "full_frequency_error": _group_median(gmd_rows, "dominant_frequency_error_hz", variant="full", horizon_s=horizon, status="completed"),
            }
        )
    residual_components = sorted(
        {row["component"] for row in residual_rows},
        key=lambda component: -_group_median(residual_rows, "predicted_to_true_force_rms_ratio", component=component),
    )
    source_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in dataset_rows:
        source_counts[(str(row["role"]), str(row["source_dataset"]))] += 1
    design_index = {(row["cohort"], row["representation"], row["method"]): row for row in design_rows}

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
<div class="status"><div class="card"><b>M0-GMD 物理意义</b><br>{html.escape(str(physical['status']))}</div><div class="card"><b>七节点前端</b><br>{html.escape(str(cluster['status']))}</div></div>
<img src="figures/fig1_physical_cluster_audit.svg" alt="物理意义与聚合审计主图">
<img src="figures/fig2_highorder_cluster_comparison.svg" alt="高阶模型七节点对照">
<pre>{body}</pre>
</main></body></html>"""
    (result_dir / "report.html").write_text(document, encoding="utf-8")


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    if stage != "diagnostic":
        raise RuntimeError("physical-cluster audit implements only the diagnostic stage")
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
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
    train_ids = [event_id for event_id in split.train_ids if store.metadata(event_id).valid_length >= minimum_steps]
    val_ids = [event_id for event_id in split.val_ids if store.metadata(event_id).valid_length >= minimum_steps]
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
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for representation in ("generator_10", "clustered_7"):
        samples[representation] = {}
        for role, ids in (("train", train_ids), ("val", val_ids)):
            adapted = [
                adapt_event(
                    store.load_event(event_id),
                    inertia_prior=inertia,
                    request=request,
                    representation=representation,
                )
                for event_id in ids
            ]
            for sample in adapted:
                audit_model_sample(sample)
            samples[representation][role] = adapted

    checkpoint = root / str(config["frozen_gmd_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
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
    aggregate, reconstruct, projection = cluster_aggregation_matrices(mass10, CLUSTERED_7_PARTITION)
    mass_matrix = np.diag(mass10)
    projection_audit = {
        "left_inverse_error": float(np.max(np.abs(aggregate @ reconstruct - np.eye(7)))),
        "idempotence_error": float(np.max(np.abs(projection @ projection - projection))),
        "mass_self_adjoint_error": float(np.max(np.abs(projection.T @ mass_matrix - mass_matrix @ projection))),
        "rank": int(np.linalg.matrix_rank(projection)),
        "mass_orthogonal_projection": True,
    }

    random_pair_sets = _random_pair_sets(int(config["random_partition_draws"]), int(config["seed"]))
    projection_rows: list[dict[str, Any]] = []
    coherency_rows: list[dict[str, Any]] = []
    for role in ("train", "val"):
        role_projection, role_coherency = _projection_and_coherency_audit(
            samples["generator_10"][role],
            role=role,
            random_pair_sets=random_pair_sets,
            maximum_time=float(config["modal_horizon_s"]),
        )
        projection_rows.extend(role_projection)
        coherency_rows.extend(role_coherency)

    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir()
    comparison_models, design_rows = _fit_comparison_models(samples, config, checkpoint_dir)
    horizons = tuple(float(value) for value in config["rollout_horizons_s"])
    comparison_rows = _comparison_rollouts(comparison_models, samples, horizons)
    gmd_rows, modal_rows = _gmd_rollout_audit(frozen_gmd, samples["clustered_7"]["val"], horizons)
    residual_rows = _residual_structure_audit(
        frozen_gmd, samples["clustered_7"]["val"], float(config["modal_horizon_s"])
    )

    claims = _claim_assessment(
        config,
        projection_rows,
        coherency_rows,
        design_rows,
        comparison_rows,
        gmd_rows,
        modal_rows,
        residual_rows,
    )
    _write_csv(result_dir / "dataset_support.csv", dataset_rows)
    _write_csv(result_dir / "projection_metrics.csv", projection_rows)
    _write_csv(result_dir / "coherency_metrics.csv", coherency_rows)
    _write_csv(result_dir / "design_conditioning.csv", design_rows)
    _write_csv(result_dir / "representation_rollout_metrics.csv", comparison_rows)
    _write_csv(result_dir / "gmd_rollout_metrics.csv", gmd_rows)
    _write_csv(result_dir / "gmd_modal_metrics.csv", modal_rows)
    _write_csv(result_dir / "residual_structure_metrics.csv", residual_rows)
    _json(result_dir / "projection_audit.json", projection_audit)
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
        },
    )

    _make_figures(
        result_dir, gmd_rows, projection_rows, coherency_rows, design_rows, comparison_rows
    )
    _write_report(
        result_dir,
        run_id=run_id,
        claims=claims,
        projection_audit=projection_audit,
        dataset_rows=dataset_rows,
        design_rows=design_rows,
        comparison_rows=comparison_rows,
        gmd_rows=gmd_rows,
        residual_rows=residual_rows,
    )

    process_gates = {
        "test_data_unopened": True,
        "labels_unused_for_fit_selection_or_thresholds": True,
        "frozen_checkpoint_selection_complete": True,
        "projection_identities_pass": bool(
            max(
                projection_audit["left_inverse_error"],
                projection_audit["idempotence_error"],
                projection_audit["mass_self_adjoint_error"],
            )
            < 1e-12
        ),
        "frozen_gmd_structural_audit_pass": gmd_physics_audit.get("status") == "pass",
        "comparison_models_completed": len(comparison_models) == 6,
        "projection_rows_present": bool(projection_rows),
        "coherency_rows_present": bool(coherency_rows),
        "gmd_rollout_rows_present": bool(gmd_rows),
        "residual_rows_present": bool(residual_rows),
        "scientific_result_may_be_negative": True,
    }
    process_acceptance = {
        "status": "PASS" if all(value for key, value in process_gates.items() if key != "scientific_result_may_be_negative") else "FAIL",
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
