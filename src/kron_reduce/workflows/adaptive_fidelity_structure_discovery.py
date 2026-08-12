from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data import (
    CanonicalStore,
    ModelDataRequest,
    adapt_event,
    cumulative_cohorts,
    fit_stratification_policy,
    load_split,
    stratify_samples,
)
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation import (
    classify_predicted_trajectory,
    dominant_frequency,
    exact_dmd_modes,
    match_dmd_modes,
    spectral_overlap,
    spectral_signature,
    trajectory_metrics,
)
from kron_reduce.evaluation.metrics import aggregate_trajectory_metrics
from kron_reduce.models import (
    GraphModalDiscoveryM0Model,
    M0Model,
    PortStructuredDiscoveryM0Model,
    RunContext,
    TransientEmfM0Model,
)
from kron_reduce.physics import (
    CLUSTERED_7_PARTITION,
    aggregate_model_sample,
    discover_nested_partitions,
    reconstruct_model_state,
    spatial_projection_audit,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for original in materialized:
            row = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in original.items()
            }
            writer.writerow(row)


def _source(sample: dict[str, Any]) -> str:
    return str(sample["training_context"]["metadata"]["source_dataset"])


def _stress_thresholds(samples: list[dict[str, Any]]) -> tuple[float, ...]:
    values = []
    for sample in samples:
        static = sample["model_inputs"]["static"]
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        delta0 = np.asarray(static["delta0"], dtype=np.float64).reshape(-1)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64)
        values.append(
            float(
                np.max(
                    np.abs(
                        np.sin(incidence.T @ delta0)
                        - np.sin(incidence.T @ equilibrium)
                    )
                )
            )
        )
    quantiles = np.quantile(values, (0.25, 0.5, 0.75))
    return tuple(sorted({0.0, *(float(value) for value in quantiles)}))


def _context(config: dict[str, Any], train: list[dict[str, Any]]) -> RunContext:
    return RunContext(
        seed=int(config["seed"]),
        epochs=0,
        horizon_steps=int(config["selection_horizon_steps"]),
        max_cases=len(train),
        learning_rate=0.0,
        ridge_values=tuple(float(value) for value in config["ridge_values"]),
        integral_windows=tuple(int(value) for value in config["integral_windows"]),
        max_internal_step=float(config["max_internal_step"]),
        fit_max_steps=int(config["fit_max_steps"]),
        weak_max_windows_per_case_horizon=int(
            config["weak_max_windows_per_case_horizon"]
        ),
    )


def _representation_samples(
    raw: list[dict[str, Any]], partitions: dict[str, tuple[tuple[int, ...], ...] | None]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for name, partition in partitions.items():
        if partition is None:
            output[name] = raw
        else:
            output[name] = [aggregate_model_sample(sample, partition, name) for sample in raw]
        for sample in output[name]:
            audit_model_sample(sample)
    return output


def _subset(samples: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample["event_id"] in ids]


def _rollout_row(
    model: Any,
    representation_sample: dict[str, Any],
    raw_sample: dict[str, Any],
    partition: tuple[tuple[int, ...], ...] | None,
    horizon: int,
) -> dict[str, Any]:
    raw_target = raw_sample["training_context"]["targets"]
    target = representation_sample["training_context"]["targets"]
    steps = min(horizon, len(target["t"]), len(raw_target["t"]))
    base = {
        "event_id": raw_sample["event_id"],
        "source_dataset": _source(raw_sample),
        "fault_line": raw_sample["training_context"]["metadata"]["fault_line"],
        "horizon_steps": int(horizon),
        "n_available_steps": len(raw_target["t"]),
    }
    if steps < horizon:
        return {**base, "status": "ineligible_short_horizon"}
    static = representation_sample["model_inputs"]["static"]
    exogenous = representation_sample["model_inputs"]["exogenous"]
    try:
        delta, omega = model.rollout(
            static,
            np.asarray(target["t"][:steps]),
            {"Pm": np.asarray(exogenous["Pm"][:steps]), "interpolation": "linear"},
        )
        raw_mass = np.asarray(raw_sample["model_inputs"]["static"]["mass"])
        delta = reconstruct_model_state(delta, raw_mass, partition)
        omega = reconstruct_model_state(omega, raw_mass, partition)
        truth_delta = np.asarray(raw_target["delta"][:steps])
        truth_omega = np.asarray(raw_target["omega"][:steps])
        metrics = trajectory_metrics(truth_delta, truth_omega, delta, omega)
        raw_incidence = np.asarray(raw_sample["model_inputs"]["static"]["incidence"])
        edge_rmse = float(
            np.sqrt(np.mean(((delta - truth_delta) @ raw_incidence) ** 2))
        )
        dt = float(np.median(np.diff(np.asarray(raw_target["t"][:steps]))))
        frequency, truth_power = spectral_signature(
            truth_omega, dt, channel_weights=raw_mass, max_frequency_hz=5.0
        )
        _, predicted_power = spectral_signature(
            omega, dt, channel_weights=raw_mass, max_frequency_hz=5.0
        )
        overlap = spectral_overlap(truth_power, predicted_power)
        truth_dominant = dominant_frequency(frequency, truth_power)
        predicted_dominant = dominant_frequency(frequency, predicted_power)
        modal: dict[str, float] = {
            "modal_frequency_relative_error": float("nan"),
            "modal_damping_absolute_error": float("nan"),
            "modal_mac": float("nan"),
        }
        if steps >= 500:
            truth_modes = exact_dmd_modes(truth_omega, dt, rank=min(6, truth_omega.shape[1]))
            predicted_modes = exact_dmd_modes(omega, dt, rank=min(6, omega.shape[1]))
            matches = match_dmd_modes(truth_modes, predicted_modes)
            if matches:
                strongest = max(matches, key=lambda row: row["reference_strength"])
                modal = {
                    "modal_frequency_relative_error": strongest["relative_frequency_error"],
                    "modal_damping_absolute_error": strongest["absolute_damping_ratio_error"],
                    "modal_mac": strongest["mac"],
                }
        convergence = classify_predicted_trajectory(delta, omega)
        return {
            **base,
            "status": "completed",
            **metrics,
            "edge_angle_rmse": edge_rmse,
            "spectral_overlap": overlap,
            "dominant_frequency_truth_hz": truth_dominant,
            "dominant_frequency_predicted_hz": predicted_dominant,
            "dominant_frequency_absolute_error_hz": abs(predicted_dominant - truth_dominant),
            "predicted_finite_time_stability": convergence,
            "maximum_coi_delta_drift": float(np.max(np.abs(delta @ raw_mass))),
            "maximum_coi_omega_drift": float(np.max(np.abs(omega @ raw_mass))),
            **modal,
        }
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
        return {**base, "status": "failed", "error": repr(error)}


def _evaluate_candidate(
    model: Any,
    representation: str,
    representation_samples: list[dict[str, Any]],
    raw_samples: list[dict[str, Any]],
    partition: tuple[tuple[int, ...], ...] | None,
    horizons: tuple[int, ...],
    *,
    cohort: str,
    branch: str,
) -> list[dict[str, Any]]:
    raw_by_id = {sample["event_id"]: sample for sample in raw_samples}
    rows = []
    for sample in representation_samples:
        for horizon in horizons:
            rows.append(
                {
                    "cohort": cohort,
                    "representation": representation,
                    "branch": branch,
                    **_rollout_row(
                        model, sample, raw_by_id[sample["event_id"]], partition, horizon
                    ),
                }
            )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted(
        {
            (
                str(row["cohort"]),
                str(row["representation"]),
                str(row["branch"]),
                int(row["horizon_steps"]),
            )
            for row in rows
        }
    )
    for cohort, representation, branch, horizon in keys:
        group = [
            row
            for row in rows
            if (
                row["cohort"],
                row["representation"],
                row["branch"],
                int(row["horizon_steps"]),
            )
            == (cohort, representation, branch, horizon)
        ]
        eligible = [row for row in group if row["status"] != "ineligible_short_horizon"]
        completed = [row for row in eligible if row["status"] == "completed"]
        metric_rows = [row for row in eligible if row["status"] == "completed"] + [
            {"rmse": float("nan")} for row in eligible if row["status"] == "failed"
        ]
        aggregate = aggregate_trajectory_metrics(metric_rows) if metric_rows else {
            "mean_rmse": float("nan"),
            "median_rmse": float("nan"),
            "pooled_rmse": float("nan"),
            "worst_10pct_cvar_rmse": float("nan"),
            "failure_rate": float("nan"),
        }

        def mean(key: str, completed_rows: list[dict[str, Any]] = completed) -> float:
            values = np.asarray(
                [row.get(key, np.nan) for row in completed_rows], dtype=float
            )
            values = values[np.isfinite(values)]
            return float(np.mean(values)) if len(values) else float("nan")

        def median(key: str, completed_rows: list[dict[str, Any]] = completed) -> float:
            values = np.asarray(
                [row.get(key, np.nan) for row in completed_rows], dtype=float
            )
            values = values[np.isfinite(values)]
            return float(np.median(values)) if len(values) else float("nan")

        output.append(
            {
                "cohort": cohort,
                "representation": representation,
                "branch": branch,
                "horizon_steps": horizon,
                "eligible_cases": len(eligible),
                "completed_cases": len(completed),
                **aggregate,
                "mean_edge_angle_rmse": mean("edge_angle_rmse"),
                "mean_spectral_overlap": mean("spectral_overlap"),
                "median_dominant_frequency_error_hz": median(
                    "dominant_frequency_absolute_error_hz"
                ),
                "median_modal_frequency_relative_error": median(
                    "modal_frequency_relative_error"
                ),
                "median_modal_damping_absolute_error": median(
                    "modal_damping_absolute_error"
                ),
                "median_modal_mac": median("modal_mac"),
                "maximum_coi_drift": max(
                    [
                        max(
                            float(row["maximum_coi_delta_drift"]),
                            float(row["maximum_coi_omega_drift"]),
                        )
                        for row in completed
                    ],
                    default=float("nan"),
                ),
            }
        )
    return output


def _complexity(branch: str, n_nodes: int, diagnostics: dict[str, Any]) -> int:
    if branch == "B2":
        return 2 * n_nodes
    if branch == "B3":
        return 3 * n_nodes
    if branch in {"BS", "B4", "BM-r"}:
        rank = int(diagnostics.get("selected_modal_rank", n_nodes - 1))
        memory = int(diagnostics.get("selected_memory_order", 0))
        return 3 * n_nodes + rank * max(memory, 1)
    if branch.startswith("PSD"):
        memory = len(diagnostics.get("selected_fit_taus", []))
        return 3 * n_nodes + int(diagnostics.get("selected_support", {}).get("active_groups", 0)) + memory
    return 1000 + n_nodes


def _branch_for_gmd(model: GraphModalDiscoveryM0Model) -> str:
    diagnostics = model.diagnostics
    if diagnostics.get("selected_has_dynamic_closure"):
        return "B4" if int(diagnostics.get("selected_memory_order", 0)) == 1 else "BM-r"
    return "BS"


def _select(
    aggregate: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    cohort: str,
    structure_train_cases: int,
) -> dict[str, Any]:
    minimum_cases = int(config["minimum_selection_cases"])
    if structure_train_cases < minimum_cases:
        return {
            "status": "INSUFFICIENT_INFORMATION",
            "reason": "fewer than the predeclared Q-A train cases",
            "structure_train_cases": structure_train_cases,
        }
    horizons = sorted(
        {
            int(row["horizon_steps"])
            for row in aggregate
            if row["cohort"] == cohort and int(row["eligible_cases"]) >= minimum_cases
        }
    )
    if not horizons:
        return {
            "status": "INSUFFICIENT_INFORMATION",
            "reason": "no validation horizon has enough eligible cases",
            "structure_train_cases": structure_train_cases,
        }
    horizon = max(horizons)
    rows = [
        row
        for row in aggregate
        if row["cohort"] == cohort and int(row["horizon_steps"]) == horizon
    ]
    metadata = {
        (row["representation"], row["branch"]): row for row in candidates if row["cohort"] == cohort
    }
    admissible = [
        row
        for row in rows
        if row["failure_rate"] == 0.0
        and np.isfinite(row["mean_rmse"])
        and metadata[(row["representation"], row["branch"])]["physics_status"] == "pass"
    ]
    if not admissible:
        return {
            "status": "INSUFFICIENT_INFORMATION",
            "reason": "no candidate passed finite-rollout and physics gates",
            "selection_horizon": horizon,
        }
    best_rmse = min(float(row["mean_rmse"]) for row in admissible)
    tolerance = float(config["minimum_model_improvement"])
    near_best = [
        row for row in admissible if row["mean_rmse"] <= best_rmse * (1.0 + tolerance) + 1e-15
    ]
    selected = min(
        near_best,
        key=lambda row: (
            metadata[(row["representation"], row["branch"])]["complexity"],
            row["mean_rmse"],
            -float(row.get("mean_spectral_overlap", 0.0)),
            row["representation"],
            row["branch"],
        ),
    )
    selected_meta = metadata[(selected["representation"], selected["branch"])]
    return {
        "status": "SELECTED",
        "selection_horizon": horizon,
        "minimum_validation_improvement": tolerance,
        "best_mean_rmse": best_rmse,
        "selected_representation": selected["representation"],
        "selected_branch": selected["branch"],
        "selected_mean_rmse": selected["mean_rmse"],
        "selected_spectral_overlap": selected["mean_spectral_overlap"],
        "selected_modal_frequency_error": selected[
            "median_modal_frequency_relative_error"
        ],
        "selected_complexity": selected_meta["complexity"],
        "selected_model_key": selected_meta["model_key"],
        "near_best_candidates": [
            {
                "representation": row["representation"],
                "branch": row["branch"],
                "mean_rmse": row["mean_rmse"],
                "complexity": metadata[(row["representation"], row["branch"])]["complexity"],
            }
            for row in near_best
        ],
        "interpretation": (
            "minimum effective branch supported by this excitation corpus; "
            "unselected latent modules are not claimed absent"
        ),
    }


def _fit_models(
    config: dict[str, Any],
    train_views: dict[str, list[dict[str, Any]]],
    val_views: dict[str, list[dict[str, Any]]],
    raw_train_cohorts: dict[str, list[dict[str, Any]]],
    raw_val_cohorts: dict[str, list[dict[str, Any]]],
    partitions: dict[str, tuple[tuple[int, ...], ...] | None],
    checkpoints: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    models: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    horizons = tuple(int(value) for value in config["report_horizons"])
    screening_horizon = int(config["selection_horizon_steps"])
    screening_horizons = (screening_horizon,)
    long_horizons = tuple(value for value in horizons if value != screening_horizon)

    for cohort in ("C1", "C2", "C3"):
        train_ids = {sample["event_id"] for sample in raw_train_cohorts[cohort]}
        val_ids = {sample["event_id"] for sample in raw_val_cohorts[cohort]}
        if not train_ids or not val_ids:
            continue
        context = _context(config, raw_train_cohorts[cohort])
        b2_summaries = []
        for representation, partition in partitions.items():
            train = _subset(train_views[representation], train_ids)
            val = _subset(val_views[representation], val_ids)
            if not train or not val:
                continue
            model = M0Model(max_internal_step=float(config["max_internal_step"]))
            fit = model.fit(train, val, context)
            key = f"{cohort}::{representation}::B2"
            models[key] = model
            model.save(checkpoints / f"{cohort}-{representation}-B2.npz")
            rows = _evaluate_candidate(
                model,
                representation,
                val,
                raw_val_cohorts[cohort],
                partition,
                screening_horizons,
                cohort=cohort,
                branch="B2",
            )
            metric_rows.extend(rows)
            summary = next(
                row
                for row in _aggregate(rows)
                if int(row["horizon_steps"]) == screening_horizon
            )
            b2_summaries.append((float(summary["mean_rmse"]), representation))
            candidates.append(
                {
                    "cohort": cohort,
                    "representation": representation,
                    "branch": "B2",
                    "model_key": key,
                    "complexity": _complexity("B2", len(model.damping), model.diagnostics),
                    "physics_status": model.physics_audit()["status"],
                    "fit_validation_loss": fit.best_validation_loss,
                    "selected_ridge": model.selected_ridge,
                    "condition_number": model.diagnostics.get("matrix_condition_number"),
                    "effective_rank": model.diagnostics.get("effective_rank"),
                }
            )
        finite_b2 = [row for row in b2_summaries if np.isfinite(row[0])]
        if not finite_b2:
            continue
        best_b2 = min(value for value, _ in finite_b2)
        shortlist = [
            name
            for value, name in sorted(finite_b2)
            if value <= best_b2 * (1.0 + float(config["spatial_shortlist_tolerance"]))
        ]
        required_controls = [
            name
            for name in ("clustered_7_fixed", "generator_10")
            if name in partitions
        ]
        capacity = max(
            0, int(config["maximum_spatial_shortlist"]) - len(required_controls)
        )
        shortlist = [
            *[name for name in shortlist if name not in required_controls][:capacity],
            *required_controls,
        ]

        # The complete 5--10-node B2 family is a cheap H=100 spatial screen.
        # Only shortlisted representations incur the H=500/H=1000 evidence
        # cost.  This preserves the decisive long-horizon comparison without
        # spending it on candidates already rejected by the predeclared screen.
        if long_horizons:
            for representation in shortlist:
                val = _subset(val_views[representation], val_ids)
                metric_rows.extend(
                    _evaluate_candidate(
                        models[f"{cohort}::{representation}::B2"],
                        representation,
                        val,
                        raw_val_cohorts[cohort],
                        partitions[representation],
                        long_horizons,
                        cohort=cohort,
                        branch="B2",
                    )
                )

        b3_summaries = []
        for representation in shortlist:
            train = _subset(train_views[representation], train_ids)
            val = _subset(val_views[representation], val_ids)
            anchor = deepcopy(models[f"{cohort}::{representation}::B2"])
            model = TransientEmfM0Model(
                anchor=anchor,
                alpha_candidates=tuple(float(value) for value in config["b3_alpha_candidates"]),
                tau_candidates=tuple(float(value) for value in config["b3_tau_candidates"]),
                activation_threshold_candidates=_stress_thresholds(train),
                activation_width=float(config["b3_activation_width"]),
                max_source_loss_degradation=float(config["maximum_source_loss_degradation"]),
                max_internal_step=float(config["max_internal_step"]),
            )
            fit = model.fit(train, val, context)
            key = f"{cohort}::{representation}::B3"
            models[key] = model
            model.save(checkpoints / f"{cohort}-{representation}-B3.npz")
            rows = _evaluate_candidate(
                model,
                representation,
                val,
                raw_val_cohorts[cohort],
                partitions[representation],
                horizons,
                cohort=cohort,
                branch="B3",
            )
            metric_rows.extend(rows)
            summary_rows = _aggregate(rows)
            summary = max(
                (row for row in summary_rows if int(row["eligible_cases"]) > 0),
                key=lambda row: int(row["horizon_steps"]),
            )
            b3_summaries.append((float(summary["mean_rmse"]), representation))
            candidates.append(
                {
                    "cohort": cohort,
                    "representation": representation,
                    "branch": "B3",
                    "model_key": key,
                    "complexity": _complexity("B3", len(model.damping), model.diagnostics),
                    "physics_status": model.physics_audit()["status"],
                    "fit_validation_loss": fit.best_validation_loss,
                    "selected_alpha": model.alpha,
                    "selected_tau": model.tau,
                    "selected_activation_threshold": model.activation_threshold,
                    "selected_is_b2_fallback": model.diagnostics["selected_is_M0_fallback"],
                }
            )
        finite_b3 = [row for row in b3_summaries if np.isfinite(row[0])]
        if not finite_b3:
            continue
        residual_representation = min(finite_b3)[1]
        train = _subset(train_views[residual_representation], train_ids)
        val = _subset(val_views[residual_representation], val_ids)
        gmd = GraphModalDiscoveryM0Model(
            backbone=deepcopy(models[f"{cohort}::{residual_representation}::B3"]),
            rank_candidates=tuple(int(value) for value in config["gmd_rank_candidates"]),
            tau_mode_candidates=tuple(
                tuple(float(value) for value in row) for row in config["gmd_tau_mode_candidates"]
            ),
            group_thresholds=tuple(float(value) for value in config["gmd_group_thresholds"]),
            coefficient_scales=tuple(float(value) for value in config["gmd_coefficient_scales"]),
            ridge_values=tuple(float(value) for value in config["gmd_ridge_values"]),
            weak_windows=tuple(int(value) for value in config["integral_windows"]),
            rollout_short_horizon=int(config["gmd_short_horizon"]),
            rollout_long_shortlist=int(config["gmd_long_shortlist"]),
            train_shortlist=int(config["gmd_train_shortlist"]),
            max_internal_step=float(config["max_internal_step"]),
        )
        fit = gmd.fit(train, val, context)
        branch = _branch_for_gmd(gmd)
        key = f"{cohort}::{residual_representation}::{branch}"
        models[key] = gmd
        gmd.save(checkpoints / f"{cohort}-{residual_representation}-{branch}.npz")
        metric_rows.extend(
            _evaluate_candidate(
                gmd,
                residual_representation,
                val,
                raw_val_cohorts[cohort],
                partitions[residual_representation],
                horizons,
                cohort=cohort,
                branch=branch,
            )
        )
        candidates.append(
            {
                "cohort": cohort,
                "representation": residual_representation,
                "branch": branch,
                "model_key": key,
                "complexity": _complexity(branch, len(gmd.backbone.damping), gmd.diagnostics),
                "physics_status": gmd.physics_audit()["status"],
                "fit_validation_loss": fit.best_validation_loss,
                "selected_modal_rank": gmd.diagnostics["selected_modal_rank"],
                "selected_memory_order": gmd.diagnostics["selected_memory_order"],
                "selected_taus": gmd.diagnostics["selected_taus"],
                "selected_features": gmd.diagnostics["selected_feature_names"],
                "selected_is_zero_closure": gmd.diagnostics["selected_is_zero_closure"],
            }
        )

        if cohort == "C3" and bool(config["enable_port_structure_discovery"]):
            psd = PortStructuredDiscoveryM0Model(
                backbone=deepcopy(models[f"{cohort}::{residual_representation}::B3"]),
                tau_mode_candidates=tuple(
                    tuple(float(value) for value in row)
                    for row in config["psd_tau_mode_candidates"]
                ),
                ridge_values=tuple(float(value) for value in config["psd_ridge_values"]),
                group_thresholds=tuple(float(value) for value in config["psd_group_thresholds"]),
                coefficient_scales=tuple(float(value) for value in config["psd_coefficient_scales"]),
                minimum_validation_improvement=float(config["minimum_model_improvement"]),
                maximum_source_loss_degradation=float(config["maximum_source_loss_degradation"]),
                max_internal_step=float(config["max_internal_step"]),
            )
            fit = psd.fit(train, val, context)
            branch = "PSD-B4" if psd.diagnostics.get("selected_fit_taus") else "PSD-BS"
            key = f"{cohort}::{residual_representation}::{branch}"
            models[key] = psd
            psd.save(checkpoints / f"{cohort}-{residual_representation}-{branch}.npz")
            metric_rows.extend(
                _evaluate_candidate(
                    psd,
                    residual_representation,
                    val,
                    raw_val_cohorts[cohort],
                    partitions[residual_representation],
                    horizons,
                    cohort=cohort,
                    branch=branch,
                )
            )
            candidates.append(
                {
                    "cohort": cohort,
                    "representation": residual_representation,
                    "branch": branch,
                    "model_key": key,
                    "complexity": _complexity(branch, len(psd.damping), psd.diagnostics),
                    "physics_status": psd.physics_audit()["status"],
                    "fit_validation_loss": fit.best_validation_loss,
                    "promoted": psd.diagnostics["promoted"],
                    "selected_support": psd.diagnostics["selected_support"],
                    "cross_source_edge_support_jaccard": psd.diagnostics.get(
                        "cross_source_edge_support_jaccard"
                    ),
                    "selected_taus": psd.diagnostics.get("selected_fit_taus", []),
                }
            )
    return models, candidates, metric_rows


def _parameter_rows(models: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, model in models.items():
        diagnostics = model.diagnostics
        backbone = getattr(model, "backbone", None)
        damping = getattr(model, "damping", None)
        coupling = getattr(model, "coupling", None)
        if damping is None and backbone is not None:
            damping = backbone.damping
        if coupling is None and backbone is not None:
            coupling = backbone.coupling
        rows.append(
            {
                "model_key": key,
                "method": model.method_name,
                "minimum_damping": float(np.min(damping)) if damping is not None else None,
                "median_damping": float(np.median(damping)) if damping is not None else None,
                "maximum_damping": float(np.max(damping)) if damping is not None else None,
                "active_coupling_edges": int(np.sum(np.asarray(coupling) > 0.0))
                if coupling is not None
                else None,
                "coupling_l2_norm": float(np.linalg.norm(coupling))
                if coupling is not None
                else None,
                "condition_number": diagnostics.get("matrix_condition_number"),
                "effective_rank": diagnostics.get("effective_rank"),
                "selected_modal_rank": diagnostics.get("selected_modal_rank"),
                "selected_memory_order": diagnostics.get("selected_memory_order"),
                "inertia_policy": "declared_prior_fixed_by_existing_COI_coordinate_definition",
                "machine_parameter_recovery_claim": False,
            }
        )
    return rows


def _write_report(
    result_dir: Path,
    run_id: str,
    selections: dict[str, Any],
    policy: dict[str, Any],
    counts: dict[str, Any],
) -> None:
    global_selection = selections.get("C3", {})
    report = f"""# M0-AFSD：自适应有效阶次与结构发现首轮报告

运行：`{run_id}`。范围严格限定为 F1、seed 324、validation-only 和 observed Pm 条件重积分；test 未读取。

## 结论状态

- C1：`{selections.get('C1', {}).get('status', 'NOT_RUN')}`，选择 `{selections.get('C1', {}).get('selected_representation', '-')}` / `{selections.get('C1', {}).get('selected_branch', '-')}`。
- C2：`{selections.get('C2', {}).get('status', 'NOT_RUN')}`，选择 `{selections.get('C2', {}).get('selected_representation', '-')}` / `{selections.get('C2', {}).get('selected_branch', '-')}`。
- C3：`{global_selection.get('status', 'NOT_RUN')}`，全局候选为 `{global_selection.get('selected_representation', '-')}` / `{global_selection.get('selected_branch', '-')}`。

这里的分枝表示当前扰动集合支持的最小有效阶次。没有被选择的模块不能解释为物理上不存在。

## 数据与选择边界

- 分层阈值仅由训练集建立：`{policy.get('created_from_case_ids') and '是' or '否'}`。
- 训练层计数：`{json.dumps(counts.get('train', {}), ensure_ascii=False)}`。
- 验证层计数：`{json.dumps(counts.get('val', {}), ensure_ascii=False)}`。
- observed Pm 是条件重积分输入，不是自主因果预测。
- 电压、Pe、Qe、标签、事件时间和 test ID 均未参与拟合或选择。
- 惯量保持为 canonical COI 坐标的声明先验；本轮不宣称恢复机器惯量。

## 产物

`data_strata.csv` 给出资格、激励和丰富度证据；`candidate_models.csv` 与
`selection_marker.json` 给出模型树和冻结选择；`per_case_metrics.csv`、
`aggregate_metrics.csv` 和 `cross_level_metrics.csv` 给出1/5/10秒重积分与模态证据；
`physics_audits.json`、`integration_convergence.csv` 和 `lineage.json` 给出可审计边界。
"""
    (result_dir / "report.md").write_text(report, encoding="utf-8")


def _latest_result(root: Path, experiment_name: str, stage: str) -> Path:
    prefix = f"{experiment_name}-{stage}-"
    candidates = sorted(
        path
        for path in (root / "results").glob(f"{prefix}*")
        if path.is_dir() and not (path / "RUN_FAILED.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed {experiment_name}/{stage} result")
    return candidates[-1]


def _verify_hashes(result_dir: Path) -> list[str]:
    expected = json.loads(
        (result_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    return [
        relative
        for relative, digest in expected.items()
        if not (result_dir / relative).is_file()
        or _sha256(result_dir / relative) != digest
    ]


def _run_adversarial_audit(
    root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic = _latest_result(
        root, str(config["diagnostic_experiment"]), "diagnostic"
    )
    lineage = json.loads((diagnostic / "lineage.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (diagnostic / "selection_marker.json").read_text(encoding="utf-8")
    )
    policy = json.loads(
        (diagnostic / "stratification_policy.json").read_text(encoding="utf-8")
    )
    hierarchy = json.loads(
        (diagnostic / "spatial_hierarchy.json").read_text(encoding="utf-8")
    )
    selected_rows = [
        row
        for row in selection.get("selected_by_corpus", {}).values()
        if row.get("status") == "SELECTED"
    ]
    gates = {
        "diagnostic_artifact_hashes_match": not _verify_hashes(diagnostic),
        "test_data_never_read": not lineage.get("test_ids_read"),
        "labels_not_used": not lineage.get(
            "labels_used_for_fit_selection_or_stratification"
        ),
        "stratification_is_train_only": (
            policy.get("validation_or_test_used") is False
            and policy.get("labels_used") is False
            and set(policy.get("created_from_case_ids", []))
            == set(lineage.get("train_ids_used", []))
        ),
        "spatial_hierarchy_is_train_only": (
            hierarchy["diagnostics"].get("validation_or_test_used") is False
            and hierarchy["diagnostics"].get("labels_used") is False
        ),
        "selection_marker_precedes_report_tables": (
            (diagnostic / "selection_marker.json").stat().st_mtime_ns
            <= (diagnostic / "aggregate_metrics.csv").stat().st_mtime_ns
        ),
        "selected_candidates_pass_physics_and_rollout_gates": bool(selected_rows)
        and all(
            row.get("selected_mean_rmse") is not None
            and np.isfinite(float(row["selected_mean_rmse"]))
            for row in selected_rows
        ),
        "observed_pm_claim_is_conditional": (
            selection.get("observed_Pm_is_conditional") is True
            and lineage.get("observed_Pm_is_conditional_noncausal_sensitivity") is True
        ),
        "no_event_specific_future_routing": selection.get(
            "event_specific_future_routing"
        )
        is False,
        "absence_and_true_order_are_not_claimed": (
            selection.get("unselected_modules_claimed_absent") is False
            and selection.get("absolute_true_order_recovery_claim") is False
        ),
        "inertia_recovery_boundary_is_explicit": lineage.get(
            "inertia_is_declared_prior_not_reidentified"
        )
        is True,
        "no_uncertified_lyap_name": all(
            "LYAP" not in str(value)
            for row in selection.get("selected_by_corpus", {}).values()
            for value in row.values()
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "run_id": run_id,
        "diagnostic_result": diagnostic.name,
        "status": status,
        "decision": "ACCEPTABLE" if status == "PASS" else "REJECT",
        "gates": gates,
        "adversarial_questions": [
            "Did short or non-finite trajectories enter structural selection?",
            "Were validation or test responses used to construct strata or partitions?",
            "Was future response used to route individual validation events to a branch?",
            "Was failure to select a latent block interpreted as physical absence?",
            "Were GMD modal residuals or PSD effective edges called physical topology?",
            "Was fixed canonical inertia misreported as a recovered machine parameter?",
            "Was observed Pm presented as an autonomous causal deployment input?",
            "Was an energy-regularized candidate given the M2A-LYAP name without certificates?",
        ],
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# M0-AFSD 对抗性审查\n\n"
        f"**结论：{review['decision']}**\n\n"
        + "\n".join(
            f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items()
        )
        + "\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("M0-AFSD adversarial audit failed")


def _run_acceptance(
    root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic = _latest_result(
        root, str(config["diagnostic_experiment"]), "diagnostic"
    )
    adversarial = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_audit"
    )
    review = json.loads(
        (adversarial / "adversarial_review.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (diagnostic / "selection_marker.json").read_text(encoding="utf-8")
    )
    gates = {
        "diagnostic_hashes_still_match": not _verify_hashes(diagnostic),
        "adversarial_review_passed": review.get("status") == "PASS",
        "selection_is_frozen": selection.get("selection_complete") is True,
        "test_remained_locked": not selection.get("test_ids_read"),
        "single_fold_single_seed_boundary": (
            selection.get("fold") == "F1" and int(selection.get("seed", -1)) == 324
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    acceptance = {
        "run_id": run_id,
        "status": status,
        "decision": (
            "ACCEPT_FIRST_ROUND_WITH_NEGATIVE_SCIENTIFIC_RESULTS_ALLOWED"
            if status == "PASS"
            else "REJECT"
        ),
        "diagnostic_result": diagnostic.name,
        "adversarial_result": adversarial.name,
        "gates": gates,
        "claim_boundary": (
            "F1 seed324 validation-only conditional observed-Pm effective-order study; "
            "no test, five-fold, autonomous-Pm, physical-line or true-machine-order claim"
        ),
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# M0-AFSD 首轮接收审计\n\n"
        f"**结论：{acceptance['decision']}**\n\n"
        + "\n".join(
            f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items()
        )
        + "\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("M0-AFSD acceptance failed")


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if stage == "adversarial_audit":
        _run_adversarial_audit(root, config, result_dir, run_id)
        return
    if stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
        return
    if stage != "diagnostic":
        raise ValueError(f"unsupported M0-AFSD stage: {stage}")
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    if (
        config.get("test_access") != "forbidden"
        or config.get("pm_mode") != "observed"
        or config.get("fold") != "F1"
        or int(config.get("seed", -1)) != 324
    ):
        raise RuntimeError("AFSD pilot requires F1/seed324, observed Pm and locked test")

    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold=str(config["fold"]),
    )
    minimum_steps = int(config["minimum_real_steps"])
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
        raise RuntimeError("no eligible F1 train/validation trajectories")
    request = ModelDataRequest.initial_state_rollout("observed")
    inertia = store.dataset_metadata.M_prior
    raw_train = [
        adapt_event(
            store.load_event(event_id),
            inertia_prior=inertia,
            request=request,
            representation="generator_10",
        )
        for event_id in train_ids
    ]
    raw_val = [
        adapt_event(
            store.load_event(event_id),
            inertia_prior=inertia,
            request=request,
            representation="generator_10",
        )
        for event_id in val_ids
    ]
    for sample in raw_train + raw_val:
        audit_model_sample(sample)

    horizons = tuple(int(value) for value in config["report_horizons"])
    policy, train_strata = fit_stratification_policy(
        raw_train,
        report_horizons=horizons,
        minimum_structure_steps=int(config["minimum_structure_steps"]),
        hankel_delays=int(config["hankel_delays"]),
        hankel_relative_floor=float(config["hankel_relative_floor"]),
        maximum_hankel_condition=float(config["maximum_hankel_condition"]),
    )
    val_strata = stratify_samples(raw_val, policy)
    train_cohorts, train_cohort_policy = cumulative_cohorts(
        raw_train,
        train_strata,
        activation_edges=policy.activation_edges,
        minimum_cases=int(config["minimum_selection_cases"]),
        allow_train_fallback=True,
    )
    val_cohorts, val_cohort_policy = cumulative_cohorts(
        raw_val,
        val_strata,
        activation_edges=policy.activation_edges,
        minimum_cases=1,
        allow_train_fallback=False,
    )

    structure_train_ids = {
        row["event_id"] for row in train_strata if row["structure_selection_eligible"]
    }
    partition_training = [
        sample for sample in raw_train if sample["event_id"] in structure_train_ids
    ]
    if len(partition_training) < int(config["minimum_selection_cases"]):
        partition_training = train_cohorts["C3"]
    hierarchy, hierarchy_diagnostics = discover_nested_partitions(
        partition_training,
        cluster_counts=tuple(int(value) for value in config["cluster_counts"]),
        maximum_steps=int(config["maximum_analysis_steps"]),
    )
    partitions: dict[str, tuple[tuple[int, ...], ...] | None] = {
        **{f"adaptive_k{count}": partition for count, partition in hierarchy.items()},
        "clustered_7_fixed": CLUSTERED_7_PARTITION,
        "generator_10": None,
    }
    spatial_rows = []
    for name, partition in partitions.items():
        if partition is None:
            for sample in partition_training:
                for signal in ("delta", "omega"):
                    spatial_rows.append(
                        {
                            "representation": name,
                            "event_id": sample["event_id"],
                            "source_dataset": _source(sample),
                            "signal": signal,
                            "n_clusters": 10,
                            "discarded_mass_energy_fraction": 0.0,
                            "retained_mass_energy_fraction": 1.0,
                        }
                    )
        else:
            spatial_rows.extend(
                {"representation": name, **row}
                for row in spatial_projection_audit(
                    partition_training,
                    partition,
                    maximum_steps=int(config["maximum_analysis_steps"]),
                )
            )

    train_views = _representation_samples(raw_train, partitions)
    val_views = _representation_samples(raw_val, partitions)
    checkpoints = result_dir / "checkpoints"
    checkpoints.mkdir()
    models, candidates, metric_rows = _fit_models(
        config,
        train_views,
        val_views,
        train_cohorts,
        val_cohorts,
        partitions,
        checkpoints,
    )
    aggregate = _aggregate(metric_rows)
    strata_by_id = {row["event_id"]: row for row in train_strata}
    selections = {
        cohort: _select(
            aggregate,
            candidates,
            config,
            cohort,
            sum(
                strata_by_id[sample["event_id"]]["structure_selection_eligible"]
                for sample in train_cohorts[cohort]
            ),
        )
        for cohort in ("C1", "C2", "C3")
    }

    selection_marker = {
        "run_id": run_id,
        "selection_complete": True,
        "fold": "F1",
        "seed": 324,
        "algorithm": "M0-AFSD",
        "selected_by_corpus": selections,
        "deployment_candidate": selections.get("C3"),
        "event_specific_future_routing": False,
        "test_ids_read": [],
        "labels_used_for_fit_or_selection": False,
        "measured_Pe_voltage_Qe_used": False,
        "observed_Pm_is_conditional": True,
        "unselected_modules_claimed_absent": False,
        "absolute_true_order_recovery_claim": False,
    }
    _json(result_dir / "selection_marker.json", selection_marker)

    cross_rows: list[dict[str, Any]] = []
    for training_cohort, selection in selections.items():
        if selection.get("status") != "SELECTED":
            continue
        key = str(selection["selected_model_key"])
        model = models[key]
        representation = str(selection["selected_representation"])
        branch = str(selection["selected_branch"])
        for target_cohort in ("C1", "C2", "C3"):
            ids = {sample["event_id"] for sample in val_cohorts[target_cohort]}
            target_rows = [
                {
                    **row,
                    "cohort": f"{training_cohort}->{target_cohort}",
                    "training_cohort": training_cohort,
                    "target_cohort": target_cohort,
                }
                for row in _evaluate_candidate(
                    model,
                    representation,
                    _subset(val_views[representation], ids),
                    val_cohorts[target_cohort],
                    partitions[representation],
                    horizons,
                    cohort=target_cohort,
                    branch=branch,
                )
            ]
            cross_rows.extend(target_rows)

    convergence_rows = []
    for cohort, selection in selections.items():
        if selection.get("status") != "SELECTED" or not val_cohorts[cohort]:
            continue
        model = models[str(selection["selected_model_key"])]
        representation = str(selection["selected_representation"])
        sample_id = val_cohorts[cohort][0]["event_id"]
        sample = next(row for row in val_views[representation] if row["event_id"] == sample_id)
        steps = min(max(horizons), len(sample["training_context"]["targets"]["t"]))
        try:
            audit = model.convergence_audit(
                sample["model_inputs"]["static"],
                np.asarray(sample["training_context"]["targets"]["t"][:steps]),
                {
                    "Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]),
                    "interpolation": "linear",
                },
            )
        except AttributeError:
            audit = {"convergent": False, "reason": "model_has_no_convergence_audit"}
        convergence_rows.append(
            {
                "cohort": cohort,
                "event_id": sample_id,
                "representation": representation,
                "branch": selection["selected_branch"],
                **audit,
            }
        )

    for role, rows in (("train", train_strata), ("val", val_strata)):
        for row in rows:
            row["role"] = role
            metadata = store.metadata(row["event_id"])
            row["source_dataset"] = metadata.source_dataset
            row["fault_line"] = metadata.fault_line
    _csv(result_dir / "data_strata.csv", [*train_strata, *val_strata])
    _csv(result_dir / "spatial_projection_audit.csv", spatial_rows)
    _csv(result_dir / "candidate_models.csv", candidates)
    _csv(result_dir / "per_case_metrics.csv", metric_rows)
    _csv(result_dir / "aggregate_metrics.csv", aggregate)
    _csv(result_dir / "cross_level_metrics.csv", cross_rows)
    _csv(result_dir / "cross_level_aggregate.csv", _aggregate(cross_rows) if cross_rows else [])
    _csv(result_dir / "learned_parameter_audit.csv", _parameter_rows(models))
    _csv(result_dir / "integration_convergence.csv", convergence_rows)
    _json(result_dir / "stratification_policy.json", policy.as_dict())
    _json(result_dir / "cohort_policy.json", {"train": train_cohort_policy, "val": val_cohort_policy})
    _json(result_dir / "spatial_hierarchy.json", {"partitions": partitions, "diagnostics": hierarchy_diagnostics})
    _json(result_dir / "physics_audits.json", {key: model.physics_audit() for key, model in models.items()})
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "dataset_revision": store.dataset_metadata.dataset_revision,
            "dataset_build_id": store.dataset_metadata.build_id,
            "fold": "F1",
            "seed": 324,
            "train_ids_used": train_ids,
            "val_ids_used": val_ids,
            "test_ids_read": [],
            "labels_used_for_fit_selection_or_stratification": False,
            "stratification_created_from": "train_only_delta_omega_Pm",
            "observed_Pm_is_conditional_noncausal_sensitivity": True,
            "inertia_is_declared_prior_not_reidentified": True,
            "config_sha256": _sha256(config_path),
        },
    )
    counts = {
        "train": train_cohort_policy["counts"],
        "val": val_cohort_policy["counts"],
    }
    _write_report(result_dir, run_id, selections, policy.as_dict(), counts)
    artifacts = {
        str(path.relative_to(result_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)
