from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kron_reduce.evaluation.modal import exact_dmd_modes, match_dmd_modes
from kron_reduce.models import M0Model, PortStructuredDiscoveryM0Model, TransientEmfM0Model
from kron_reduce.physics import aggregate_model_sample
from kron_reduce.physics.aggregation import aggregate_state
from kron_reduce.physics.global_quotient import (
    holdout_global_quotient_gap,
    unified_quotient_agglomeration,
)
from kron_reduce.workflows.adaptive_fidelity_structure_discovery import (
    _context,
    _evaluate_candidate,
    _rollout_row,
)
from kron_reduce.workflows.m0_afsd_v2_module_selection import (
    MODULES,
    _fit_stack,
    _module_code,
    _module_mask,
    _paired_cluster_bootstrap,
    _residual_features,
    _same_support_refits,
    _support_bootstrap,
    _theta,
)
from kron_reduce.workflows.m0_afsd_v5_unified_quotient_selection import (
    _load_data as _load_v5_data,
)
from kron_reduce.workflows.m0_afsd_v5_unified_quotient_selection import _sample_path

WORKFLOW = "m0_afsd_v6_full_model_subset_validation"
ROLLOUT_LABEL = "conditional rollout with observed Pm"


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
        for row in materialized:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _write_hashes(result_dir: Path) -> None:
    rows = {
        path.name: _sha256(path)
        for path in sorted(result_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", rows)


def _verify_hashes(result_dir: Path) -> list[str]:
    expected = json.loads((result_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        name
        for name, digest in expected.items()
        if not (result_dir / name).is_file() or _sha256(result_dir / name) != digest
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    matches = sorted(root.glob(f"results/{experiment}-{stage}-*"))
    valid = [path for path in matches if not (path / "RUN_FAILED.json").exists()]
    if not valid:
        raise FileNotFoundError(f"no completed {stage} run for {experiment}")
    return valid[-1]


def _validate_scope(config: dict[str, Any]) -> None:
    expected = {
        "fold": "F1",
        "seed": 324,
        "pm_mode": "observed",
        "test_access": "forbidden",
        "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden",
    }
    mismatched = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise RuntimeError(f"v6 scope violation: {mismatched}")
    if config.get("rollout_label") != ROLLOUT_LABEL:
        raise RuntimeError("observed-Pm conditional rollout label is mandatory")


def deterministic_fault_line_roles(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    fit_fraction: float,
    selection_fraction: float,
) -> dict[str, str]:
    """Assign source-balanced, fault-line-disjoint development roles."""

    if fit_fraction <= 0 or selection_fraction <= 0 or fit_fraction + selection_fraction >= 1:
        raise ValueError("development fractions must leave nonempty audit mass")
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_source[str(row["source_dataset"])].add(str(row["fault_line"]))
    output: dict[str, str] = {}
    for source_index, source in enumerate(sorted(by_source)):
        lines = sorted(by_source[source])
        if len(lines) < 3:
            raise ValueError(f"source {source!r} has fewer than three fault-line clusters")
        rng = np.random.default_rng(int(seed) + 1009 * (source_index + 1))
        order = np.asarray(lines, dtype=object)[rng.permutation(len(lines))].tolist()
        n_fit = min(max(1, round(fit_fraction * len(lines))), len(lines) - 2)
        n_selection = min(
            max(1, round(selection_fraction * len(lines))),
            len(lines) - n_fit - 1,
        )
        for index, line in enumerate(order):
            role = "fit" if index < n_fit else "selection" if index < n_fit + n_selection else "audit"
            if str(line) in output:
                raise ValueError("a fault line cannot be assigned by two source strata")
            output[str(line)] = role
    return output


def _subset_eligible(row: dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(
        row.get("quality_grade") == "Q-A"
        and row.get("amplitude_level") == config["subset_amplitude_level"]
        and row.get("richness_level") == config["subset_richness_level"]
        and row.get("structure_selection_eligible")
        and row.get("eligible_h1000")
    )


def _select_minimum_branch(
    rows: Sequence[dict[str, Any]],
    complexity: dict[str, int],
    tolerance: float,
) -> dict[str, Any]:
    admissible = [
        row
        for row in rows
        if float(row.get("failure_rate", 1.0)) == 0.0
        and np.isfinite(float(row.get("mean_rmse", np.nan)))
        and row["branch"] in complexity
    ]
    if not admissible:
        return {"status": "INSUFFICIENT_INFORMATION", "reason": "no finite zero-failure branch"}
    best = min(float(row["mean_rmse"]) for row in admissible)
    near = [row for row in admissible if float(row["mean_rmse"]) <= best * (1.0 + tolerance) + 1e-15]
    selected = min(near, key=lambda row: (complexity[row["branch"]], float(row["mean_rmse"]), row["branch"]))
    return {
        "status": "SELECTED",
        "route": selected["branch"],
        "mean_rmse": float(selected["mean_rmse"]),
        "best_mean_rmse": best,
        "complexity": complexity[selected["branch"]],
        "within_best_tolerance": float(tolerance),
        "near_best": sorted(row["branch"] for row in near),
    }


def _hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "registered_before_response_metrics": True,
        "primary_hypothesis": (
            "v5-fixed raw ten-machine space still requires data-selected dynamic closure; "
            "the minimum admissible branch is chosen from B2, full-grid M0-3EG and the "
            "same-support conservative/dissipative/memory lattice using train-only roles"
        ),
        "subset_hypothesis": (
            "the source-blind Q-A A-low/R-low stratum may admit a coarser quotient and "
            "can improve retained-output rollout accuracy through regularization"
        ),
        "global_spatial_branch": "raw generator_10 fixed by accepted v5 evidence",
        "dynamic_candidates": ["B2", "B3", "SS-zero", "SS-C", "SS-D", "SS-M", "SS-CD", "SS-CM", "SS-DM", "SS-CDM"],
        "selection_horizon_steps": 1000,
        "equivalence_tolerance": float(config["minimum_model_improvement"]),
        "subset_rule": {
            "quality": "Q-A",
            "amplitude": config["subset_amplitude_level"],
            "richness": config["subset_richness_level"],
            "source_used": False,
            "offline_only": True,
            "uqg_tolerance": float(config["subset_uqg_tolerance"]),
            "pilot_informed": True,
        },
        "primary_metrics": [
            "1/5/10 s mean, median, pooled and worst-10%-CVaR RMSE",
            "omega and edge-angle RMSE",
            "dominant frequency, damping, MAC and spectral overlap",
            "failure rate, COI drift and RK4 convergence",
        ],
        "acceptance_logic": {
            "model": "minimum-complexity branch within 2% of best train-selection 10 s RMSE",
            "confirmation": "fault-line-disjoint train audit and frozen F1 validation; neither changes the branch",
            "cluster_better": "at least 2% lower validation RMSE with positive fault-line bootstrap lower bound and zero failures",
            "negative_cluster_result_is_valid": True,
        },
        "claim_boundaries": config["claim_boundaries"],
        "historical_result_artifacts_read": [],
        "test_ids_read": [],
    }


def _run_hypothesis(config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    _validate_scope(config)
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "hypothesis_registry.json", _hypothesis_registry(config, run_id))
    _json(result_dir / "lineage.json", {"run_id": run_id, "stage": "hypothesis_registration", "test_ids_read": []})
    _write_hashes(result_dir)


def _model_complexity(models: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {"B2": 20, "B3": 30}
    for route, model in models.items():
        if not route.startswith("SS-"):
            continue
        support = model.diagnostics.get("selected_support", {})
        active = int(model.diagnostics.get("active_coefficients_after_refit", 0))
        dynamic = len(model.taus) * int(support.get("active_memory_edges", 0))
        output[route] = 30 + active + dynamic
    return output


def _nonzero_psd_for_lattice(
    fitted: PortStructuredDiscoveryM0Model,
    train: list[dict[str, Any]],
    config: dict[str, Any],
) -> PortStructuredDiscoveryM0Model:
    """Materialize the best screened nonzero closure even when the promotion gate keeps zero."""

    selected = fitted.diagnostics.get("selected_validation", {})
    if selected.get("ridge") is not None:
        return fitted
    candidates = [
        row
        for row in fitted.diagnostics.get("candidate_table", [])
        if row.get("candidate") == "port_structured_closure"
        and row.get("ridge") is not None
        and np.isfinite(float(row.get("standardized_loss", row.get("short_standardized_loss", np.inf))))
    ]
    if not candidates:
        raise RuntimeError("PSD lattice has no finite nonzero screened candidate")
    choice = min(
        candidates,
        key=lambda row: (
            float(row.get("standardized_loss", row.get("short_standardized_loss", np.inf))),
            len(row.get("fit_taus", [])),
            float(row["ridge"]),
        ),
    )
    fit_taus = tuple(float(value) for value in choice["fit_taus"])
    context = _context(config, train)
    prepared = fitted._prepare_weak_paths(
        train,
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
    )
    stats = fitted._build_normal_equations(prepared, fit_taus, windows=context.integral_windows)
    theta, solver = fitted._group_prune_and_refit(
        stats,
        float(choice["ridge"]),
        float(choice["relative_group_threshold"]),
    )
    theta *= float(choice["coefficient_scale"])
    candidate = deepcopy(fitted)
    candidate._set_candidate(
        {"n_edges": int(choice["n_edges"]), "fit_taus": fit_taus, "theta": theta}
    )
    materialized = {
        **choice,
        **solver,
        "fit_taus": list(fit_taus),
        "n_edges": int(choice["n_edges"]),
    }
    candidate.diagnostics = {
        **candidate.diagnostics,
        "selected_validation": materialized,
        "selected_fit_taus": list(fit_taus),
        "selected_support": candidate._support_from_theta(
            theta, int(choice["n_edges"]), len(fit_taus)
        ),
        "promoted": False,
        "lattice_materialized_despite_zero_promotion": True,
    }
    candidate.diagnostics["storage_audit"] = candidate._conservative_storage_audit(train)
    return candidate


def _validated_failed_development_cache(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None:
    """Reuse only deterministic development checkpoints from a plot-only failed run."""

    matches = sorted(
        root.glob(
            f"results/{config['controlled_experiment']}-controlled_experiment-*"
        ),
        reverse=True,
    )
    for candidate in matches:
        if not (candidate / "RUN_FAILED.json").is_file():
            continue
        frozen = candidate / "frozen_config.toml"
        checkpoint = candidate / "checkpoints"
        required = {
            "B2": checkpoint / "development_raw10-B2.npz",
            "B3": checkpoint / "development_raw10-B3.npz",
            "PSD-selected": checkpoint / "development_raw10-PSD-lattice.npz",
        }
        if (
            not frozen.is_file()
            or frozen.read_bytes() != config_path.read_bytes()
            or not all(path.is_file() for path in required.values())
        ):
            continue
        models = {
            "B2": M0Model.load(required["B2"]),
            "B3": TransientEmfM0Model.load(required["B3"]),
            "PSD-selected": PortStructuredDiscoveryM0Model.load(
                required["PSD-selected"]
            ),
        }
        if any(model.physics_audit().get("status") != "pass" for model in models.values()):
            continue
        b3_diagnostics = models["B3"].diagnostics
        if (
            b3_diagnostics.get("alpha_candidates")
            != list(config["b3_alpha_candidates"])
            or b3_diagnostics.get("tau_candidates")
            != list(config["b3_tau_candidates"])
        ):
            continue
        hashes = {
            str(path.relative_to(candidate)): _sha256(path)
            for path in required.values()
        }
        metadata = [
            {
                "route": route,
                "recovered_deterministic_development_cache": True,
                "physics_status": model.physics_audit()["status"],
            }
            for route, model in models.items()
        ]
        return (
            models,
            metadata,
            {
                "status": "VALIDATED_DEVELOPMENT_CACHE_REUSED",
                "failed_run": candidate.name,
                "failure_stage": "optional_plot_render_after_numerical_computation",
                "cache_is_evidence": False,
                "downstream_metrics_recomputed": True,
                "frozen_config_sha256": _sha256(frozen),
                "checkpoint_hashes": hashes,
            },
        )
    return None


def _detailed_aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(str(row["cohort"]), str(row["representation"]), str(row["branch"]), int(row["horizon_steps"])) for row in rows})
    for cohort, representation, branch, horizon in keys:
        group = [row for row in rows if (row["cohort"], row["representation"], row["branch"], int(row["horizon_steps"])) == (cohort, representation, branch, horizon)]
        eligible = [row for row in group if row["status"] != "ineligible_short_horizon"]
        completed = [row for row in eligible if row["status"] == "completed"]
        metrics = completed + [{"rmse": float("nan")} for row in eligible if row["status"] == "failed"]
        summary = aggregate_trajectory_metrics(metrics) if metrics else aggregate_trajectory_metrics([])

        def mean(key: str, completed_rows: list[dict[str, Any]] = completed) -> float:
            values = np.asarray([row.get(key, np.nan) for row in completed_rows], dtype=float)
            values = values[np.isfinite(values)]
            return float(np.mean(values)) if len(values) else float("nan")

        def median(key: str, completed_rows: list[dict[str, Any]] = completed) -> float:
            values = np.asarray([row.get(key, np.nan) for row in completed_rows], dtype=float)
            values = values[np.isfinite(values)]
            return float(np.median(values)) if len(values) else float("nan")

        line_means = []
        for line in sorted({str(row["fault_line"]) for row in completed}):
            line_means.append(float(np.mean([float(row["rmse"]) for row in completed if str(row["fault_line"]) == line])))
        output.append(
            {
                "cohort": cohort,
                "representation": representation,
                "branch": branch,
                "horizon_steps": horizon,
                "eligible_cases": len(eligible),
                "completed_cases": len(completed),
                "fault_line_clusters": len(line_means),
                **summary,
                "fault_line_equal_mean_rmse": float(np.mean(line_means)) if line_means else float("nan"),
                "mean_delta_rmse": mean("delta_rmse"),
                "mean_omega_rmse": mean("omega_rmse"),
                "mean_edge_angle_rmse": mean("edge_angle_rmse"),
                "mean_spectral_overlap": mean("spectral_overlap"),
                "median_dominant_frequency_error_hz": median("dominant_frequency_absolute_error_hz"),
                "median_modal_frequency_relative_error": median("modal_frequency_relative_error"),
                "median_modal_damping_absolute_error": median("modal_damping_absolute_error"),
                "median_modal_mac": median("modal_mac"),
                "maximum_coi_drift": max(
                    [max(float(row["maximum_coi_delta_drift"]), float(row["maximum_coi_omega_drift"])) for row in completed],
                    default=float("nan"),
                ),
            }
        )
    return output


def _fixed_family_refit(
    development: dict[str, Any],
    structure_train: list[dict[str, Any]],
    coefficient_train: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Refit train-selected hyperparameters, then freeze Q-A support before Q-B coefficients."""

    if not structure_train or not coefficient_train:
        raise ValueError("fixed refit requires nonempty structure and coefficient populations")
    preliminary_b2: M0Model = development["B2"]
    frozen = dict(config)
    frozen["ridge_values"] = [float(preliminary_b2.selected_ridge)]
    b2 = M0Model(max_internal_step=float(config["max_internal_step"]))
    b2.fit(coefficient_train, coefficient_train, _context(frozen, coefficient_train))
    b2.diagnostics = {
        **b2.diagnostics,
        "frozen_from_development": True,
        "Q_B_used_for_structure": False,
        "coefficient_population_size": len(coefficient_train),
    }

    preliminary_b3: TransientEmfM0Model = development["B3"]
    b3 = TransientEmfM0Model(
        anchor=deepcopy(b2),
        alpha=float(preliminary_b3.alpha),
        tau=float(preliminary_b3.tau),
        activation_threshold=float(preliminary_b3.activation_threshold),
        activation_width=float(preliminary_b3.activation_width),
        max_internal_step=float(config["max_internal_step"]),
    )
    equilibrium_residual = max(
        b3.equilibrium_residual(sample["model_inputs"]["static"])
        for sample in structure_train
    )
    b3.diagnostics = {
        "optimizer": "frozen_development_grid_choice",
        "epochs": 0,
        "selected_alpha": b3.alpha,
        "selected_tau": b3.tau,
        "selected_activation_threshold": b3.activation_threshold,
        "selected_is_M0_fallback": bool(b3.alpha == 0.0),
        "maximum_equilibrium_residual": float(equilibrium_residual),
        "Q_B_used_for_structure": False,
    }

    preliminary_psd: PortStructuredDiscoveryM0Model = development["PSD-selected"]
    selected = preliminary_psd.diagnostics["selected_validation"]
    fit_taus = tuple(float(value) for value in selected["fit_taus"])
    ridge = float(selected["ridge"])
    scale = float(selected["coefficient_scale"])
    n_edges = int(selected["n_edges"])
    n_modes = len(fit_taus)
    template = deepcopy(preliminary_psd)
    template.backbone = deepcopy(b3)
    context_structure = _context(config, structure_train)
    prepared_structure = template._prepare_weak_paths(
        structure_train,
        horizon_steps=context_structure.horizon_steps,
        max_fit_steps=context_structure.fit_max_steps,
    )
    structure_stats = template._build_normal_equations(
        prepared_structure, fit_taus, windows=context_structure.integral_windows
    )
    context_coefficient = _context(config, coefficient_train)
    prepared_coefficient = template._prepare_weak_paths(
        coefficient_train,
        horizon_steps=context_coefficient.horizon_steps,
        max_fit_steps=context_coefficient.fit_max_steps,
    )
    coefficient_stats = template._build_normal_equations(
        prepared_coefficient, fit_taus, windows=context_coefficient.integral_windows
    )
    preliminary_support = _theta(preliminary_psd, n_modes) > max(
        float(np.max(_theta(preliminary_psd, n_modes))) * 1e-12,
        np.finfo(float).eps,
    )
    models: dict[str, Any] = {"B2": b2, "B3": b3}
    rows: list[dict[str, Any]] = [
        {
            "route": "B2",
            "structure_population": "fixed equation",
            "coefficient_population_size": len(coefficient_train),
            "physics_status": b2.physics_audit()["status"],
        },
        {
            "route": "B3",
            "structure_population": "Q-A development grid choice",
            "coefficient_population_size": len(coefficient_train),
            "selected_alpha": b3.alpha,
            "selected_tau": b3.tau,
            "physics_status": b3.physics_audit()["status"],
        },
    ]
    for flags in product((False, True), repeat=3):
        modules = {module for module, enabled in zip(MODULES, flags, strict=True) if enabled}
        route = _module_code(modules)
        module_columns = _module_mask(n_edges, n_modes, modules)
        discovery_columns = preliminary_support & module_columns
        theta_qa, _, qa_solver = template._factorized_nonnegative_ridge(
            structure_stats["gram"],
            structure_stats["response"],
            structure_stats["target_energy"],
            structure_stats["effective_blocks"],
            ridge,
            discovery_columns,
        )
        qa_support = theta_qa > max(float(np.max(theta_qa)) * 1e-12, np.finfo(float).eps)
        theta, _, solver = template._factorized_nonnegative_ridge(
            coefficient_stats["gram"],
            coefficient_stats["response"],
            coefficient_stats["target_energy"],
            coefficient_stats["effective_blocks"],
            ridge,
            qa_support,
        )
        theta *= scale
        candidate = deepcopy(template)
        candidate._set_candidate({"n_edges": n_edges, "fit_taus": fit_taus, "theta": theta})
        candidate.diagnostics = {
            "selected_validation": dict(selected),
            "selected_support": candidate._support_from_theta(theta, n_edges, n_modes),
            "enabled_modules": sorted(modules),
            "same_support_ablation": True,
            "frozen_development_hyperparameters": True,
            "Q_A_structure_population_size": len(structure_train),
            "Q_B_used_for_structure": False,
            "coefficient_population_size": len(coefficient_train),
            "preliminary_support_columns": int(np.sum(preliminary_support)),
            "Q_A_support_columns": int(np.sum(qa_support)),
            "active_coefficients_after_refit": int(np.sum(theta > 0)),
            "fixed_ridge": ridge,
            "fixed_coefficient_scale": scale,
            "fixed_fit_taus": list(fit_taus),
            "Q_A_support_solver": qa_solver,
            "coefficient_refit_solver": solver,
        }
        candidate.diagnostics["storage_audit"] = candidate._conservative_storage_audit(
            structure_train
        )
        models[route] = candidate
        rows.append(
            {
                "route": route,
                "enabled_modules": sorted(modules),
                "fit_taus": list(fit_taus),
                "Q_A_support_columns": int(np.sum(qa_support)),
                "active_coefficients_after_refit": int(np.sum(theta > 0)),
                "matrix_condition_number": solver["matrix_condition_number"],
                "effective_rank": solver["effective_rank"],
                "weak_residual_rmse": solver["weak_residual_rmse"],
                "physics_status": candidate.physics_audit()["status"],
            }
        )
    return models, rows


def _evaluate_routes(
    models: dict[str, Any],
    samples: list[dict[str, Any]],
    raw_samples: list[dict[str, Any]],
    partition: tuple[tuple[int, ...], ...] | None,
    horizons: tuple[int, ...],
    *,
    cohort: str,
    representation: str,
) -> list[dict[str, Any]]:
    rows = []
    for route, model in models.items():
        rows.extend(
            _evaluate_candidate(
                model,
                representation,
                samples,
                raw_samples,
                partition,
                horizons,
                cohort=cohort,
                branch=route,
            )
        )
    return rows


def _modal_match_rows(
    model: Any,
    samples: list[dict[str, Any]],
    horizons: Sequence[int],
    *,
    branch: str,
    population: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample in samples:
        target = sample["training_context"]["targets"]
        for horizon in horizons:
            steps = min(int(horizon), len(target["t"]))
            if steps < int(horizon) or steps < 500:
                continue
            static = sample["model_inputs"]["static"]
            try:
                delta, omega = model.rollout(
                    static,
                    np.asarray(target["t"][:steps]),
                    {
                        "Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]),
                        "interpolation": "linear",
                    },
                )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
                continue
            del delta
            dt = float(np.median(np.diff(np.asarray(target["t"][:steps]))))
            reference = exact_dmd_modes(np.asarray(target["omega"][:steps]), dt, rank=min(6, omega.shape[1]))
            candidate = exact_dmd_modes(omega, dt, rank=min(6, omega.shape[1]))
            for index, match in enumerate(match_dmd_modes(reference, candidate)):
                output.append(
                    {
                        "event_id": sample["event_id"],
                        "fault_line": sample["training_context"]["metadata"]["fault_line"],
                        "source_dataset": sample["training_context"]["metadata"]["source_dataset"],
                        "population": population,
                        "branch": branch,
                        "horizon_steps": int(horizon),
                        "match_index": index,
                        **match,
                    }
                )
    return output


def _sign_flip_test(
    reference: Sequence[dict[str, Any]],
    comparator: Sequence[dict[str, Any]],
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, Any]:
    left = {str(row["event_id"]): row for row in reference if row["status"] == "completed"}
    right = {str(row["event_id"]): row for row in comparator if row["status"] == "completed"}
    by_line: dict[str, list[float]] = defaultdict(list)
    for event_id in sorted(set(left) & set(right)):
        by_line[str(left[event_id]["fault_line"])].append(float(right[event_id]["rmse"]) - float(left[event_id]["rmse"]))
    line_differences = np.asarray([np.mean(values) for _, values in sorted(by_line.items())])
    if len(line_differences) < 2:
        return {"status": "INSUFFICIENT_INFORMATION", "n_fault_lines": len(line_differences)}
    observed = float(np.mean(line_differences))
    if len(line_differences) <= 15:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=len(line_differences))))
        null = np.mean(signs * line_differences[None, :], axis=1)
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(draws, len(line_differences)))
        null = np.mean(signs * line_differences[None, :], axis=1)
        method = "random"
    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))
    return {
        "status": "completed",
        "method": method,
        "n_fault_lines": len(line_differences),
        "observed_comparator_minus_reference": observed,
        "two_sided_p_value": p_value,
        "draws": len(null),
    }


def _subset_fixed_model(
    development: dict[str, Any],
    samples: list[dict[str, Any]],
    selected_route: str,
    config: dict[str, Any],
) -> Any:
    preliminary_b2: M0Model = development["B2"]
    frozen = dict(config)
    frozen["ridge_values"] = [float(preliminary_b2.selected_ridge)]
    b2 = M0Model(max_internal_step=float(config["max_internal_step"]))
    b2.fit(samples, samples, _context(frozen, samples))
    if selected_route == "B2":
        return b2
    preliminary_b3: TransientEmfM0Model = development["B3"]
    b3 = TransientEmfM0Model(
        anchor=deepcopy(b2),
        alpha=float(preliminary_b3.alpha),
        tau=float(preliminary_b3.tau),
        activation_threshold=float(preliminary_b3.activation_threshold),
        activation_width=float(preliminary_b3.activation_width),
        max_internal_step=float(config["max_internal_step"]),
    )
    b3.diagnostics = {
        "maximum_equilibrium_residual": max(
            b3.equilibrium_residual(sample["model_inputs"]["static"])
            for sample in samples
        ),
        "subset_refit_with_global_frozen_dynamic_hyperparameters": True,
    }
    if selected_route == "B3":
        return b3

    preliminary_psd: PortStructuredDiscoveryM0Model = development["PSD-selected"]
    selected = preliminary_psd.diagnostics["selected_validation"]
    fit_taus = tuple(float(value) for value in selected["fit_taus"])
    ridge = float(selected["ridge"])
    threshold = float(selected["relative_group_threshold"])
    scale = float(selected["coefficient_scale"])
    n_edges = int(np.asarray(samples[0]["model_inputs"]["static"]["incidence"]).shape[1])
    template = deepcopy(preliminary_psd)
    template.backbone = b3
    context = _context(config, samples)
    prepared = template._prepare_weak_paths(
        samples,
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
    )
    stats = template._build_normal_equations(prepared, fit_taus, windows=context.integral_windows)
    discovered, _ = template._group_prune_and_refit(stats, ridge, threshold)
    discovered_support = discovered > max(
        float(np.max(discovered)) * 1e-12, np.finfo(float).eps
    )
    symbols = set() if selected_route == "SS-zero" else set(selected_route.removeprefix("SS-"))
    modules = {
        name
        for name, symbol in (("conservative", "C"), ("dissipative", "D"), ("memory", "M"))
        if symbol in symbols
    }
    active = discovered_support & _module_mask(n_edges, len(fit_taus), modules)
    theta, _, solver = template._factorized_nonnegative_ridge(
        stats["gram"],
        stats["response"],
        stats["target_energy"],
        stats["effective_blocks"],
        ridge,
        active,
    )
    theta *= scale
    candidate = deepcopy(template)
    candidate._set_candidate({"n_edges": n_edges, "fit_taus": fit_taus, "theta": theta})
    candidate.diagnostics = {
        "selected_validation": dict(selected),
        "selected_support": candidate._support_from_theta(theta, n_edges, len(fit_taus)),
        "enabled_modules": sorted(modules),
        "subset_refit_with_global_frozen_dynamic_hyperparameters": True,
        "subset_support_rediscovered_with_frozen_threshold": True,
        "active_coefficients_after_refit": int(np.sum(theta > 0)),
        "coefficient_refit_solver": solver,
    }
    candidate.diagnostics["storage_audit"] = candidate._conservative_storage_audit(samples)
    return candidate


def _subset_comparison_row(
    model: Any,
    representation_sample: dict[str, Any],
    raw_sample: dict[str, Any],
    partition: tuple[tuple[int, ...], ...] | None,
    comparison_partition: tuple[tuple[int, ...], ...],
    horizon: int,
    *,
    branch: str,
) -> dict[str, Any]:
    row = {
        "cohort": "subset_validation",
        "representation": "generator_10" if partition is None else f"quotient_{len(partition)}",
        "branch": branch,
        **_rollout_row(model, representation_sample, raw_sample, partition, horizon),
    }
    if row["status"] != "completed":
        return row
    target = representation_sample["training_context"]["targets"]
    steps = int(horizon)
    try:
        delta, omega = model.rollout(
            representation_sample["model_inputs"]["static"],
            np.asarray(target["t"][:steps]),
            {
                "Pm": np.asarray(representation_sample["model_inputs"]["exogenous"]["Pm"][:steps]),
                "interpolation": "linear",
            },
        )
        raw_target = raw_sample["training_context"]["targets"]
        mass = np.asarray(raw_sample["model_inputs"]["static"]["mass"])
        truth_delta = aggregate_state(
            np.asarray(raw_target["delta"][:steps]), mass, comparison_partition
        )
        truth_omega = aggregate_state(
            np.asarray(raw_target["omega"][:steps]), mass, comparison_partition
        )
        if partition is None:
            pred_delta = aggregate_state(delta, mass, comparison_partition)
            pred_omega = aggregate_state(omega, mass, comparison_partition)
        else:
            pred_delta, pred_omega = delta, omega
        q_metrics = trajectory_metrics(truth_delta, truth_omega, pred_delta, pred_omega)
        row.update({f"quotient_{key}": value for key, value in q_metrics.items()})
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
        row["quotient_rmse"] = float("nan")
    return row


def _subset_experiment(
    data: dict[str, Any],
    row_by_id: dict[str, dict[str, Any]],
    development: dict[str, Any],
    selected_route: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_ids = {
        event_id
        for event_id, row in row_by_id.items()
        if row["role"] == "train" and _subset_eligible(row, config)
    }
    validation_ids = {
        event_id
        for event_id, row in row_by_id.items()
        if row["role"] == "validation" and _subset_eligible(row, config)
    }
    train_raw = [sample for sample in data["train"] if str(sample["event_id"]) in train_ids]
    validation_raw = [sample for sample in data["validation"] if str(sample["event_id"]) in validation_ids]
    mass = np.asarray(train_raw[0]["model_inputs"]["static"]["mass"], dtype=float)
    paths = [_sample_path(sample, row_by_id[str(sample["event_id"])], 1001) for sample in train_raw]
    selection = unified_quotient_agglomeration(
        paths,
        mass,
        horizons=tuple(int(value) for value in config["report_horizons"]),
        screen_horizon=100,
        minimum_clusters=int(config["subset_minimum_clusters"]),
        folds=min(int(config["subset_uqg_folds"]), len({path["fault_line"] for path in paths})),
        delays=int(config["subset_probe_delays"]),
        ridge=float(config["subset_probe_ridge"]),
        rank_cap=int(config["subset_probe_rank_cap"]),
        max_rows_per_fault_line=int(config["subset_maximum_rows_per_fault_line"]),
        tail_fraction=float(config["subset_tail_fraction"]),
        tolerance=float(config["subset_uqg_tolerance"]),
        screen_bootstrap_draws=min(200, int(config["subset_bootstrap_draws"])),
        final_bootstrap_draws=int(config["subset_bootstrap_draws"]),
        seed=int(config["seed"]),
    )
    if not selection["selected_path"]:
        return (
            {
                "status": "NO_CERTIFIED_SUBSET_MERGE",
                "selection": selection,
                "train_cases": len(train_raw),
                "validation_cases": len(validation_raw),
            },
            [],
            {},
            {},
        )
    partition = tuple(tuple(group) for group in selection["selected_partition"])
    validation_paths = [
        _sample_path(sample, row_by_id[str(sample["event_id"])], 1001)
        for sample in validation_raw
    ]
    holdout = (
        holdout_global_quotient_gap(
            paths,
            validation_paths,
            mass,
            partition,
            horizons=tuple(int(value) for value in config["report_horizons"]),
            delays=int(config["subset_probe_delays"]),
            ridge=float(config["subset_probe_ridge"]),
            rank_cap=int(config["subset_probe_rank_cap"]),
            max_rows_per_fault_line=int(config["subset_maximum_rows_per_fault_line"]),
            tail_fraction=float(config["subset_tail_fraction"]),
            tolerance=float(config["subset_uqg_tolerance"]),
            bootstrap_draws=int(config["subset_bootstrap_draws"]),
            seed=int(config["seed"]),
        )
        if validation_paths
        else {"status": "INSUFFICIENT_INFORMATION"}
    )
    train_cluster = [aggregate_model_sample(sample, partition, f"subset_q{len(partition)}") for sample in train_raw]
    validation_cluster = [aggregate_model_sample(sample, partition, f"subset_q{len(partition)}") for sample in validation_raw]
    for sample in train_cluster + validation_cluster:
        audit_model_sample(sample)
    raw_model = _subset_fixed_model(development, train_raw, selected_route, config)
    cluster_model = _subset_fixed_model(development, train_cluster, selected_route, config)
    horizons = tuple(int(value) for value in config["report_horizons"])
    metric_rows: list[dict[str, Any]] = []
    for raw_sample, cluster_sample in zip(validation_raw, validation_cluster, strict=True):
        for horizon in horizons:
            metric_rows.append(
                _subset_comparison_row(
                    raw_model,
                    raw_sample,
                    raw_sample,
                    None,
                    partition,
                    horizon,
                    branch="raw10_subset_refit",
                )
            )
            metric_rows.append(
                _subset_comparison_row(
                    cluster_model,
                    cluster_sample,
                    raw_sample,
                    partition,
                    partition,
                    horizon,
                    branch=f"cluster{len(partition)}_subset_refit",
                )
            )
    h1000 = [row for row in metric_rows if int(row["horizon_steps"]) == 1000]
    cluster_rows = [row for row in h1000 if row["branch"].startswith("cluster")]
    raw_rows = [row for row in h1000 if row["branch"].startswith("raw10")]
    bootstrap = _paired_cluster_bootstrap(
        cluster_rows,
        raw_rows,
        draws=int(config["performance_bootstrap_draws"]),
        confidence=float(config["bootstrap_confidence"]),
        seed=int(config["seed"]) + 6000,
    )
    raw_mean = float(np.mean([row["rmse"] for row in raw_rows if row["status"] == "completed"])) if raw_rows else float("nan")
    cluster_mean = float(np.mean([row["rmse"] for row in cluster_rows if row["status"] == "completed"])) if cluster_rows else float("nan")
    raw_quotient_mean = float(
        np.mean([row["quotient_rmse"] for row in raw_rows if np.isfinite(row.get("quotient_rmse", np.nan))])
    ) if raw_rows else float("nan")
    cluster_quotient_mean = float(
        np.mean([row["quotient_rmse"] for row in cluster_rows if np.isfinite(row.get("quotient_rmse", np.nan))])
    ) if cluster_rows else float("nan")
    improvement = (raw_mean - cluster_mean) / max(raw_mean, np.finfo(float).eps)
    status = (
        "CLUSTER_BETTER"
        if holdout.get("status") == "EMPIRICALLY_SUFFICIENT"
        and improvement >= float(config["minimum_model_improvement"])
        and float(bootstrap.get("ci_lower", -np.inf)) > 0.0
        else "CLUSTER_NOT_BETTER"
        if np.isfinite(improvement) and float(bootstrap.get("ci_upper", np.inf)) <= 0.0
        else "INSUFFICIENT_INFORMATION"
    )
    benchmark: dict[str, Any] = {}
    if validation_raw:
        for name, model, sample in (
            ("raw10", raw_model, validation_raw[0]),
            (f"cluster{len(partition)}", cluster_model, validation_cluster[0]),
        ):
            target = sample["training_context"]["targets"]
            timings = []
            for _ in range(3):
                start = perf_counter()
                model.rollout(
                    sample["model_inputs"]["static"],
                    np.asarray(target["t"][:1000]),
                    {"Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:1000]), "interpolation": "linear"},
                )
                timings.append(perf_counter() - start)
            benchmark[name] = {"median_seconds": float(np.median(timings)), "repeats": 3}
    summary = {
        "status": status,
        "subset_rule": {
            "quality": "Q-A",
            "amplitude_level": config["subset_amplitude_level"],
            "richness_level": config["subset_richness_level"],
            "source_label_used": False,
        },
        "train_cases": len(train_raw),
        "train_fault_lines": len({sample["training_context"]["metadata"]["fault_line"] for sample in train_raw}),
        "validation_cases": len(validation_raw),
        "validation_fault_lines": len({sample["training_context"]["metadata"]["fault_line"] for sample in validation_raw}),
        "partition": [list(group) for group in partition],
        "certified_training_merges": len(selection["selected_path"]),
        "minimality_status": selection["status"],
        "holdout_uqg": holdout,
        "raw10_mean_rmse_10s": raw_mean,
        "cluster_mean_rmse_10s": cluster_mean,
        "raw10_quotient_mean_rmse_10s": raw_quotient_mean,
        "cluster_quotient_mean_rmse_10s": cluster_quotient_mean,
        "relative_improvement": improvement,
        "performance_bootstrap": bootstrap,
        "claim_scope": "offline conditional stratum; not an online router or global spatial model",
    }
    return summary, metric_rows, benchmark, {"raw10": raw_model, "cluster": cluster_model}


def _system_information(model: Any, selected_route: str) -> dict[str, Any]:
    backbone = model
    if isinstance(model, PortStructuredDiscoveryM0Model):
        backbone = model.backbone
    m0 = backbone.anchor if isinstance(backbone, TransientEmfM0Model) else backbone
    output: dict[str, Any] = {
        "selected_route": selected_route,
        "parameter_interpretation": "effective dynamic-equivalent parameters, not physical machine or 39-bus line recovery",
        "damping": np.asarray(m0.damping).tolist(),
        "coupling": np.asarray(m0.coupling).tolist(),
        "selected_ridge": m0.selected_ridge,
        "minimum_damping": float(np.min(m0.damping)),
        "minimum_coupling": float(np.min(m0.coupling)),
        "active_damping_count": int(np.sum(np.asarray(m0.damping) > 1e-12)),
        "active_coupling_count": int(np.sum(np.asarray(m0.coupling) > 1e-12)),
    }
    if isinstance(backbone, TransientEmfM0Model):
        output["M0_3EG"] = {
            "alpha": backbone.alpha,
            "tau_seconds": backbone.tau,
            "activation_threshold": backbone.activation_threshold,
            "is_exact_B2_fallback": bool(backbone.alpha == 0.0),
        }
    if isinstance(model, PortStructuredDiscoveryM0Model):
        output["port_structured_closure"] = {
            "enabled_modules": model.diagnostics.get("enabled_modules", []),
            "memory_taus_seconds": np.asarray(model.taus).tolist(),
            "selected_support": model.diagnostics.get("selected_support", {}),
            "static_feature_names": list(model.STATIC_FEATURES),
            "static_coefficients": np.asarray(model.static_coefficients).tolist(),
            "memory_coefficients": np.asarray(model.memory_coefficients).tolist(),
        }
    return output


def _render_rollout_examples(
    result_dir: Path,
    model: Any,
    validation: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        return {"status": "SKIPPED", "reason": repr(error)}
    completed = sorted(
        (row for row in selected_rows if row["status"] == "completed" and int(row["horizon_steps"]) == 1000),
        key=lambda row: float(row["rmse"]),
    )
    if not completed:
        return {"status": "SKIPPED", "reason": "no completed 10 s rollouts"}
    indices = sorted({0, len(completed) // 2, len(completed) - 1})
    chosen = [completed[index] for index in indices]
    by_id = {str(sample["event_id"]): sample for sample in validation}
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11.5, 3.0 * len(chosen)), squeeze=False)
    for row_index, row in enumerate(chosen):
        sample = by_id[str(row["event_id"])]
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        t = np.asarray(target["t"][:1000])
        pred_delta, pred_omega = model.rollout(
            static,
            t,
            {"Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:1000]), "interpolation": "linear"},
        )
        truth_delta = np.asarray(target["delta"][:1000])
        truth_omega = np.asarray(target["omega"][:1000])
        incidence = np.asarray(static["incidence"])
        ax = axes[row_index, 0]
        ax.plot(t, truth_omega, color="#222222", alpha=0.30, linewidth=0.8)
        ax.plot(t, pred_omega, color="#D55E00", alpha=0.30, linewidth=0.8, linestyle="--")
        ax.plot([], [], color="#222222", label="actual")
        ax.plot([], [], color="#D55E00", linestyle="--", label="prediction")
        ax.set_ylabel(r"$\omega$ (p.u.)")
        ax.set_title(f"{row['event_id']} | RMSE={float(row['rmse']):.4g}")
        ax.grid(alpha=0.2)
        ax = axes[row_index, 1]
        truth_edge = truth_delta @ incidence
        pred_edge = pred_delta @ incidence
        ax.plot(t, np.sqrt(np.mean(truth_edge**2, axis=1)), color="#0072B2", label="actual edge RMS")
        ax.plot(t, np.sqrt(np.mean(pred_edge**2, axis=1)), color="#E69F00", linestyle="--", label="predicted edge RMS")
        ax.set_ylabel("edge-angle RMS (rad)")
        ax.grid(alpha=0.2)
        if row_index == 0:
            axes[row_index, 0].legend(frameon=False, ncol=2)
            axes[row_index, 1].legend(frameon=False, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("time after clearing (s)")
    fig.suptitle(f"F1 frozen validation — {ROLLOUT_LABEL}", y=0.995)
    fig.tight_layout()
    png = result_dir / "selected_model_rollout_examples.png"
    svg = result_dir / "selected_model_rollout_examples.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"status": "completed", "event_ids": [row["event_id"] for row in chosen], "png": png.name, "svg": svg.name}


def _run_controlled(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    _validate_scope(config)
    hypothesis = _latest_result(root, str(config["hypothesis_experiment"]), "hypothesis_registration")
    if _verify_hashes(hypothesis):
        raise RuntimeError("v6 hypothesis artifacts changed before controlled experiment")
    registry = json.loads((hypothesis / "hypothesis_registry.json").read_text(encoding="utf-8"))
    if not registry.get("registered_before_response_metrics"):
        raise RuntimeError("v6 hypotheses were not preregistered")
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    data = _load_v5_data(root, config)
    row_by_id = {str(row["event_id"]): row for row in data["rows"]}
    qa_rows = [
        row
        for row in data["rows"]
        if row["role"] == "train" and bool(row["eligible_for_v5_selection"])
    ]
    roles = deterministic_fault_line_roles(
        qa_rows,
        seed=int(config["seed"]),
        fit_fraction=float(config["development_fit_fraction"]),
        selection_fraction=float(config["development_selection_fraction"]),
    )
    qa_ids = {str(row["event_id"]) for row in qa_rows}
    qa_train = [sample for sample in data["train"] if str(sample["event_id"]) in qa_ids]
    development_fit = [sample for sample in qa_train if roles[str(row_by_id[str(sample["event_id"])]["fault_line"])] == "fit"]
    development_selection = [sample for sample in qa_train if roles[str(row_by_id[str(sample["event_id"])]["fault_line"])] == "selection"]
    development_audit = [sample for sample in qa_train if roles[str(row_by_id[str(sample["event_id"])]["fault_line"])] == "audit"]
    coefficient_ids = {
        str(row["event_id"])
        for row in data["rows"]
        if row["role"] == "train"
        and bool(row["coefficient_fit_eligible"])
        and bool(row["eligible_h1000"])
    }
    coefficient_train = [sample for sample in data["train"] if str(sample["event_id"]) in coefficient_ids]
    checkpoints = result_dir / "checkpoints"
    checkpoints.mkdir()
    recovery = _validated_failed_development_cache(root, config_path, config)
    if recovery is None:
        development, development_metadata = _fit_stack(
            development_fit,
            development_selection,
            config,
            checkpoints,
            "development_raw10",
        )
        development["PSD-selected"] = _nonzero_psd_for_lattice(
            development["PSD-selected"], development_fit, config
        )
        development_cache = {
            "status": "RECOMPUTED",
            "cache_is_evidence": False,
        }
    else:
        development, development_metadata, development_cache = recovery
    development["B2"].save(checkpoints / "development_raw10-B2.npz")
    development["B3"].save(checkpoints / "development_raw10-B3.npz")
    development["PSD-selected"].save(
        checkpoints / "development_raw10-PSD-lattice.npz"
    )
    same_support, same_support_design, _ = _same_support_refits(
        development["PSD-selected"],
        development_fit,
        [*development_fit, *development_selection],
        config,
    )
    development_routes = {"B2": development["B2"], "B3": development["B3"], **same_support}
    horizons = tuple(int(value) for value in config["report_horizons"])
    development_rows = [
        *_evaluate_routes(
            development_routes,
            development_selection,
            development_selection,
            None,
            horizons,
            cohort="development_selection",
            representation="generator_10",
        ),
        *_evaluate_routes(
            development_routes,
            development_audit,
            development_audit,
            None,
            horizons,
            cohort="development_audit",
            representation="generator_10",
        ),
    ]
    development_aggregate = _detailed_aggregate(development_rows)
    complexity = _model_complexity(development_routes)
    selector_rows = [
        row
        for row in development_aggregate
        if row["cohort"] == "development_selection" and int(row["horizon_steps"]) == 1000
    ]
    selection = _select_minimum_branch(
        selector_rows,
        complexity,
        float(config["minimum_model_improvement"]),
    )
    if selection["status"] != "SELECTED":
        raise RuntimeError(f"v6 dynamic branch selection failed: {selection}")
    selected_route = str(selection["route"])
    audit_selected = next(
        row
        for row in development_aggregate
        if row["cohort"] == "development_audit"
        and row["branch"] == selected_route
        and int(row["horizon_steps"]) == 1000
    )
    selection["fault_line_disjoint_audit"] = audit_selected
    final_models, final_fit_rows = _fixed_family_refit(
        development,
        qa_train,
        coefficient_train,
        config,
    )
    validation = list(data["validation"])
    final_rows = _evaluate_routes(
        final_models,
        validation,
        validation,
        None,
        horizons,
        cohort="frozen_F1_validation",
        representation="generator_10",
    )
    final_aggregate = _detailed_aggregate(final_rows)
    source_aggregate: list[dict[str, Any]] = []
    for source in sorted({str(row["source_dataset"]) for row in data["rows"] if row["role"] == "validation"}):
        scoped = [row for row in final_rows if str(row["source_dataset"]) == source]
        scoped_copy = [{**row, "cohort": f"frozen_validation::{source}"} for row in scoped]
        source_aggregate.extend(_detailed_aggregate(scoped_copy))
    selected_model = final_models[selected_route]
    selected_validation_rows = [row for row in final_rows if row["branch"] == selected_route]
    comparison_rows = []
    selected_h1000 = [row for row in selected_validation_rows if int(row["horizon_steps"]) == 1000]
    for comparator_route in sorted(final_models):
        if comparator_route == selected_route:
            continue
        comparator = [row for row in final_rows if row["branch"] == comparator_route and int(row["horizon_steps"]) == 1000]
        comparison_rows.append(
            {
                "reference_route": selected_route,
                "comparator_route": comparator_route,
                **_paired_cluster_bootstrap(
                    selected_h1000,
                    comparator,
                    draws=int(config["performance_bootstrap_draws"]),
                    confidence=float(config["bootstrap_confidence"]),
                    seed=int(config["seed"]) + 17 * len(comparison_rows),
                ),
                "sign_flip": _sign_flip_test(
                    selected_h1000,
                    comparator,
                    seed=int(config["seed"]) + 117 * len(comparison_rows),
                ),
            }
        )
    modal_rows = [
        *_modal_match_rows(
            final_models["B2"], validation, (500, 1000), branch="B2", population="frozen_F1_validation"
        ),
        *_modal_match_rows(
            selected_model,
            validation,
            (500, 1000),
            branch=selected_route,
            population="frozen_F1_validation",
        ),
    ]
    convergence = {}
    for sample in validation[:3]:
        target = sample["training_context"]["targets"]
        steps = min(1000, len(target["t"]))
        convergence[str(sample["event_id"])] = selected_model.convergence_audit(
            sample["model_inputs"]["static"],
            np.asarray(target["t"][:steps]),
            {"Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]), "interpolation": "linear"},
        )
    residual_rows = [
        {
            "route": selected_route,
            **_residual_features(selected_model, sample, sample, None, 1000),
        }
        for sample in validation
        if len(sample["training_context"]["targets"]["t"]) >= 1000
    ]
    full_psd = final_models["SS-CDM"]
    edge_support, module_support, support_summary = _support_bootstrap(full_psd, qa_train, config)
    subset_summary, subset_rows, subset_benchmark, subset_models = _subset_experiment(
        data,
        row_by_id,
        development,
        selected_route,
        config,
    )
    subset_aggregate = _detailed_aggregate(subset_rows) if subset_rows else []
    physics = {route: model.physics_audit() for route, model in final_models.items()}
    try:
        figure = _render_rollout_examples(
            result_dir, selected_model, validation, selected_validation_rows
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        figure = {"status": "FAILED_OPTIONAL", "reason": repr(error)}
    for route, model in final_models.items():
        model.save(checkpoints / f"final-{route}.npz")
    for route, model in subset_models.items():
        model.save(checkpoints / f"subset-{route}.npz")
    selection_marker = {
        "run_id": run_id,
        "selected_route": selected_route,
        "selection": selection,
        "global_spatial_representation": "generator_10",
        "validation_did_not_change_selection": True,
        "Q_B_used_for_structure": False,
        "Q_B_used_after_structure_freeze": True,
        "rollout_label": ROLLOUT_LABEL,
        "test_ids_read": [],
        "other_folds_read": [],
    }
    _json(result_dir / "selection_marker.json", selection_marker)
    _csv(result_dir / "data_qualification.csv", data["rows"])
    _csv(
        result_dir / "fault_line_development_roles.csv",
        [
            {
                "fault_line": line,
                "role": role,
                "source_dataset": next(row["source_dataset"] for row in qa_rows if str(row["fault_line"]) == line),
            }
            for line, role in sorted(roles.items())
        ],
    )
    _csv(result_dir / "development_fit_metadata.csv", development_metadata)
    _csv(result_dir / "development_same_support_design.csv", same_support_design)
    _csv(result_dir / "development_per_case_metrics.csv", development_rows)
    _csv(result_dir / "development_aggregate_metrics.csv", development_aggregate)
    _csv(result_dir / "final_fit_diagnostics.csv", final_fit_rows)
    _csv(result_dir / "validation_per_case_metrics.csv", final_rows)
    _csv(result_dir / "validation_aggregate_metrics.csv", final_aggregate)
    _csv(result_dir / "validation_source_metrics.csv", source_aggregate)
    _csv(result_dir / "modal_matches.csv", modal_rows)
    _csv(result_dir / "residual_per_case.csv", residual_rows)
    _csv(result_dir / "performance_cluster_bootstrap.csv", comparison_rows)
    _csv(result_dir / "edge_support_bootstrap.csv", edge_support)
    _csv(result_dir / "module_support_bootstrap.csv", module_support)
    _csv(result_dir / "subset_per_case_metrics.csv", subset_rows)
    _csv(result_dir / "subset_aggregate_metrics.csv", subset_aggregate)
    _json(result_dir / "support_bootstrap_summary.json", support_summary)
    _json(result_dir / "integration_convergence.json", convergence)
    _json(result_dir / "physics_audits.json", physics)
    _json(result_dir / "system_information.json", _system_information(selected_model, selected_route))
    _json(result_dir / "subset_experiment.json", subset_summary)
    _json(result_dir / "subset_runtime_benchmark.json", subset_benchmark)
    _json(result_dir / "rollout_figure_manifest.json", figure)
    _json(result_dir / "development_cache_audit.json", development_cache)
    _json(result_dir / "stratification_policy.json", data["policy"].as_dict())
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "controlled_experiment",
            "hypothesis_result": hypothesis.name,
            "hypothesis_sha256": _sha256(hypothesis / "hypothesis_registry.json"),
            "dataset_revision": data["store"].dataset_metadata.dataset_revision,
            "canonical_data_mutated": False,
            "v1_v5_result_artifacts_read": [],
            "failed_run_development_cache": development_cache,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _run_residual(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    if _verify_hashes(controlled):
        raise RuntimeError("controlled artifacts changed before residual diagnosis")
    selection = json.loads((controlled / "selection_marker.json").read_text(encoding="utf-8"))
    rows = _read_csv(controlled / "residual_per_case.csv")
    completed = [row for row in rows if row.get("status") == "completed"]
    summary = {
        "run_id": run_id,
        "selected_route": selection["selected_route"],
        "completed_cases": len(completed),
        "mean_omega_residual_rmse": float(np.mean([_float(row["omega_residual_rmse"]) for row in completed])) if completed else float("nan"),
        "median_lag1_autocorrelation": float(np.median([_float(row["omega_residual_lag1"]) for row in completed])) if completed else float("nan"),
        "median_lag20_autocorrelation": float(np.median([_float(row["omega_residual_lag20"]) for row in completed])) if completed else float("nan"),
        "median_residual_stable_rank": float(np.median([_float(row["omega_residual_stable_rank"]) for row in completed])) if completed else float("nan"),
        "median_low_frequency_fraction": float(np.median([_float(row["omega_residual_low_frequency_fraction"]) for row in completed])) if completed else float("nan"),
        "diagnostic_interpretation": (
            "persistent autocorrelation or low-frequency energy is evidence of unresolved dynamics, "
            "not permission to change the frozen v6 branch after validation"
        ),
        "test_ids_read": [],
    }
    _json(result_dir / "residual_diagnosis.json", summary)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "residual_diagnosis",
            "controlled_result": controlled.name,
            "controlled_hash_manifest": _sha256(controlled / "artifact_hashes.json"),
            "test_ids_read": [],
        },
    )
    _write_hashes(result_dir)


def _run_adversarial(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    if _verify_hashes(controlled) or _verify_hashes(residual):
        raise RuntimeError("v6 parent artifacts changed before adversarial review")
    marker = json.loads((controlled / "selection_marker.json").read_text(encoding="utf-8"))
    subset = json.loads((controlled / "subset_experiment.json").read_text(encoding="utf-8"))
    roles = _read_csv(controlled / "fault_line_development_roles.csv")
    data_rows = _read_csv(controlled / "data_qualification.csv")
    validation_metrics = _read_csv(controlled / "validation_aggregate_metrics.csv")
    role_by_line = defaultdict(set)
    for row in roles:
        role_by_line[row["fault_line"]].add(row["role"])
    selected_h1000 = next(
        row
        for row in validation_metrics
        if row["branch"] == marker["selected_route"]
        and row["cohort"] == "frozen_F1_validation"
        and int(row["horizon_steps"]) == 1000
    )
    checks = {
        "controlled_artifacts_immutable": not _verify_hashes(controlled),
        "hypothesis_precedes_metrics": json.loads((controlled / "lineage.json").read_text(encoding="utf-8"))["hypothesis_result"].startswith(str(config["hypothesis_experiment"])),
        "development_fault_lines_disjoint": all(len(values) == 1 for values in role_by_line.values()),
        "Q_B_excluded_from_structure": bool(marker["Q_B_used_for_structure"] is False),
        "validation_frozen_after_selection": bool(marker["validation_did_not_change_selection"]),
        "test_locked": marker["test_ids_read"] == [],
        "other_folds_locked": marker["other_folds_read"] == [],
        "observed_pm_named_conditional": marker["rollout_label"] == ROLLOUT_LABEL,
        "selected_validation_zero_failures": _float(selected_h1000["failure_rate"]) == 0.0,
        "subset_rule_source_blind": bool(subset.get("subset_rule", {}).get("source_label_used") is False),
        "subset_not_promoted_to_global": "not an online router" in str(subset.get("claim_scope", "")),
    }
    alternative_explanations = [
        {
            "threat": "dynamic closure may absorb source heterogeneity rather than missing state physics",
            "disposition": "retained; single F1 and mixed sources prevent unique device semantics",
        },
        {
            "threat": "observed Pm supplies future forcing and inflates deployable accuracy",
            "disposition": "retained; every result is conditional rollout with observed Pm",
        },
        {
            "threat": "subset clustering gain may be regularization rather than physical coherency",
            "disposition": "tested by raw10 subset refit, full-state error, quotient error and runtime; no universal claim",
        },
        {
            "threat": "six-percent subset UQG budget was informed by exploratory probing",
            "disposition": "explicitly marked pilot-informed; subset result is exploratory rather than confirmatory",
        },
        {
            "threat": "validation includes only a few A-low/R-low fault lines",
            "disposition": "cluster superiority requires positive fault-line CI; otherwise returns INSUFFICIENT_INFORMATION",
        },
        {
            "threat": "DMD poles of nonlinear forced responses are not autonomous eigenvalues",
            "disposition": "reported as empirical ringdown modes only",
        },
    ]
    review = {
        "run_id": run_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "alternative_explanations": alternative_explanations,
        "data_population": {
            "train_QA": sum(row["role"] == "train" and row["quality_grade"] == "Q-A" for row in data_rows),
            "train_QB": sum(row["role"] == "train" and row["quality_grade"] == "Q-B" for row in data_rows),
            "validation": sum(row["role"] == "validation" for row in data_rows),
        },
        "subset_status": subset.get("status"),
        "test_ids_read": [],
    }
    _json(result_dir / "adversarial_review.json", review)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "adversarial_review",
            "controlled_result": controlled.name,
            "residual_result": residual.name,
            "test_ids_read": [],
        },
    )
    _write_hashes(result_dir)
    if review["status"] != "PASS":
        raise RuntimeError("v6 adversarial review failed")


def _modal_summary(rows: Sequence[dict[str, str]], branch: str, horizon: int) -> dict[str, float]:
    selected = [row for row in rows if row["branch"] == branch and int(row["horizon_steps"]) == horizon]
    return {
        "matched_modes": len(selected),
        "median_reference_frequency_hz": float(np.median([_float(row["reference_frequency_hz"]) for row in selected])) if selected else float("nan"),
        "median_candidate_frequency_hz": float(np.median([_float(row["candidate_frequency_hz"]) for row in selected])) if selected else float("nan"),
        "median_relative_frequency_error": float(np.median([_float(row["relative_frequency_error"]) for row in selected])) if selected else float("nan"),
        "median_absolute_damping_error": float(np.median([_float(row["absolute_damping_ratio_error"]) for row in selected])) if selected else float("nan"),
        "median_MAC": float(np.median([_float(row["mac"]) for row in selected])) if selected else float("nan"),
    }


def _write_research_report(
    result_dir: Path,
    controlled: Path,
    marker: dict[str, Any],
    selected_metrics: list[dict[str, str]],
    b2_metrics: list[dict[str, str]],
    modal: dict[str, Any],
    subset: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    by_horizon = {int(row["horizon_steps"]): row for row in selected_metrics}
    b2_by_horizon = {int(row["horizon_steps"]): row for row in b2_metrics}
    lines = [
        "# M0-AFSD v6：完整10机动态等值与条件聚类验证研究报告",
        "",
        f"证据运行：`{controlled.name}`。本报告仅覆盖 F1、seed 324、train/validation，未读取 test 或 F2–F5。所有重积分均为 **{ROLLOUT_LABEL}**。",
        "",
        "## 摘要",
        "",
        f"v5 已证明全体合格训练扰动在2%统一商空间门下需要原始10机。v6 因而固定10机空间，在训练集内部选择 B2、完整网格 M0-3EG 与同支撑保守/耗散/记忆模块。最终训练选择为 `{marker['selected_route']}`。",
        f"冻结验证的模型命名为 **{acceptance['model_reporting_name']}**；该名称不等同于真实设备阶次或39母线拓扑恢复。",
        "",
        "## 方法",
        "",
        "M0 通过多视界积分弱形式一次求解非负阻尼与完全图耦合；M0-3EG 在其上增加因果准稳态初始化的节点电气储能；端口结构闭合进一步在边上搜索势能、耗散与稳定记忆。开发拟合、分枝选择和独立训练审计按 fault line 分离，F1 validation 只在规则冻结后确认。Q-A 决定阶次、模块和支撑；Q-B 仅在支撑冻结后参与系数重拟合。",
        "",
        "模型选择采用字典序原则：数值与物理可行、10秒误差、2%等价区间内最小复杂度。未入选模块只能解释为当前扰动没有提供其必要性证据。",
        "",
        "## 冻结验证重积分结果",
        "",
        "| 视界 | 选定模型 mean RMSE | median | pooled | worst 10% CVaR | ω RMSE | 边角差 RMSE | B2 mean RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in (100, 500, 1000):
        row = by_horizon[horizon]
        b2 = b2_by_horizon[horizon]
        lines.append(
            f"| {horizon / 100:.0f} s | {_float(row['mean_rmse']):.6f} | {_float(row['median_rmse']):.6f} | {_float(row['pooled_rmse']):.6f} | {_float(row['worst_10pct_cvar_rmse']):.6f} | {_float(row['mean_omega_rmse']):.6f} | {_float(row['mean_edge_angle_rmse']):.6f} | {_float(b2['mean_rmse']):.6f} |"
        )
    lines.extend(
        [
            "",
            "以上 RMSE 在原始10机状态坐标中计算；失败轨迹不记为零。频率和边角差另列，避免混合量纲的总RMSE掩盖局部误差。",
            "",
            "## 振荡与系统信息",
            "",
            f"10秒经验 DMD 匹配模态数为 {modal['matched_modes']}；参考主频中位数 {modal['median_reference_frequency_hz']:.4f} Hz，预测中位数 {modal['median_candidate_frequency_hz']:.4f} Hz；频率相对误差中位数 {modal['median_relative_frequency_error']:.3f}，阻尼比绝对误差中位数 {modal['median_absolute_damping_error']:.3f}，MAC 中位数 {modal['median_MAC']:.3f}。这些是受迫非线性轨迹的经验环降描述，不宣称为自治系统精确特征值。",
            "",
            f"模态绝对门：`{acceptance['modal_absolute_gate']}`。参数、模块、时间常数和边支撑见 `system_information.json`、`edge_support_bootstrap.csv` 与 `module_support_bootstrap.csv`。",
            "",
            "## 条件聚类实验",
            "",
            f"子集规则为 Q-A、{subset.get('subset_rule', {}).get('amplitude_level')}、{subset.get('subset_rule', {}).get('richness_level')}，不使用 source 标签。训练得到分区 `{subset.get('partition')}`，训练证书合并数 {subset.get('certified_training_merges')}，最小性状态 `{subset.get('minimality_status')}`。",
            f"10秒原始10机子集重拟合 RMSE 为 {_float(subset.get('raw10_mean_rmse_10s')):.6f}，聚类模型为 {_float(subset.get('cluster_mean_rmse_10s')):.6f}，相对改进 {_float(subset.get('relative_improvement')) * 100:.2f}%，最终判断 `{subset.get('status')}`。",
            "",
            "该子集容差为探索性、受先导审计启发的6%，不能与全局2%充分性门混同。即使聚类在该子集上更优，也只说明对这一离线扰动层存在有利的偏差—方差权衡，不证明真实系统只有更少节点，也不能作为在线未来轨迹路由器。",
            "",
            "## 审计结论与边界",
            "",
            f"Acceptance：`{acceptance['status']}`。COI、积分收敛和物理硬门状态见 acceptance.json。当前证据不包括 test、其他fold、自主 Pm ODE、第二个独立系统或真实设备参数恢复。",
            "",
            "## 证据索引",
            "",
            "- `selection_marker.json`：冻结分枝与数据角色；",
            "- `validation_aggregate_metrics.csv`：1/5/10秒主结果；",
            "- `modal_matches.csv`：逐事件经验模态；",
            "- `system_information.json`：显式参数和结构；",
            "- `subset_experiment.json`：条件聚类结论；",
            "- `selected_model_rollout_examples.png`：实际与预测轨迹。",
        ]
    )
    (result_dir / "research_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_acceptance(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    adversarial = _latest_result(root, str(config["adversarial_experiment"]), "adversarial_review")
    if _verify_hashes(controlled) or _verify_hashes(residual) or _verify_hashes(adversarial):
        raise RuntimeError("v6 parent artifacts changed before acceptance")
    marker = json.loads((controlled / "selection_marker.json").read_text(encoding="utf-8"))
    physics = json.loads((controlled / "physics_audits.json").read_text(encoding="utf-8"))
    convergence = json.loads((controlled / "integration_convergence.json").read_text(encoding="utf-8"))
    subset = json.loads((controlled / "subset_experiment.json").read_text(encoding="utf-8"))
    aggregate = _read_csv(controlled / "validation_aggregate_metrics.csv")
    modal_rows = _read_csv(controlled / "modal_matches.csv")
    selected_route = marker["selected_route"]
    selected_metrics = [row for row in aggregate if row["cohort"] == "frozen_F1_validation" and row["branch"] == selected_route]
    b2_metrics = [row for row in aggregate if row["cohort"] == "frozen_F1_validation" and row["branch"] == "B2"]
    h1000 = next(row for row in selected_metrics if int(row["horizon_steps"]) == 1000)
    modal = _modal_summary(modal_rows, selected_route, 1000)
    modal_absolute = bool(
        modal["matched_modes"] > 0
        and modal["median_relative_frequency_error"] <= 0.10
        and modal["median_absolute_damping_error"] <= 0.05
        and modal["median_MAC"] >= 0.80
    )
    convergence_pass = bool(convergence) and all(bool(row.get("convergent")) for row in convergence.values())
    gates = {
        "selected_branch_frozen_before_validation": bool(marker["validation_did_not_change_selection"]),
        "validation_failure_rate_zero": _float(h1000["failure_rate"]) == 0.0,
        "COI_drift_below_limit": _float(h1000["maximum_coi_drift"]) <= float(config["maximum_coi_drift"]),
        "RK4_convergent": convergence_pass,
        "physics_gate": physics[selected_route]["status"] == "pass",
        "fault_line_audit_completed": marker["selection"]["fault_line_disjoint_audit"]["failure_rate"] == 0.0,
        "test_locked": marker["test_ids_read"] == [],
        "other_folds_locked": marker["other_folds_read"] == [],
        "subset_question_resolved_honestly": subset.get("status") in {"CLUSTER_BETTER", "CLUSTER_NOT_BETTER", "INSUFFICIENT_INFORMATION", "NO_CERTIFIED_SUBSET_MERGE"},
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    reporting_name = selected_route if modal_absolute else "rollout-accurate dynamic equivalent"
    acceptance = {
        "run_id": run_id,
        "status": status,
        "terminal_state": "FULL_MODEL_AND_SUBSET_COMPARISON_COMPLETE" if status == "PASS" else "MODEL_CLASS_INADEQUATE",
        "selected_route": selected_route,
        "model_reporting_name": reporting_name,
        "modal_absolute_gate": "PASS" if modal_absolute else "FAIL",
        "modal_summary_10s": modal,
        "validation_10s": h1000,
        "subset_status": subset.get("status"),
        "gates": gates,
        "claim_boundary": ROLLOUT_LABEL,
        "test_ids_read": [],
        "merge_gate_required": True,
    }
    _json(result_dir / "acceptance.json", acceptance)
    _write_research_report(
        result_dir,
        controlled,
        marker,
        selected_metrics,
        b2_metrics,
        modal,
        subset,
        acceptance,
    )
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "acceptance",
            "controlled_result": controlled.name,
            "residual_result": residual.name,
            "adversarial_result": adversarial.name,
            "test_ids_read": [],
        },
    )
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError("v6 acceptance failed")


def run(*, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str) -> None:
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _validate_scope(config)
    if stage == "hypothesis_registration":
        _run_hypothesis(config_path, config, result_dir, run_id)
    elif stage == "controlled_experiment":
        _run_controlled(root, config_path, config, result_dir, run_id)
    elif stage == "residual_diagnosis":
        _run_residual(root, config, result_dir, run_id)
    elif stage == "adversarial_review":
        _run_adversarial(root, config, result_dir, run_id)
    elif stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported v6 stage: {stage}")
