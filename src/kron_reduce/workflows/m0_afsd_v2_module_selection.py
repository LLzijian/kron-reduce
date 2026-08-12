from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from itertools import product
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
from kron_reduce.models import (
    M0Model,
    PortStructuredDiscoveryM0Model,
    TransientEmfM0Model,
)
from kron_reduce.physics import (
    aggregate_model_sample,
    discover_nested_partitions,
    reconstruct_model_state,
)
from kron_reduce.workflows.adaptive_fidelity_structure_discovery import (
    _aggregate,
    _context,
    _evaluate_candidate,
    _source,
    _stress_thresholds,
)

MODULES = ("conservative", "dissipative", "memory")


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
    artifacts = {
        str(path.relative_to(result_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)


def _verify_hashes(result_dir: Path) -> list[str]:
    manifest = json.loads((result_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        relative
        for relative, expected in manifest.items()
        if not (result_dir / relative).is_file()
        or _sha256(result_dir / relative) != expected
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    prefix = f"{experiment}-{stage}-"
    candidates = sorted(
        (
            path
            for path in (root / "results").glob(f"{prefix}*")
            if path.is_dir() and not (path / "RUN_FAILED.json").exists()
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"no completed v2 result for {experiment}/{stage}")
    result = candidates[-1]
    if "m0_afsd_v2" not in result.name or "m0_afsd_f1_seed324-diagnostic" in result.name:
        raise RuntimeError("v2 may not consume a v1 result")
    return result


def _validate_scope(config: dict[str, Any]) -> None:
    if (
        config.get("protocol_revision") != "m0_afsd_v2_module_selection"
        or config.get("test_access") != "forbidden"
        or config.get("pm_mode") != "observed"
        or config.get("fold") != "F1"
        or int(config.get("seed", -1)) != 324
    ):
        raise RuntimeError("v2 requires F1/seed324, observed Pm, and locked test")


def _hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "protocol": "m0_afsd_v2_module_selection",
        "round": 1,
        "registered_before_response_metrics": True,
        "scope": {
            "fold": "F1",
            "seed": 324,
            "test_access": "forbidden",
            "Pm_policy": "observed_schedule_conditional_rollout",
        },
        "hypotheses": [
            {
                "id": "H1",
                "statement": "B2 or inactive M0-3E is insufficient for C3 long-horizon dynamics",
                "falsifier": "a simpler model lies within 2% of the best 10-second RMSE",
            },
            {
                "id": "H2",
                "statement": "at least one port-structured closure module is necessary",
                "falsifier": "its same-support knockout is within 2% or its cluster-bootstrap CI crosses zero",
            },
            {
                "id": "H3",
                "statement": "a C3-trained global model remains finite on C1 and C2",
                "falsifier": "any applicable cross-layer rollout fails or is non-finite",
            },
            {
                "id": "H4",
                "statement": "claimed state blocks and edges are repeatable across train fault-line clusters",
                "falsifier": "bootstrap support is below the predeclared 80% threshold",
            },
        ],
        "controlled_factors": {
            "module_lattice": [
                [
                    module
                    for module, enabled in zip(MODULES, flags, strict=True)
                    if enabled
                ]
                for flags in product((False, True), repeat=3)
            ],
            "same_support_envelope": True,
            "support_preserving_refit": True,
            "fixed_integrator": "COI-preserving RK4 max step 0.01 s",
            "validation_horizons": list(config["report_horizons"]),
        },
        "gates": {
            "minimum_relative_improvement": float(config["minimum_model_improvement"]),
            "bootstrap_draws": int(config["bootstrap_draws"]),
            "bootstrap_confidence": float(config["bootstrap_confidence"]),
            "minimum_module_support": float(config["minimum_module_bootstrap_support"]),
            "minimum_edge_support": float(config["minimum_edge_bootstrap_support"]),
            "maximum_coi_drift": float(config["maximum_coi_drift"]),
            "failure_rate": 0.0,
        },
        "claim_boundaries": config["claim_boundaries"],
        "next_experiments_by_information_gain": [
            "fixed-support bootstrap consensus refit",
            "expand stable memory time-scale grid",
            "retain one mass-orthogonal intra-cluster mode",
        ],
    }


def _run_hypothesis_registration(
    config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    _validate_scope(config)
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    registry = _hypothesis_registry(config, run_id)
    _json(result_dir / "hypothesis_registry.json", registry)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "hypothesis_registration",
            "protocol": "m0_afsd_v2_module_selection",
            "v1_results_read": [],
            "test_ids_read": [],
            "config_sha256": _sha256(config_path),
        },
    )
    _write_hashes(result_dir)


def _load_data(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold="F1",
    )
    minimum = int(config["minimum_real_steps"])
    train_ids = [
        event_id
        for event_id in split.train_ids
        if store.metadata(event_id).valid_length >= minimum
    ]
    val_ids = [
        event_id
        for event_id in split.val_ids
        if store.metadata(event_id).valid_length >= minimum
    ]
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
    train_cohorts, train_policy = cumulative_cohorts(
        raw_train,
        train_strata,
        activation_edges=policy.activation_edges,
        minimum_cases=int(config["minimum_selection_cases"]),
        allow_train_fallback=True,
    )
    val_cohorts, val_policy = cumulative_cohorts(
        raw_val,
        val_strata,
        activation_edges=policy.activation_edges,
        minimum_cases=1,
        allow_train_fallback=False,
    )
    return {
        "store": store,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "raw_train": raw_train,
        "raw_val": raw_val,
        "train_strata": train_strata,
        "val_strata": val_strata,
        "policy": policy,
        "train_cohorts": train_cohorts,
        "val_cohorts": val_cohorts,
        "cohort_policy": {"train": train_policy, "val": val_policy},
    }


def _discover_partition(
    cohort: list[dict[str, Any]], strata: dict[str, dict[str, Any]], config: dict[str, Any]
) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any]]:
    eligible = [
        sample
        for sample in cohort
        if strata[sample["event_id"]]["structure_selection_eligible"]
    ]
    if len(eligible) < int(config["minimum_selection_cases"]):
        eligible = cohort
    count = int(config["spatial_cluster_count"])
    hierarchy, diagnostics = discover_nested_partitions(
        eligible,
        cluster_counts=(count,),
        maximum_steps=int(config["maximum_analysis_steps"]),
    )
    return hierarchy[count], diagnostics


def _fit_stack(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint_dir: Path,
    cohort: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = _context(config, train)
    b2 = M0Model(max_internal_step=float(config["max_internal_step"]))
    b2_fit = b2.fit(train, val, context)
    b2.save(checkpoint_dir / f"{cohort}-B2.npz")
    b3 = TransientEmfM0Model(
        anchor=deepcopy(b2),
        alpha_candidates=tuple(float(value) for value in config["b3_alpha_candidates"]),
        tau_candidates=tuple(float(value) for value in config["b3_tau_candidates"]),
        activation_threshold_candidates=_stress_thresholds(train),
        activation_width=float(config["b3_activation_width"]),
        max_source_loss_degradation=float(config["maximum_source_loss_degradation"]),
        max_internal_step=float(config["max_internal_step"]),
    )
    b3_fit = b3.fit(train, val, context)
    b3.save(checkpoint_dir / f"{cohort}-B3.npz")
    psd = PortStructuredDiscoveryM0Model(
        backbone=deepcopy(b3),
        tau_mode_candidates=tuple(
            tuple(float(value) for value in row) for row in config["psd_tau_mode_candidates"]
        ),
        ridge_values=tuple(float(value) for value in config["psd_ridge_values"]),
        group_thresholds=tuple(float(value) for value in config["psd_group_thresholds"]),
        coefficient_scales=tuple(float(value) for value in config["psd_coefficient_scales"]),
        minimum_validation_improvement=float(config["minimum_model_improvement"]),
        maximum_source_loss_degradation=float(config["maximum_source_loss_degradation"]),
        max_internal_step=float(config["max_internal_step"]),
    )
    psd_fit = psd.fit(train, val, context)
    psd.save(checkpoint_dir / f"{cohort}-PSD.npz")
    metadata = [
        {
            "training_cohort": cohort,
            "route": "B2",
            "fit_validation_loss": b2_fit.best_validation_loss,
            "physics_status": b2.physics_audit()["status"],
            "selected_ridge": b2.selected_ridge,
        },
        {
            "training_cohort": cohort,
            "route": "B3",
            "fit_validation_loss": b3_fit.best_validation_loss,
            "physics_status": b3.physics_audit()["status"],
            "selected_alpha": b3.alpha,
            "selected_tau": b3.tau,
            "selected_is_B2_fallback": b3.diagnostics["selected_is_M0_fallback"],
        },
        {
            "training_cohort": cohort,
            "route": "PSD-selected",
            "fit_validation_loss": psd_fit.best_validation_loss,
            "physics_status": psd.physics_audit()["status"],
            "promoted": psd.diagnostics["promoted"],
            "selected_taus": psd.diagnostics["selected_fit_taus"],
            "selected_support": psd.diagnostics["selected_support"],
        },
    ]
    return {"B2": b2, "B3": b3, "PSD-selected": psd}, metadata


def _theta(model: PortStructuredDiscoveryM0Model, n_modes: int) -> np.ndarray:
    static = np.asarray(model.static_coefficients, dtype=float).reshape(-1)
    memory = np.asarray(model.memory_coefficients, dtype=float)[:n_modes].reshape(-1)
    return np.concatenate((static, memory))


def _module_mask(n_edges: int, n_modes: int, modules: set[str]) -> np.ndarray:
    mask = np.zeros(len(PortStructuredDiscoveryM0Model.STATIC_FEATURES) * n_edges + n_modes * n_edges, bool)
    if "conservative" in modules:
        mask[: 3 * n_edges] = True
    if "dissipative" in modules:
        mask[3 * n_edges : 5 * n_edges] = True
    if "memory" in modules and n_modes:
        mask[5 * n_edges :] = True
    return mask


def _module_code(modules: set[str]) -> str:
    symbols = {"conservative": "C", "dissipative": "D", "memory": "M"}
    return "SS-" + ("".join(symbols[name] for name in MODULES if name in modules) or "zero")


def _same_support_refits(
    fitted: PortStructuredDiscoveryM0Model,
    train: list[dict[str, Any]],
    audit_samples: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, PortStructuredDiscoveryM0Model], list[dict[str, Any]], dict[str, Any]]:
    selected = fitted.diagnostics["selected_validation"]
    fit_taus = tuple(float(value) for value in selected["fit_taus"])
    n_edges = int(selected["n_edges"])
    context = _context(config, train)
    prepared = fitted._prepare_weak_paths(
        train,
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
    )
    stats = fitted._build_normal_equations(
        prepared, fit_taus, windows=context.integral_windows
    )
    full_theta = _theta(fitted, len(fit_taus))
    full_support = full_theta > max(float(np.max(full_theta)) * 1e-12, np.finfo(float).eps)
    ridge = float(selected["ridge"])
    scale = float(selected["coefficient_scale"])
    models: dict[str, PortStructuredDiscoveryM0Model] = {}
    rows = []
    for flags in product((False, True), repeat=3):
        modules = {
            module for module, enabled in zip(MODULES, flags, strict=True) if enabled
        }
        active = full_support & _module_mask(n_edges, len(fit_taus), modules)
        theta, _, solver = fitted._factorized_nonnegative_ridge(
            stats["gram"],
            stats["response"],
            stats["target_energy"],
            stats["effective_blocks"],
            ridge,
            active,
        )
        theta = theta * scale
        candidate = deepcopy(fitted)
        candidate._set_candidate(
            {"n_edges": n_edges, "fit_taus": fit_taus, "theta": theta}
        )
        route = _module_code(modules)
        candidate.diagnostics = {
            **candidate.diagnostics,
            "same_support_ablation": True,
            "support_envelope_sha256": hashlib.sha256(full_support.tobytes()).hexdigest(),
            "enabled_modules": sorted(modules),
            "active_columns_allowed": int(np.sum(active)),
            "active_coefficients_after_refit": int(np.sum(theta > 0)),
            "fixed_ridge": ridge,
            "fixed_coefficient_scale": scale,
            "fixed_fit_taus": list(fit_taus),
            "refit_solver": solver,
        }
        candidate.diagnostics["selected_support"] = candidate._support_from_theta(
            theta, n_edges, len(fit_taus)
        )
        candidate.diagnostics["storage_audit"] = candidate._conservative_storage_audit(
            audit_samples
        )
        models[route] = candidate
        rows.append(
            {
                "route": route,
                "enabled_modules": sorted(modules),
                "module_count": len(modules),
                "support_envelope_sha256": candidate.diagnostics["support_envelope_sha256"],
                "support_envelope_columns": int(np.sum(full_support)),
                "active_columns_allowed": int(np.sum(active)),
                "active_coefficients_after_refit": int(np.sum(theta > 0)),
                "ridge": ridge,
                "coefficient_scale": scale,
                "fit_taus": list(fit_taus),
                "physics_status": candidate.physics_audit()["status"],
                **solver,
            }
        )
    return models, rows, stats


def _combine_stats(stats: list[tuple[dict[str, Any], int]]) -> dict[str, Any]:
    if not stats:
        raise ValueError("no weak statistics to combine")
    template = stats[0][0]
    output = {
        "gram": np.zeros_like(template["gram"]),
        "response": np.zeros_like(template["response"]),
        "target_energy": 0.0,
        "effective_blocks": 0,
        "n_scalar_equations": 0,
        "n_parameters": template["n_parameters"],
        "n_static_parameters": template["n_static_parameters"],
        "n_memory_parameters": template["n_memory_parameters"],
        "n_edges": template["n_edges"],
        "n_memory_modes": template["n_memory_modes"],
    }
    for row, multiplicity in stats:
        output["gram"] += multiplicity * row["gram"]
        output["response"] += multiplicity * row["response"]
        output["target_energy"] += multiplicity * row["target_energy"]
        output["effective_blocks"] += multiplicity * row["effective_blocks"]
        output["n_scalar_equations"] += multiplicity * row["n_scalar_equations"]
    return output


def _support_bootstrap(
    model: PortStructuredDiscoveryM0Model,
    train: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected = model.diagnostics["selected_validation"]
    fit_taus = tuple(float(value) for value in selected["fit_taus"])
    ridge = float(selected["ridge"])
    threshold = float(selected["relative_group_threshold"])
    scale = float(selected["coefficient_scale"])
    context = _context(config, train)
    by_line: dict[str, list[dict[str, Any]]] = {}
    for sample in train:
        line = str(sample["training_context"]["metadata"]["fault_line"])
        by_line.setdefault(line, []).append(sample)
    cluster_stats = {}
    for line, samples in by_line.items():
        prepared = model._prepare_weak_paths(
            samples,
            horizon_steps=context.horizon_steps,
            max_fit_steps=context.fit_max_steps,
        )
        cluster_stats[line] = model._build_normal_equations(
            prepared, fit_taus, windows=context.integral_windows
        )
    lines = sorted(cluster_stats)
    draws = int(config["bootstrap_draws"])
    rng = np.random.default_rng(int(config["seed"]))
    n_edges = int(selected["n_edges"])
    edge_count = np.zeros(n_edges, dtype=int)
    memory_edge_count = np.zeros(n_edges, dtype=int)
    module_count = {module: 0 for module in MODULES}
    jaccard = []
    selected_edges = set(model.diagnostics["selected_support"]["edge_indices"])
    for _ in range(draws):
        counts = Counter(rng.choice(lines, size=len(lines), replace=True).tolist())
        stats = _combine_stats([(cluster_stats[line], count) for line, count in counts.items()])
        theta, _ = model._group_prune_and_refit(stats, ridge, threshold)
        theta *= scale
        support = model._support_from_theta(theta, n_edges, len(fit_taus))
        edges = set(support["edge_indices"])
        edge_count[list(edges)] += 1
        n_static = len(model.STATIC_FEATURES) * n_edges
        static = theta[:n_static].reshape(len(model.STATIC_FEATURES), n_edges)
        memory = theta[n_static:].reshape(len(fit_taus), n_edges) if fit_taus else np.empty((0, n_edges))
        memory_edges = np.flatnonzero(np.any(memory > 0, axis=0)) if fit_taus else np.asarray([], int)
        memory_edge_count[memory_edges] += 1
        if np.any(static[:3] > 0):
            module_count["conservative"] += 1
        if np.any(static[3:] > 0):
            module_count["dissipative"] += 1
        if np.any(memory > 0):
            module_count["memory"] += 1
        jaccard.append(len(edges & selected_edges) / max(len(edges | selected_edges), 1))
    edge_rows = [
        {
            "edge_index": edge,
            "edge_support_rate": edge_count[edge] / draws,
            "memory_edge_support_rate": memory_edge_count[edge] / draws,
            "stable_edge": edge_count[edge] / draws
            >= float(config["minimum_edge_bootstrap_support"]),
            "stable_memory_edge": memory_edge_count[edge] / draws
            >= float(config["minimum_edge_bootstrap_support"]),
        }
        for edge in range(n_edges)
    ]
    module_rows = [
        {
            "module": module,
            "support_rate": module_count[module] / draws,
            "stable": module_count[module] / draws
            >= float(config["minimum_module_bootstrap_support"]),
        }
        for module in MODULES
    ]
    summary = {
        "draws": draws,
        "cluster_unit": "fault_line",
        "n_fault_line_clusters": len(lines),
        "seed": int(config["seed"]),
        "fixed_hyperparameters": {
            "fit_taus": list(fit_taus),
            "ridge": ridge,
            "relative_group_threshold": threshold,
            "coefficient_scale": scale,
        },
        "median_edge_support_jaccard": float(np.median(jaccard)),
        "minimum_claim_support": float(config["minimum_edge_bootstrap_support"]),
    }
    return edge_rows, module_rows, summary


def _paired_cluster_bootstrap(
    reference: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    *,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    left = {row["event_id"]: row for row in reference if row["status"] == "completed"}
    right = {row["event_id"]: row for row in comparator if row["status"] == "completed"}
    common = sorted(set(left) & set(right))
    by_line: dict[str, list[float]] = {}
    for event_id in common:
        line = str(left[event_id]["fault_line"])
        by_line.setdefault(line, []).append(float(right[event_id]["rmse"]) - float(left[event_id]["rmse"]))
    lines = sorted(by_line)
    if not lines:
        return {"status": "INSUFFICIENT_INFORMATION", "n_paired_cases": 0}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choice(lines, size=len(lines), replace=True)
        values = [value for line in selected for value in by_line[str(line)]]
        samples.append(float(np.mean(values)))
    alpha = (1.0 - confidence) / 2.0
    differences = [value for line in lines for value in by_line[line]]
    return {
        "status": "completed",
        "n_paired_cases": len(common),
        "n_fault_line_clusters": len(lines),
        "draws": draws,
        "cluster_unit": "fault_line",
        "mean_difference_comparator_minus_reference": float(np.mean(differences)),
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
    }


def _residual_features(
    model: Any,
    sample: dict[str, Any],
    raw: dict[str, Any],
    partition: tuple[tuple[int, ...], ...],
    horizon: int,
) -> dict[str, Any]:
    target = sample["training_context"]["targets"]
    raw_target = raw["training_context"]["targets"]
    steps = min(horizon, len(target["t"]), len(raw_target["t"]))
    base = {
        "event_id": raw["event_id"],
        "fault_line": raw["training_context"]["metadata"]["fault_line"],
        "source_dataset": _source(raw),
        "horizon_steps": horizon,
    }
    if steps < horizon:
        return {**base, "status": "ineligible_short_horizon"}
    try:
        delta, omega = model.rollout(
            sample["model_inputs"]["static"],
            np.asarray(target["t"][:steps]),
            {
                "Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]),
                "interpolation": "linear",
            },
        )
        mass = np.asarray(raw["model_inputs"]["static"]["mass"])
        delta = reconstruct_model_state(delta, mass, partition)
        omega = reconstruct_model_state(omega, mass, partition)
        residual = omega - np.asarray(raw_target["omega"][:steps])
        centered = residual - np.mean(residual, axis=0, keepdims=True)
        variance = float(np.sum(centered**2))

        def autocorrelation(lag: int) -> float:
            if lag >= len(centered) or variance <= np.finfo(float).eps:
                return 0.0
            return float(np.sum(centered[:-lag] * centered[lag:]) / variance)

        delays = min(12, max(2, len(centered) // 4))
        hankel = np.stack(
            [centered[index : index + delays].reshape(-1) for index in range(len(centered) - delays + 1)]
        )
        singular = np.linalg.svd(hankel, compute_uv=False)
        stable_rank = float(np.sum(singular**2) / max(float(singular[0] ** 2), np.finfo(float).eps))
        spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
        total = float(np.sum(spectrum))
        low_bins = max(1, spectrum.shape[0] // 10)
        return {
            **base,
            "status": "completed",
            "omega_residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "omega_residual_lag1": autocorrelation(1),
            "omega_residual_lag20": autocorrelation(min(20, len(centered) - 1)),
            "omega_residual_stable_rank": stable_rank,
            "omega_residual_low_frequency_fraction": float(np.sum(spectrum[:low_bins]) / max(total, np.finfo(float).eps)),
        }
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
        return {**base, "status": "failed", "error": repr(error)}


def _select_same_support(
    aggregate: list[dict[str, Any]], rows: list[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    table = {
        row["route"]: row for row in rows
    }
    candidates = [
        row
        for row in aggregate
        if row["cohort"] == "C3->C3"
        and int(row["horizon_steps"]) == 1000
        and str(row["branch"]).startswith("SS-")
        and float(row["failure_rate"]) == 0.0
        and np.isfinite(float(row["mean_rmse"]))
    ]
    if not candidates:
        return {"status": "INSUFFICIENT_INFORMATION", "reason": "no finite same-support candidate"}
    best = min(float(row["mean_rmse"]) for row in candidates)
    near = [row for row in candidates if float(row["mean_rmse"]) <= best * (1.0 + tolerance)]
    selected = min(
        near,
        key=lambda row: (
            int(table[row["branch"]]["module_count"]),
            float(row["mean_rmse"]),
            row["branch"],
        ),
    )
    return {
        "status": "SELECTED",
        "route": selected["branch"],
        "enabled_modules": table[selected["branch"]]["enabled_modules"],
        "module_count": table[selected["branch"]]["module_count"],
        "mean_rmse": selected["mean_rmse"],
        "best_mean_rmse": best,
        "within_best_tolerance": tolerance,
        "physics_status": table[selected["branch"]]["physics_status"],
    }


def _run_controlled(
    root: Path, config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    _validate_scope(config)
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    if _verify_hashes(hypothesis):
        raise RuntimeError("hypothesis artifacts changed after registration")
    registry = json.loads((hypothesis / "hypothesis_registry.json").read_text(encoding="utf-8"))
    if not registry.get("registered_before_response_metrics"):
        raise RuntimeError("hypotheses were not preregistered")
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    data = _load_data(root, config)
    train_strata_by_id = {row["event_id"]: row for row in data["train_strata"]}
    raw_val_by_id = {sample["event_id"]: sample for sample in data["raw_val"]}
    horizons = tuple(int(value) for value in config["report_horizons"])
    checkpoints = result_dir / "checkpoints"
    checkpoints.mkdir()
    fitted: dict[str, dict[str, Any]] = {}
    partitions: dict[str, tuple[tuple[int, ...], ...]] = {}
    partition_diagnostics = {}
    views: dict[str, dict[str, list[dict[str, Any]]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for cohort in ("C1", "C2", "C3"):
        partition, diagnostics = _discover_partition(
            data["train_cohorts"][cohort], train_strata_by_id, config
        )
        partitions[cohort] = partition
        partition_diagnostics[cohort] = diagnostics
        train_view = [
            aggregate_model_sample(sample, partition, f"{cohort}_adaptive_k7")
            for sample in data["train_cohorts"][cohort]
        ]
        val_view = [
            aggregate_model_sample(sample, partition, f"{cohort}_adaptive_k7")
            for sample in data["raw_val"]
        ]
        for sample in train_view + val_view:
            audit_model_sample(sample)
        views[cohort] = {"train": train_view, "val": val_view}
        val_ids = {sample["event_id"] for sample in data["val_cohorts"][cohort]}
        fit_val = [sample for sample in val_view if sample["event_id"] in val_ids]
        models, metadata = _fit_stack(
            train_view,
            fit_val,
            config,
            checkpoints,
            cohort,
        )
        fitted[cohort] = models
        candidate_rows.extend(metadata)
        for route, model in models.items():
            for target_cohort in ("C1", "C2", "C3"):
                target_ids = {
                    sample["event_id"] for sample in data["val_cohorts"][target_cohort]
                }
                target_view = [sample for sample in val_view if sample["event_id"] in target_ids]
                rows = _evaluate_candidate(
                    model,
                    f"{cohort}_adaptive_k7",
                    target_view,
                    data["val_cohorts"][target_cohort],
                    partition,
                    horizons,
                    cohort=f"{cohort}->{target_cohort}",
                    branch=route,
                )
                metric_rows.extend(rows)

    c3_psd = fitted["C3"]["PSD-selected"]
    same_support, same_support_rows, _ = _same_support_refits(
        c3_psd,
        views["C3"]["train"],
        [*views["C3"]["train"], *views["C3"]["val"]],
        config,
    )
    for route, model in same_support.items():
        model.save(checkpoints / f"C3-{route}.npz")
        for target_cohort in ("C1", "C2", "C3"):
            target_ids = {
                sample["event_id"] for sample in data["val_cohorts"][target_cohort]
            }
            target_view = [
                sample for sample in views["C3"]["val"] if sample["event_id"] in target_ids
            ]
            rows = _evaluate_candidate(
                model,
                "C3_adaptive_k7",
                target_view,
                data["val_cohorts"][target_cohort],
                partitions["C3"],
                horizons,
                cohort=f"C3->{target_cohort}",
                branch=route,
            )
            metric_rows.extend(rows)
            if target_cohort == "C3":
                for sample in target_view:
                    residual_rows.append(
                        {
                            "training_cohort": "C3",
                            "target_cohort": "C3",
                            "route": route,
                            **_residual_features(
                                model,
                                sample,
                                raw_val_by_id[sample["event_id"]],
                                partitions["C3"],
                                max(horizons),
                            ),
                        }
                    )
    aggregate = _aggregate(metric_rows)
    selection = _select_same_support(
        aggregate, same_support_rows, float(config["minimum_model_improvement"])
    )
    edge_support, module_support, support_summary = _support_bootstrap(
        c3_psd, views["C3"]["train"], config
    )
    performance_bootstrap = []
    c3_h1000 = [
        row
        for row in metric_rows
        if row["cohort"] == "C3->C3" and int(row["horizon_steps"]) == 1000
    ]
    selected_route = selection.get("route")
    if selected_route:
        reference = [row for row in c3_h1000 if row["branch"] == selected_route]
        comparisons = {
            row["branch"]
            for row in c3_h1000
            if row["branch"] in {"B2", "B3", *same_support}
            and row["branch"] != selected_route
        }
        for comparator_route in sorted(comparisons):
            comparison = [row for row in c3_h1000 if row["branch"] == comparator_route]
            performance_bootstrap.append(
                {
                    "reference_route": selected_route,
                    "comparator_route": comparator_route,
                    **_paired_cluster_bootstrap(
                        reference,
                        comparison,
                        draws=int(config["bootstrap_draws"]),
                        confidence=float(config["bootstrap_confidence"]),
                        seed=int(config["seed"]),
                    ),
                }
            )
    convergence = {}
    if selected_route:
        sample = next(
            sample
            for sample in views["C3"]["val"]
            if sample["event_id"] in {row["event_id"] for row in data["val_cohorts"]["C3"]}
        )
        steps = min(max(horizons), len(sample["training_context"]["targets"]["t"]))
        convergence = same_support[selected_route].convergence_audit(
            sample["model_inputs"]["static"],
            np.asarray(sample["training_context"]["targets"]["t"][:steps]),
            {
                "Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]),
                "interpolation": "linear",
            },
        )
    marker = {
        "run_id": run_id,
        "round": 1,
        "controlled_selection_complete": True,
        "selected_same_support_candidate": selection,
        "test_ids_read": [],
        "fold": "F1",
        "seed": 324,
        "observed_Pm_is_conditional": True,
        "event_specific_future_routing": False,
        "physical_line_recovery_claim": False,
        "absolute_true_order_claim": False,
        "unselected_module_absence_claim": False,
    }
    _json(result_dir / "controlled_selection_marker.json", marker)
    for role, rows in (("train", data["train_strata"]), ("val", data["val_strata"])):
        for row in rows:
            row["role"] = role
            metadata = data["store"].metadata(row["event_id"])
            row["fault_line"] = metadata.fault_line
            row["source_dataset"] = metadata.source_dataset
    _csv(result_dir / "data_strata.csv", [*data["train_strata"], *data["val_strata"]])
    _csv(result_dir / "candidate_models.csv", candidate_rows)
    _csv(result_dir / "same_support_ablation_design.csv", same_support_rows)
    _csv(result_dir / "per_case_metrics.csv", metric_rows)
    _csv(result_dir / "aggregate_metrics.csv", aggregate)
    _csv(result_dir / "residual_per_case.csv", residual_rows)
    _csv(result_dir / "edge_support_bootstrap.csv", edge_support)
    _csv(result_dir / "module_support_bootstrap.csv", module_support)
    _csv(result_dir / "performance_cluster_bootstrap.csv", performance_bootstrap)
    _json(result_dir / "support_bootstrap_summary.json", support_summary)
    _json(result_dir / "integration_convergence.json", convergence)
    _json(result_dir / "stratification_policy.json", data["policy"].as_dict())
    _json(result_dir / "cohort_policy.json", data["cohort_policy"])
    _json(
        result_dir / "spatial_partitions.json",
        {"partitions": partitions, "diagnostics": partition_diagnostics},
    )
    physics = {
        f"{cohort}::{route}": model.physics_audit()
        for cohort, models in fitted.items()
        for route, model in models.items()
    }
    physics.update(
        {f"C3::{route}": model.physics_audit() for route, model in same_support.items()}
    )
    _json(result_dir / "physics_audits.json", physics)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "controlled_experiment",
            "hypothesis_result": hypothesis.name,
            "hypothesis_sha256": _sha256(hypothesis / "hypothesis_registry.json"),
            "dataset_revision": data["store"].dataset_metadata.dataset_revision,
            "dataset_build_id": data["store"].dataset_metadata.build_id,
            "train_ids_used": data["train_ids"],
            "val_ids_used": data["val_ids"],
            "test_ids_read": [],
            "v1_results_read": [],
            "labels_used_for_fit_selection_or_bootstrap": False,
            "bootstrap_cluster_unit": "fault_line",
            "observed_Pm_is_conditional_noncausal_sensitivity": True,
            "inertia_is_declared_prior_not_reidentified": True,
            "config_sha256": _sha256(config_path),
        },
    )
    _write_hashes(result_dir)


def _run_residual_diagnosis(
    root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    if _verify_hashes(controlled):
        raise RuntimeError("controlled artifacts changed before residual diagnosis")
    with (controlled / "residual_per_case.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows if row["status"] == "completed"]
    summaries = []
    for route in sorted({row["route"] for row in completed}):
        group = [row for row in completed if row["route"] == route]
        summaries.append(
            {
                "route": route,
                "completed_cases": len(group),
                "mean_omega_residual_rmse": float(np.mean([float(row["omega_residual_rmse"]) for row in group])),
                "mean_absolute_lag1": float(np.mean([abs(float(row["omega_residual_lag1"])) for row in group])),
                "mean_absolute_lag20": float(np.mean([abs(float(row["omega_residual_lag20"])) for row in group])),
                "median_residual_stable_rank": float(np.median([float(row["omega_residual_stable_rank"]) for row in group])),
                "mean_low_frequency_fraction": float(np.mean([float(row["omega_residual_low_frequency_fraction"]) for row in group])),
            }
        )
    marker = json.loads(
        (controlled / "controlled_selection_marker.json").read_text(encoding="utf-8")
    )
    selected = marker["selected_same_support_candidate"].get("route")
    selected_row = next((row for row in summaries if row["route"] == selected), None)
    memory_free_route = (
        _module_code(
            {
                module
                for module, symbol in zip(MODULES, ("C", "D", "M"), strict=True)
                if selected and symbol in selected.removeprefix("SS-") and module != "memory"
            }
        )
        if selected
        else None
    )
    memory_free = next(
        (
            row
            for row in summaries
            if selected
            and row["route"] == memory_free_route
        ),
        None,
    )
    if selected_row is None:
        priority = "fixed-support bootstrap consensus refit"
        reason = "no finite same-support candidate was selected"
    elif "M" in selected and memory_free is not None and selected_row["mean_absolute_lag20"] >= memory_free["mean_absolute_lag20"]:
        priority = "expand stable memory time-scale grid"
        reason = "selected memory did not reduce long-lag residual correlation"
    elif selected_row["median_residual_stable_rank"] > 6.0:
        priority = "retain one mass-orthogonal intra-cluster mode"
        reason = "selected residual retains high stable rank"
    else:
        priority = "none_required_before_acceptance"
        reason = "registered residual diagnostics do not demand another round"
    diagnosis = {
        "run_id": run_id,
        "round": 1,
        "controlled_result": controlled.name,
        "selected_route": selected,
        "next_experiment_priority": priority,
        "priority_reason": reason,
        "test_ids_read": [],
        "v1_results_read": [],
    }
    _csv(result_dir / "residual_summary.csv", summaries)
    _json(result_dir / "residual_diagnosis.json", diagnosis)
    _write_hashes(result_dir)


def _run_adversarial_review(
    root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    residual = _latest_result(
        root, str(config["residual_experiment"]), "residual_diagnosis"
    )
    lineage = json.loads((controlled / "lineage.json").read_text(encoding="utf-8"))
    marker = json.loads(
        (controlled / "controlled_selection_marker.json").read_text(encoding="utf-8")
    )
    support = json.loads(
        (controlled / "support_bootstrap_summary.json").read_text(encoding="utf-8")
    )
    with (controlled / "same_support_ablation_design.csv").open(encoding="utf-8-sig") as handle:
        ablations = list(csv.DictReader(handle))
    with (controlled / "performance_cluster_bootstrap.csv").open(encoding="utf-8-sig") as handle:
        performance = list(csv.DictReader(handle))
    physics = json.loads((controlled / "physics_audits.json").read_text(encoding="utf-8"))
    envelopes = {row["support_envelope_sha256"] for row in ablations}
    gates = {
        "hypothesis_artifacts_immutable": not _verify_hashes(hypothesis),
        "controlled_artifacts_immutable": not _verify_hashes(controlled),
        "residual_artifacts_immutable": not _verify_hashes(residual),
        "hypothesis_precedes_controlled_metrics": lineage.get("hypothesis_result") == hypothesis.name,
        "v1_results_not_consumed": not lineage.get("v1_results_read"),
        "test_remained_locked": not lineage.get("test_ids_read") and not marker.get("test_ids_read"),
        "observed_pm_boundary_explicit": marker.get("observed_Pm_is_conditional") is True,
        "all_eight_same_support_subsets_present": len(ablations) == 8,
        "support_envelope_identical": len(envelopes) == 1,
        "fault_line_cluster_bootstrap_used": (
            support.get("cluster_unit") == "fault_line"
            and int(support.get("draws", 0)) == int(config["bootstrap_draws"])
            and all(row.get("cluster_unit") == "fault_line" for row in performance)
        ),
        "cross_level_directions_present": _cross_level_directions(controlled),
        "selected_and_ablation_physics_pass": all(
            audit.get("status") == "pass" for audit in physics.values()
        ),
        "no_uncertified_claims": (
            marker.get("physical_line_recovery_claim") is False
            and marker.get("absolute_true_order_claim") is False
            and marker.get("unselected_module_absence_claim") is False
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    diagnosis = json.loads((residual / "residual_diagnosis.json").read_text(encoding="utf-8"))
    review = {
        "run_id": run_id,
        "round": 1,
        "status": status,
        "decision": "ACCEPTABLE" if status == "PASS" else "REJECT",
        "gates": gates,
        "next_experiment_if_rejected": diagnosis["next_experiment_priority"],
        "test_ids_read": [],
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# M0-AFSD v2 对抗审查\n\n"
        f"**结论：{review['decision']}**\n\n"
        + "\n".join(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items())
        + "\n",
        encoding="utf-8",
    )
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError(
            "v2 adversarial review rejected round 1; execute the registered highest-information experiment"
        )


def _cross_level_directions(controlled: Path) -> bool:
    with (controlled / "aggregate_metrics.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cohorts = {row["cohort"] for row in rows}
    return all(
        route in cohorts
        for route in ("C1->C3", "C2->C3", "C3->C1", "C3->C2", "C3->C3")
    )


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _run_acceptance(
    root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    residual = _latest_result(
        root, str(config["residual_experiment"]), "residual_diagnosis"
    )
    adversarial = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_review"
    )
    review = json.loads((adversarial / "adversarial_review.json").read_text(encoding="utf-8"))
    marker = json.loads(
        (controlled / "controlled_selection_marker.json").read_text(encoding="utf-8")
    )
    selection = marker["selected_same_support_candidate"]
    with (controlled / "aggregate_metrics.csv").open(encoding="utf-8-sig") as handle:
        aggregate = list(csv.DictReader(handle))
    with (controlled / "performance_cluster_bootstrap.csv").open(encoding="utf-8-sig") as handle:
        bootstrap = list(csv.DictReader(handle))
    with (controlled / "module_support_bootstrap.csv").open(encoding="utf-8-sig") as handle:
        module_support = {row["module"]: row for row in csv.DictReader(handle)}
    with (controlled / "edge_support_bootstrap.csv").open(encoding="utf-8-sig") as handle:
        edge_support = list(csv.DictReader(handle))
    physics = json.loads((controlled / "physics_audits.json").read_text(encoding="utf-8"))
    convergence = json.loads(
        (controlled / "integration_convergence.json").read_text(encoding="utf-8")
    )
    selected_route = selection.get("route")

    def metric(route: str, cohort: str = "C3->C3") -> dict[str, str] | None:
        return next(
            (
                row
                for row in aggregate
                if row["branch"] == route
                and row["cohort"] == cohort
                and int(row["horizon_steps"]) == 1000
            ),
            None,
        )

    selected_metric = metric(str(selected_route)) if selected_route else None
    b2_metric = metric("B2")
    active_modules = set(selection.get("enabled_modules", []))
    necessary = {}
    for module in active_modules:
        reduced = _module_code(active_modules - {module})
        row = next(
            (
                item
                for item in bootstrap
                if item["reference_route"] == selected_route
                and item["comparator_route"] == reduced
            ),
            None,
        )
        reduced_metric = metric(reduced)
        relative = (
            (_float(reduced_metric["mean_rmse"]) - _float(selected_metric["mean_rmse"]))
            / max(_float(reduced_metric["mean_rmse"]), np.finfo(float).eps)
            if reduced_metric and selected_metric
            else float("nan")
        )
        necessary[module] = {
            "knockout_route": reduced,
            "relative_improvement": relative,
            "bootstrap_ci_lower": _float(row["ci_lower"]) if row else float("nan"),
            "bootstrap_ci_upper": _float(row["ci_upper"]) if row else float("nan"),
            "module_support_rate": _float(module_support.get(module, {}).get("support_rate")),
            "passes": bool(
                row
                and relative >= float(config["minimum_model_improvement"])
                and _float(row["ci_lower"]) > 0.0
                and _float(module_support[module]["support_rate"])
                >= float(config["minimum_module_bootstrap_support"])
            ),
        }
    modal_improvement = False
    if selected_metric and b2_metric:
        modal_improvement = any(
            (
                _float(selected_metric["median_modal_frequency_relative_error"])
                < _float(b2_metric["median_modal_frequency_relative_error"]),
                _float(selected_metric["median_modal_damping_absolute_error"])
                < _float(b2_metric["median_modal_damping_absolute_error"]),
                _float(selected_metric["median_modal_mac"])
                > _float(b2_metric["median_modal_mac"]),
            )
        )
    b2_improvement = (
        (_float(b2_metric["mean_rmse"]) - _float(selected_metric["mean_rmse"]))
        / max(_float(b2_metric["mean_rmse"]), np.finfo(float).eps)
        if selected_metric and b2_metric
        else float("nan")
    )
    cross_rows = [metric(str(selected_route), f"C3->{cohort}") for cohort in ("C1", "C2", "C3")]
    scientific_gates = {
        "same_support_candidate_selected": selection.get("status") == "SELECTED",
        "selected_failure_rate_zero": bool(selected_metric and _float(selected_metric["failure_rate"]) == 0.0),
        "selected_improves_B2_by_2pct": b2_improvement >= float(config["minimum_model_improvement"]),
        "at_least_one_modal_metric_improves": modal_improvement,
        "every_selected_module_is_necessary": bool(active_modules) and all(
            row["passes"] for row in necessary.values()
        ),
        "cross_level_global_rollouts_finite": all(
            row is not None
            and _float(row["failure_rate"]) == 0.0
            and np.isfinite(_float(row["mean_rmse"]))
            for row in cross_rows
        ),
        "physics_audit_passes": all(row.get("status") == "pass" for row in physics.values()),
        "rk4_convergence_passes": convergence.get("convergent") is True,
        "coi_drift_passes": bool(
            selected_metric
            and _float(selected_metric["maximum_coi_drift"]) <= float(config["maximum_coi_drift"])
        ),
    }
    evidence_complete = bool(
        review.get("status") == "PASS"
        and not _verify_hashes(controlled)
        and not _verify_hashes(residual)
        and not _verify_hashes(adversarial)
        and len(edge_support) > 0
        and len(bootstrap) > 0
    )
    positive = evidence_complete and all(scientific_gates.values())
    if positive:
        conclusion = "MINIMUM_EFFECTIVE_MODULE_IDENTIFIED"
        decision = "ACCEPT"
    elif evidence_complete:
        conclusion = "INSUFFICIENT_INFORMATION"
        decision = "ACCEPT_NEGATIVE_CONCLUSION"
    else:
        conclusion = "UNRESOLVED"
        decision = "REJECT"
    acceptance = {
        "run_id": run_id,
        "round": 1,
        "status": "PASS" if decision != "REJECT" else "FAIL",
        "decision": decision,
        "scientific_conclusion": conclusion,
        "selected_route": selected_route,
        "enabled_modules": sorted(active_modules),
        "relative_improvement_vs_B2": b2_improvement,
        "module_necessity": necessary,
        "scientific_gates": scientific_gates,
        "evidence_complete": evidence_complete,
        "stable_edge_indices": [
            int(row["edge_index"])
            for row in edge_support
            if row["stable_edge"].lower() == "true"
        ],
        "stable_memory_edge_indices": [
            int(row["edge_index"])
            for row in edge_support
            if row["stable_memory_edge"].lower() == "true"
        ],
        "claim_boundary": (
            "F1 seed324 validation-only observed-Pm conditional module-selection study; "
            "no test, other-fold, autonomous-Pm, physical-line, machine-parameter, or absolute-order claim"
        ),
        "test_ids_read": [],
        "merge_gate_requires_explicit_approval": True,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# M0-AFSD v2 模块选择验收\n\n"
        f"**决定：{decision}**\n\n"
        f"科学结论：`{conclusion}`。\n\n"
        + "\n".join(
            f"- {'PASS' if value else 'FAIL'} `{key}`"
            for key, value in scientific_gates.items()
        )
        + "\n",
        encoding="utf-8",
    )
    _write_hashes(result_dir)
    if decision == "REJECT":
        raise RuntimeError("v2 acceptance rejected; next registered information-gain experiment is required")


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _validate_scope(config)
    if stage == "hypothesis_registration":
        _run_hypothesis_registration(config_path, config, result_dir, run_id)
    elif stage == "controlled_experiment":
        _run_controlled(root, config_path, config, result_dir, run_id)
    elif stage == "residual_diagnosis":
        _run_residual_diagnosis(root, config, result_dir, run_id)
    elif stage == "adversarial_review":
        _run_adversarial_review(root, config, result_dir, run_id)
    elif stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported M0-AFSD v2 stage: {stage}")
