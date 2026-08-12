from __future__ import annotations

import csv
import hashlib
import html
import json
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kron_reduce.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation.metrics import (
    aggregate_trajectory_metrics,
    trajectory_metrics,
)
from kron_reduce.models import (
    CausalPmGraphModalM0Model,
    CausalPmPortStructuredM0Model,
    GraphModalDiscoveryM0Model,
    GraphModalPmDynamics,
    PortStructuredDiscoveryM0Model,
    RunContext,
    TransientEmfM0Model,
)


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
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


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


def _verify_artifact_hashes(result_dir: Path) -> list[str]:
    expected = json.loads(
        (result_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    return [
        relative
        for relative, digest in expected.items()
        if not (result_dir / relative).is_file()
        or _sha256(result_dir / relative) != digest
    ]


def _source(sample: dict[str, Any]) -> str:
    return str(sample["training_context"]["metadata"]["source_dataset"])


def _clearing_stress(sample: dict[str, Any]) -> float:
    static = sample["model_inputs"]["static"]
    incidence = np.asarray(static["incidence"], dtype=np.float64)
    delta0 = np.asarray(static["delta0"], dtype=np.float64).reshape(-1)
    equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64)
    return float(
        np.max(
            np.abs(
                np.sin(incidence.T @ delta0)
                - np.sin(incidence.T @ equilibrium)
            )
        )
    )


def _threshold_policy(train: list[dict[str, Any]]) -> dict[str, Any]:
    stress = np.asarray([_clearing_stress(sample) for sample in train])
    quantiles = np.quantile(stress, (0.25, 0.5, 0.75, 0.9))
    thresholds = sorted({0.0, *(float(value) for value in quantiles)})
    return {
        "stress_definition": "max_abs_delta_sin_at_clearing",
        "quantiles": [0.25, 0.5, 0.75, 0.9],
        "quantile_values": quantiles.tolist(),
        "threshold_candidates": thresholds,
        "activation_width": max(
            0.02, float((quantiles[-1] - quantiles[0]) / 4.0)
        ),
        "constructed_from": "train_only",
        "labels_used": False,
        "validation_or_test_used": False,
    }


def _context(config: dict[str, Any], train: list[dict], val: list[dict]) -> RunContext:
    return RunContext(
        seed=int(config["seed"]),
        epochs=0,
        horizon_steps=int(config["selection_horizon_steps"]),
        max_cases=max(len(train), len(val)),
        learning_rate=0.0,
        ridge_values=tuple(float(value) for value in config["ridge_values"]),
        integral_windows=tuple(int(value) for value in config["integral_windows"]),
        max_internal_step=float(config["max_internal_step"]),
        fit_max_steps=int(config["fit_max_steps"]),
    )


def _flatten_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for original in rows:
        row = dict(original)
        for nested in ("source_standardized_loss", "short_source_standardized_loss"):
            sources = row.pop(nested, {}) or {}
            for source, value in sources.items():
                row[f"{nested}_{source}"] = value
        for key, value in list(row.items()):
            if isinstance(value, (list, tuple, dict)):
                row[key] = json.dumps(value, ensure_ascii=False)
        output.append(row)
    return output


def _fit_chain(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoints: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _context(config, train, val)
    threshold = _threshold_policy(train)
    m0_3eg = TransientEmfM0Model(
        alpha_candidates=tuple(
            float(value) for value in config["m0_3eg_alpha_candidates"]
        ),
        tau_candidates=tuple(
            float(value) for value in config["m0_3eg_tau_candidates"]
        ),
        activation_threshold_candidates=tuple(threshold["threshold_candidates"]),
        activation_width=float(threshold["activation_width"]),
        max_source_loss_degradation=float(
            config["m0_3eg_max_source_loss_degradation"]
        ),
        max_internal_step=float(config["max_internal_step"]),
    )
    m0_3eg.fit(train, val, context)
    if m0_3eg.physics_audit().get("status") != "pass":
        raise RuntimeError("raw ten-machine M0-3EG failed its physics audit")
    m0_3eg_path = checkpoints / "M0-3EG.npz"
    m0_candidates = m0_3eg.diagnostics.pop("candidate_table")
    m0_3eg.diagnostics["candidate_table_artifact"] = "candidate_M0-3EG.csv"
    m0_3eg.save(m0_3eg_path)

    gmd = GraphModalDiscoveryM0Model(
        backbone=TransientEmfM0Model.load(m0_3eg_path),
        max_internal_step=float(config["max_internal_step"]),
    )
    gmd.fit(train, val, context)
    if gmd.physics_audit().get("status") != "pass":
        raise RuntimeError("raw ten-machine M0-GMD failed its physics audit")
    gmd_path = checkpoints / "M0-GMD.npz"
    gmd_candidates = gmd.diagnostics.pop("candidate_table")
    gmd_long = gmd.diagnostics.pop("long_horizon_candidate_table")
    gmd.diagnostics["candidate_table_artifact"] = "candidate_M0-GMD_short.csv"
    gmd.diagnostics["long_candidate_table_artifact"] = "candidate_M0-GMD_long.csv"
    gmd.save(gmd_path)

    gmd_pm = CausalPmGraphModalM0Model(
        backbone=GraphModalDiscoveryM0Model.load(gmd_path),
        pm_dynamics=GraphModalPmDynamics(
            weak_windows=tuple(int(value) for value in config["integral_windows"]),
            ridge_values=tuple(float(value) for value in config["pm_ridge_values"]),
            relative_thresholds=tuple(
                float(value) for value in config["pm_relative_thresholds"]
            ),
            spectral_margins=tuple(
                float(value) for value in config["pm_spectral_margins"]
            ),
        ),
        maximum_effective_time_constant=float(
            config["pm_maximum_effective_time_constant_s"]
        ),
        max_internal_step=float(config["max_internal_step"]),
    )
    gmd_pm.fit(train, val, context)
    if gmd_pm.physics_audit().get("status") != "pass":
        raise RuntimeError("raw ten-machine M0-GMD-PM failed its physics audit")
    gmd_pm_path = checkpoints / "M0-GMD-PM.npz"
    gmd_pm_candidates = gmd_pm.diagnostics.pop("candidate_table")
    gmd_pm_long = gmd_pm.diagnostics.pop("validation_shortlist")
    gmd_pm.diagnostics["candidate_table_artifact"] = "candidate_M0-GMD-PM.csv"
    gmd_pm.diagnostics["closed_loop_table_artifact"] = (
        "candidate_M0-GMD-PM_closed_loop.csv"
    )
    gmd_pm.save(gmd_pm_path)

    psd = PortStructuredDiscoveryM0Model(
        backbone=TransientEmfM0Model.load(m0_3eg_path),
        tau_mode_candidates=tuple(
            tuple(float(value) for value in row)
            for row in config["psd_tau_mode_candidates"]
        ),
        ridge_values=tuple(float(value) for value in config["psd_ridge_values"]),
        group_thresholds=tuple(
            float(value) for value in config["psd_group_thresholds"]
        ),
        coefficient_scales=tuple(
            float(value) for value in config["psd_coefficient_scales"]
        ),
        minimum_validation_improvement=float(
            config["psd_minimum_validation_improvement"]
        ),
        maximum_source_loss_degradation=float(
            config["psd_maximum_source_loss_degradation"]
        ),
        max_internal_step=float(config["max_internal_step"]),
    )
    psd.fit(train, val, context)
    if psd.physics_audit().get("status") != "pass":
        raise RuntimeError("raw ten-machine M0-PSD failed its physics audit")
    psd_path = checkpoints / "M0-PSD.npz"
    psd_candidates = psd.diagnostics.pop("candidate_table")
    psd_long = psd.diagnostics.pop("long_horizon_candidate_table")
    psd.diagnostics["candidate_table_artifact"] = "candidate_M0-PSD_short.csv"
    psd.diagnostics["long_candidate_table_artifact"] = "candidate_M0-PSD_long.csv"
    psd.save(psd_path)

    psd_pm = CausalPmPortStructuredM0Model(
        backbone=PortStructuredDiscoveryM0Model.load(psd_path),
        pm_dynamics=GraphModalPmDynamics(
            weak_windows=tuple(int(value) for value in config["integral_windows"]),
            ridge_values=tuple(float(value) for value in config["pm_ridge_values"]),
            relative_thresholds=tuple(
                float(value) for value in config["pm_relative_thresholds"]
            ),
            spectral_margins=tuple(
                float(value) for value in config["pm_spectral_margins"]
            ),
        ),
        maximum_effective_time_constant=float(
            config["pm_maximum_effective_time_constant_s"]
        ),
        max_internal_step=float(config["max_internal_step"]),
    )
    psd_pm.fit(train, val, context)
    if psd_pm.physics_audit().get("status") != "pass":
        raise RuntimeError("raw ten-machine M0-PSD-PM failed its physics audit")
    psd_pm_path = checkpoints / "M0-PSD-PM.npz"
    psd_pm_candidates = psd_pm.diagnostics.pop("candidate_table")
    psd_pm_long = psd_pm.diagnostics.pop("validation_shortlist")
    psd_pm.diagnostics["candidate_table_artifact"] = "candidate_M0-PSD-PM.csv"
    psd_pm.diagnostics["closed_loop_table_artifact"] = (
        "candidate_M0-PSD-PM_closed_loop.csv"
    )
    psd_pm.save(psd_pm_path)

    models = {
        "M0-3EG": m0_3eg,
        "M0-GMD": gmd,
        "M0-GMD-PM": gmd_pm,
        "M0-PSD": psd,
        "M0-PSD-PM": psd_pm,
    }
    tables = {
        "candidate_M0-3EG.csv": m0_candidates,
        "candidate_M0-GMD_short.csv": gmd_candidates,
        "candidate_M0-GMD_long.csv": gmd_long,
        "candidate_M0-GMD-PM.csv": gmd_pm_candidates,
        "candidate_M0-GMD-PM_closed_loop.csv": gmd_pm_long,
        "candidate_M0-PSD_short.csv": psd_candidates,
        "candidate_M0-PSD_long.csv": psd_long,
        "candidate_M0-PSD-PM.csv": psd_pm_candidates,
        "candidate_M0-PSD-PM_closed_loop.csv": psd_pm_long,
    }
    return models, {"threshold_policy": threshold, "tables": tables}


def _route_exogenous(
    sample: dict[str, Any], steps: int, input_policy: str
) -> dict[str, np.ndarray | str]:
    if input_policy == "observed":
        return {
            "Pm": np.asarray(sample["training_context"]["Pm_observed"][:steps]),
            "interpolation": "linear",
        }
    if input_policy == "zoh":
        pm_clear = np.asarray(sample["model_inputs"]["static"]["Pm_clear"]).reshape(
            1, -1
        )
        return {
            "Pm": np.repeat(pm_clear, steps, axis=0),
            "interpolation": "zoh",
        }
    if input_policy == "causal_pm_ode":
        return {}
    raise ValueError(f"unknown input policy: {input_policy}")


def _causal_schedule_invariance_audit(
    models: dict[str, Any], sample: dict[str, Any], steps: int
) -> dict[str, Any]:
    """Verify causal-Pm composites cannot consume a supplied future schedule.

    The two schedules share the exact clearing value.  Only samples after
    clearing are changed, by a deliberately large zero-total-power signal.
    A causal composite must integrate its train-discovered Pm ODE from
    ``Pm_clear`` and therefore return bitwise-equivalent state and Pm paths.
    """

    target = sample["training_context"]["targets"]
    static = sample["model_inputs"]["static"]
    time_grid = np.asarray(target["t"][:steps], dtype=np.float64)
    observed = np.asarray(
        sample["training_context"]["Pm_observed"][:steps], dtype=np.float64
    ).copy()
    if len(time_grid) < 2:
        raise ValueError("causality audit requires at least two time points")
    pm_clear = np.asarray(static["Pm_clear"], dtype=np.float64).reshape(-1)
    observed[0] = pm_clear
    direction = np.linspace(-1.0, 1.0, len(pm_clear), dtype=np.float64)
    direction -= np.mean(direction)
    phase = np.linspace(0.0, 8.0 * np.pi, len(time_grid), dtype=np.float64)
    counterfactual = observed.copy()
    counterfactual[1:] += 100.0 * np.sin(phase[1:, np.newaxis]) * direction
    counterfactual[0] = pm_clear
    schedules = (
        {"Pm": observed, "interpolation": "linear"},
        {"Pm": counterfactual, "interpolation": "linear"},
    )
    rows = []
    for method in ("M0-GMD-PM", "M0-PSD-PM"):
        first = models[method].rollout_with_pm(static, time_grid, schedules[0])
        second = models[method].rollout_with_pm(static, time_grid, schedules[1])
        component_errors = {
            component: float(np.max(np.abs(left - right)))
            for component, left, right in zip(
                ("delta", "omega", "Pm"), first, second, strict=True
            )
        }
        rows.append(
            {
                "method": method,
                "event_id": sample["event_id"],
                "steps": len(time_grid),
                "same_Pm_clear": bool(
                    np.array_equal(schedules[0]["Pm"][0], schedules[1]["Pm"][0])
                ),
                "future_schedule_max_abs_perturbation": float(
                    np.max(np.abs(counterfactual[1:] - observed[1:]))
                ),
                "future_perturbation_max_abs_total_power": float(
                    np.max(np.abs(np.sum(counterfactual[1:] - observed[1:], axis=1)))
                ),
                "component_max_abs_errors": component_errors,
                "max_abs_prediction_error": max(component_errors.values()),
                "passed": bool(max(component_errors.values()) <= 1e-14),
            }
        )
    return {
        "definition": (
            "same clearing value; future exogenous Pm changed by a large "
            "zero-total-power signal"
        ),
        "tolerance": 1e-14,
        "future_observed_schedule_is_forbidden_at_deployment": True,
        "rows": rows,
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
    }


def _evaluate(
    models: dict[str, Any],
    samples: list[dict[str, Any]],
    store: CanonicalStore,
    horizons: list[int],
) -> list[dict[str, Any]]:
    routes = [
        ("M0-3EG-observed", models["M0-3EG"], "observed"),
        ("M0-3EG-ZOH", models["M0-3EG"], "zoh"),
        ("M0-GMD-observed", models["M0-GMD"], "observed"),
        ("M0-GMD-ZOH", models["M0-GMD"], "zoh"),
        ("M0-GMD-PM", models["M0-GMD-PM"], "causal_pm_ode"),
        ("M0-PSD-observed", models["M0-PSD"], "observed"),
        ("M0-PSD-ZOH", models["M0-PSD"], "zoh"),
        ("M0-PSD-PM", models["M0-PSD-PM"], "causal_pm_ode"),
    ]
    rows: list[dict[str, Any]] = []
    maximum = max(horizons)
    for sample in samples:
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        time_grid = np.asarray(target["t"][:maximum])
        truth_delta = np.asarray(target["delta"][:maximum])
        truth_omega = np.asarray(target["omega"][:maximum])
        incidence = np.asarray(static["incidence"])
        mass = np.asarray(static["mass"])
        metadata = store.metadata(sample["event_id"])
        for method, model, policy in routes:
            try:
                exogenous = _route_exogenous(sample, maximum, policy)
                if policy == "causal_pm_ode":
                    pred_delta, pred_omega, pred_pm = model.rollout_with_pm(
                        static, time_grid, exogenous
                    )
                else:
                    pred_delta, pred_omega = model.rollout(
                        static, time_grid, exogenous
                    )
                    pred_pm = np.asarray(exogenous["Pm"])
                for horizon in horizons:
                    metrics = trajectory_metrics(
                        truth_delta[:horizon],
                        truth_omega[:horizon],
                        pred_delta[:horizon],
                        pred_omega[:horizon],
                    )
                    edge_error = (
                        pred_delta[:horizon] - truth_delta[:horizon]
                    ) @ incidence
                    rows.append(
                        {
                            "event_id": sample["event_id"],
                            "fault_line": metadata.fault_line,
                            "source_dataset": metadata.source_dataset,
                            "method": method,
                            "input_policy": policy,
                            "horizon_steps": horizon,
                            "status": "completed",
                            **metrics,
                            "edge_angle_rmse": float(
                                np.sqrt(np.mean(edge_error**2))
                            ),
                            "pm_rmse": float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            pred_pm[:horizon]
                                            - sample["training_context"]["Pm_observed"][
                                                :horizon
                                            ]
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            "coi_drift": float(
                                max(
                                    np.max(np.abs(pred_delta[:horizon] @ mass)),
                                    np.max(np.abs(pred_omega[:horizon] @ mass)),
                                    np.max(
                                        np.abs(
                                            np.sum(
                                                pred_pm[:horizon]
                                                - static["Pm_equilibrium"],
                                                axis=1,
                                            )
                                        )
                                    ),
                                )
                            ),
                        }
                    )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
                for horizon in horizons:
                    rows.append(
                        {
                            "event_id": sample["event_id"],
                            "fault_line": metadata.fault_line,
                            "source_dataset": metadata.source_dataset,
                            "method": method,
                            "input_policy": policy,
                            "horizon_steps": horizon,
                            "status": "failed",
                            "error": repr(exc),
                            "rmse": float("nan"),
                        }
                    )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = sorted({(row["method"], int(row["horizon_steps"])) for row in rows})
    for method, horizon in groups:
        subset = [
            row
            for row in rows
            if row["method"] == method and int(row["horizon_steps"]) == horizon
        ]
        aggregate = aggregate_trajectory_metrics(subset)
        completed = [row for row in subset if row["status"] == "completed"]
        output.append(
            {
                "method": method,
                "horizon_steps": horizon,
                "n_cases": len(subset),
                **aggregate,
                "p90_rmse": (
                    float(np.quantile([row["rmse"] for row in completed], 0.9))
                    if completed
                    else float("nan")
                ),
                "mean_omega_rmse": (
                    float(np.mean([row["omega_rmse"] for row in completed]))
                    if completed
                    else float("nan")
                ),
                "mean_edge_angle_rmse": (
                    float(np.mean([row["edge_angle_rmse"] for row in completed]))
                    if completed
                    else float("nan")
                ),
                "mean_pm_rmse": (
                    float(np.mean([row["pm_rmse"] for row in completed]))
                    if completed
                    else float("nan")
                ),
                "maximum_coi_drift": (
                    float(np.max([row["coi_drift"] for row in completed]))
                    if completed
                    else float("inf")
                ),
            }
        )
    return output


def _source_aggregate(rows: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    output = []
    groups = sorted(
        {
            (row["method"], row["source_dataset"])
            for row in rows
            if int(row["horizon_steps"]) == horizon
        }
    )
    for method, source in groups:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and row["source_dataset"] == source
            and int(row["horizon_steps"]) == horizon
        ]
        completed = [row for row in subset if row["status"] == "completed"]
        output.append(
            {
                "method": method,
                "source_dataset": source,
                "horizon_steps": horizon,
                "n_cases": len(subset),
                "failure_rate": 1.0 - len(completed) / max(len(subset), 1),
                "mean_rmse": (
                    float(np.mean([row["rmse"] for row in completed]))
                    if completed
                    else float("nan")
                ),
                "median_rmse": (
                    float(np.median([row["rmse"] for row in completed]))
                    if completed
                    else float("nan")
                ),
                "p90_rmse": (
                    float(np.quantile([row["rmse"] for row in completed], 0.9))
                    if completed
                    else float("nan")
                ),
            }
        )
    return output


def _paired_fault_line_bootstrap(
    rows: list[dict[str, Any]],
    reference: str,
    candidate: str,
    horizon: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    selected = [row for row in rows if int(row["horizon_steps"]) == horizon]
    lookup = {
        (row["event_id"], row["method"]): row
        for row in selected
        if row["status"] == "completed"
    }
    pairs = []
    for event_id in sorted({key[0] for key in lookup}):
        left = lookup.get((event_id, reference))
        right = lookup.get((event_id, candidate))
        if left is not None and right is not None:
            pairs.append(
                {
                    "event_id": event_id,
                    "fault_line": left["fault_line"],
                    "difference_candidate_minus_reference": right["rmse"] - left["rmse"],
                }
            )
    clusters = sorted({row["fault_line"] for row in pairs})
    rng = np.random.default_rng(seed)
    distribution = []
    for _ in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = [
            row["difference_candidate_minus_reference"]
            for cluster in sampled
            for row in pairs
            if row["fault_line"] == cluster
        ]
        distribution.append(float(np.mean(values)))
    observed = float(
        np.mean([row["difference_candidate_minus_reference"] for row in pairs])
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "horizon_steps": horizon,
        "n_cases": len(pairs),
        "n_fault_line_clusters": len(clusters),
        "mean_difference_candidate_minus_reference": observed,
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "candidate_better_if_negative": True,
    }


def _parameter_rows(models: dict[str, Any], edges: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    m0 = models["M0-3EG"]
    rows = [
        {
            "parameter_family": "nodal_damping",
            "index": index,
            "tail": index,
            "head": index,
            "value": float(value),
            "interpretation": "effective_D_not_machine_truth",
        }
        for index, value in enumerate(m0.damping)
    ]
    rows.extend(
        {
            "parameter_family": "effective_complete_graph_coupling",
            "index": index,
            "tail": edge[0],
            "head": edge[1],
            "value": float(value),
            "interpretation": "learned_effective_K_not_physical_line",
        }
        for index, (edge, value) in enumerate(zip(edges, m0.coupling, strict=True))
    )
    psd = models["M0-PSD"]
    for feature_index, feature in enumerate(psd.STATIC_FEATURES):
        rows.extend(
            {
                "parameter_family": f"PSD_static_{feature}",
                "index": edge_index,
                "tail": edge[0],
                "head": edge[1],
                "value": float(psd.static_coefficients[feature_index, edge_index]),
                "interpretation": (
                    "conservative_edge_potential"
                    if feature in psd.CONSERVATIVE_FEATURES
                    else "nonnegative_edge_dissipation"
                ),
            }
            for edge_index, edge in enumerate(edges)
        )
    for mode, tau in enumerate(psd.taus):
        rows.extend(
            {
                "parameter_family": f"PSD_passive_memory_tau_{tau:g}s",
                "index": edge_index,
                "tail": edge[0],
                "head": edge[1],
                "value": float(psd.memory_coefficients[mode, edge_index]),
                "interpretation": "passive_speed_memory",
            }
            for edge_index, edge in enumerate(edges)
        )
    return rows


def _lookup(aggregate: list[dict[str, Any]], method: str, horizon: int) -> dict[str, Any]:
    return next(
        row
        for row in aggregate
        if row["method"] == method and int(row["horizon_steps"]) == horizon
    )


def _relative_improvement(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(
        (reference["mean_rmse"] - candidate["mean_rmse"])
        / max(reference["mean_rmse"], np.finfo(float).eps)
    )


def _assess(
    config: dict[str, Any],
    models: dict[str, Any],
    aggregate: list[dict[str, Any]],
    bootstraps: dict[str, Any],
) -> dict[str, Any]:
    horizon = max(int(value) for value in config["report_horizons"])
    base = _lookup(aggregate, "M0-3EG-observed", horizon)
    gmd = _lookup(aggregate, "M0-GMD-observed", horizon)
    gmd_zoh = _lookup(aggregate, "M0-GMD-ZOH", horizon)
    gmd_pm = _lookup(aggregate, "M0-GMD-PM", horizon)
    psd = _lookup(aggregate, "M0-PSD-observed", horizon)
    psd_zoh = _lookup(aggregate, "M0-PSD-ZOH", horizon)
    psd_pm = _lookup(aggregate, "M0-PSD-PM", horizon)
    gmd_improvement = _relative_improvement(base, gmd)
    pm_improvement = _relative_improvement(gmd_zoh, gmd_pm)
    psd_improvement_vs_base = _relative_improvement(base, psd)
    psd_degradation_vs_gmd = float(
        (psd["mean_rmse"] - gmd["mean_rmse"])
        / max(gmd["mean_rmse"], np.finfo(float).eps)
    )
    psd_pm_improvement = _relative_improvement(psd_zoh, psd_pm)
    support = models["M0-PSD"].diagnostics["selected_support"]
    support_jaccard = models["M0-PSD"].diagnostics.get(
        "cross_source_edge_support_jaccard"
    )
    gmd_gate = bool(
        gmd_improvement >= float(config["minimum_model_improvement"])
        and gmd["failure_rate"] == 0.0
        and models["M0-GMD"].physics_audit().get("status") == "pass"
    )
    gmd_pm_gate = bool(
        pm_improvement >= float(config["minimum_model_improvement"])
        and gmd_pm["failure_rate"] == 0.0
        and models["M0-GMD-PM"].physics_audit().get("status") == "pass"
    )
    psd_discovery_gate = bool(
        models["M0-PSD"].diagnostics.get("promoted")
        and support["active_edges"] > 0
        and support_jaccard is not None
        and support_jaccard >= float(config["minimum_cross_source_support_jaccard"])
        and psd_degradation_vs_gmd
        <= float(config["maximum_physical_discovery_degradation_vs_gmd"])
        and models["M0-PSD"].physics_audit().get("status") == "pass"
        and psd["failure_rate"] == 0.0
    )
    psd_pm_gate = bool(
        psd_discovery_gate
        and psd_pm_improvement >= float(config["minimum_model_improvement"])
        and psd_pm["failure_rate"] == 0.0
        and models["M0-PSD-PM"].physics_audit().get("status") == "pass"
    )
    return {
        "scope": "F1_single_seed324_validation_only_raw_generator_10",
        "historical_mainline_was_clustered_7": True,
        "historical_exceptions": (
            "later cluster-comparison diagnostics trained generator_10 M0-3EG, "
            "but the accepted M0-3EG→M0-GMD→M0-GMD-PM chain was clustered_7"
        ),
        "h100": {
            "M0_3EG_observed": base,
            "M0_GMD_observed": gmd,
            "M0_GMD_ZOH": gmd_zoh,
            "M0_GMD_PM": gmd_pm,
            "M0_PSD_observed": psd,
            "M0_PSD_ZOH": psd_zoh,
            "M0_PSD_PM": psd_pm,
        },
        "relative_changes": {
            "M0_GMD_vs_M0_3EG_observed_improvement": gmd_improvement,
            "M0_GMD_PM_vs_M0_GMD_ZOH_improvement": pm_improvement,
            "M0_PSD_vs_M0_3EG_observed_improvement": psd_improvement_vs_base,
            "M0_PSD_degradation_vs_M0_GMD_observed": psd_degradation_vs_gmd,
            "M0_PSD_PM_vs_M0_PSD_ZOH_improvement": psd_pm_improvement,
        },
        "gates": {
            "raw10_M0_GMD_promoted_over_M0_3EG": gmd_gate,
            "raw10_M0_GMD_PM_causal_port_promoted_over_ZOH": gmd_pm_gate,
            "raw10_M0_PSD_physical_structure_promoted": psd_discovery_gate,
            "raw10_M0_PSD_PM_full_candidate_promoted": psd_pm_gate,
        },
        "PSD_structure": {
            "promoted_by_internal_validation": models["M0-PSD"].diagnostics.get(
                "promoted"
            ),
            "support": support,
            "source_supports": models["M0-PSD"].diagnostics.get(
                "source_supports", {}
            ),
            "cross_source_edge_support_jaccard": support_jaccard,
            "physics_audit": models["M0-PSD"].physics_audit(),
        },
        "paired_fault_line_bootstraps": bootstraps,
        "claim_boundary": {
            "test_opened": False,
            "five_fold_generalization_claim": False,
            "observed_Pm_is_conditional": True,
            "physical_line_recovery_claim": False,
            "machine_parameter_recovery_claim": False,
            "full_Lyapunov_certificate_claim": False,
        },
    }


def _historical_audit(root: Path) -> dict[str, Any]:
    paths = {
        "accepted_round6_config": root
        / "configs/experiments/paper_m0_gmd_pm_round6_f1_single_seed.toml",
        "accepted_round6_protocol": root / "docs/m0_gmd_pm_round6_protocol.md",
        "paper_manifest": root / "configs/project_manifest.json",
        "accepted_gmd_selection": root
        / "results/diagnostics/m0-gmd-horizon-balanced-f1-seed324-20260731/selection_marker.json",
        "accepted_gmd_pm_selection": root
        / "results/diagnostics/m0-gmd-pm-modal-tgov-bound-f1-seed324-20260801-r4/selection_marker.json",
        "later_raw10_comparison_config": root
        / "configs/experiments/paper_m0_3eg_cluster_compare_f2_seed324.toml",
    }
    config = tomllib.loads(paths["accepted_round6_config"].read_text(encoding="utf-8"))
    gmd = json.loads(paths["accepted_gmd_selection"].read_text(encoding="utf-8"))
    gmd_pm = json.loads(paths["accepted_gmd_pm_selection"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["paper_manifest"].read_text(encoding="utf-8"))
    later_raw10_diagnostics = {
        name: manifest["experiments"][name]["status"]
        for name in (
            "paper_physical_cluster_audit_f1_single_seed",
            "paper_physical_cluster_audit_f1_single_seed_v2",
            "paper_physical_cluster_audit_f1_single_seed_v3",
            "paper_m0_3eg_cluster_compare_f2_seed324",
            "paper_graph_coherent_m0_3eg_f3_seed324",
        )
    }
    return {
        "conclusion": (
            "The accepted M0-3EG→M0-GMD→M0-GMD-PM downstream chain used "
            "the clustered_7 representation. Later generator_10 runs were representation "
            "diagnostics, not the accepted downstream chain."
        ),
        "accepted_round6_representation": config["representation"],
        "accepted_downstream_chain_representation": config["representation"],
        "later_raw10_registered_roles": later_raw10_diagnostics,
        "later_raw10_runs_were_diagnostic_only": all(
            status == "diagnostic" for status in later_raw10_diagnostics.values()
        ),
        "accepted_gmd_selected_modal_rank": gmd["selected_modal_rank"],
        "accepted_gmd_pm_source_accuracy_model": gmd_pm[
            "source_accuracy_model_frozen"
        ],
        "evidence_files": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }


def _make_figure(
    result_dir: Path,
    aggregate: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    assessment: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.2), constrained_layout=True)
    methods = [
        "M0-3EG-observed",
        "M0-GMD-observed",
        "M0-GMD-PM",
        "M0-PSD-observed",
        "M0-PSD-PM",
    ]
    colors = ["#5f6b6d", "#267278", "#3f8f5f", "#b55d30", "#7f3c8d"]
    h100 = [_lookup(aggregate, method, 100)["mean_rmse"] for method in methods]
    axes[0, 0].bar(np.arange(len(methods)), h100, color=colors)
    axes[0, 0].set_xticks(np.arange(len(methods)), [
        "3EG obs", "GMD obs", "GMD-PM", "PSD obs", "PSD-PM"
    ], rotation=20, ha="right")
    axes[0, 0].set_ylabel("Mean trajectory RMSE")
    axes[0, 0].set_title("a  Raw 10-machine validation at H=100")

    for method, color in zip(methods, colors, strict=True):
        rows = sorted(
            (row for row in aggregate if row["method"] == method),
            key=lambda row: row["horizon_steps"],
        )
        axes[0, 1].plot(
            [row["horizon_steps"] for row in rows],
            [row["mean_rmse"] for row in rows],
            marker="o",
            color=color,
            label=method,
        )
    axes[0, 1].set_xlabel("Horizon (samples; 0.01 s/sample)")
    axes[0, 1].set_ylabel("Mean trajectory RMSE")
    axes[0, 1].set_title("b  Reintegration error versus horizon")
    axes[0, 1].legend(frameon=False, fontsize=7)

    damping = [
        row["value"]
        for row in parameter_rows
        if row["parameter_family"] == "nodal_damping"
    ]
    axes[1, 0].bar(np.arange(len(damping)), damping, color="#267278")
    axes[1, 0].set_xlabel("Generator index")
    axes[1, 0].set_ylabel("Learned effective D")
    axes[1, 0].set_title("c  M0-3EG learned damping (diagnostic, not ground truth)")

    psd_rows = [
        row
        for row in parameter_rows
        if row["parameter_family"].startswith("PSD_") and row["value"] > 0
    ]
    if psd_rows:
        values = sorted(psd_rows, key=lambda row: row["value"], reverse=True)[:20]
        labels = [
            f"{row['tail']}-{row['head']}:{row['parameter_family'].replace('PSD_', '')}"
            for row in values
        ]
        axes[1, 1].barh(np.arange(len(values)), [row["value"] for row in values], color="#b55d30")
        axes[1, 1].set_yticks(np.arange(len(values)), labels, fontsize=6)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlabel("Nonnegative coefficient")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "PSD selected the exact zero-closure fallback",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    axes[1, 1].set_title("d  Discovered port-structured effective interactions")
    figure.suptitle(
        "Raw 10-machine M0 chain and gated physical-structure discovery\n"
        + ", ".join(
            f"{key}={'PASS' if value else 'NO'}"
            for key, value in assessment["gates"].items()
        ),
        fontsize=11,
    )
    figure_dir = result_dir / "figures"
    figure_dir.mkdir()
    figure.savefig(figure_dir / "fig1_raw10_model_chain.svg")
    figure.savefig(figure_dir / "fig1_raw10_model_chain.png", dpi=300)
    plt.close(figure)


def _write_report(
    result_dir: Path,
    run_id: str,
    assessment: dict[str, Any],
    historical: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> None:
    changes = assessment["relative_changes"]
    gates = assessment["gates"]
    h100 = assessment["h100"]
    sources = "\n".join(
        f"- {row['method']} / {row['source_dataset']}: mean RMSE={row['mean_rmse']:.6f}, "
        f"failure={row['failure_rate']:.1%}"
        for row in source_rows
        if row["method"]
        in {"M0-3EG-observed", "M0-GMD-observed", "M0-GMD-PM", "M0-PSD-observed"}
    )
    report = f"""# 原始 10 机 M0 主线与物理结构发现：F1/seed324 关键节点报告

运行：`{run_id}`。本报告严格限于 F1、seed324、validation-only；测试集从未读取。

## 结论先行

历史审计结论是：**此前被接受的 M0-3EG → M0-GMD → M0-GMD-PM 主线确实使用 `clustered_7`**。后续虽然做过 `generator_10` 的 M0-3EG 表征对照，但它们是诊断实验，不是已接受的下游 GMD/PM 主线。证据见 `historical_representation_audit.json`。

本轮已经把同一主线迁移到原始 10 机：

- M0-3EG observed-Pm H=100 mean RMSE：{h100['M0_3EG_observed']['mean_rmse']:.6f}；
- M0-GMD observed-Pm：{h100['M0_GMD_observed']['mean_rmse']:.6f}，相对 M0-3EG 改善 {changes['M0_GMD_vs_M0_3EG_observed_improvement']:.2%}；
- M0-GMD-PM 因果部署：{h100['M0_GMD_PM']['mean_rmse']:.6f}，相对同一 GMD 的 ZOH 改善 {changes['M0_GMD_PM_vs_M0_GMD_ZOH_improvement']:.2%}；
- 新的 M0-PSD observed-Pm：{h100['M0_PSD_observed']['mean_rmse']:.6f}，相对 M0-3EG 改善 {changes['M0_PSD_vs_M0_3EG_observed_improvement']:.2%}；
- M0-PSD-PM 因果部署：{h100['M0_PSD_PM']['mean_rmse']:.6f}，相对同一 PSD 的 ZOH 改善 {changes['M0_PSD_PM_vs_M0_PSD_ZOH_improvement']:.2%}。

门控结论：

- 10 机 M0-GMD 晋级：**{'通过' if gates['raw10_M0_GMD_promoted_over_M0_3EG'] else '未通过'}**；
- 10 机 M0-GMD-PM 晋级：**{'通过' if gates['raw10_M0_GMD_PM_causal_port_promoted_over_ZOH'] else '未通过'}**；
- M0-PSD 强物理结构发现晋级：**{'通过' if gates['raw10_M0_PSD_physical_structure_promoted'] else '未通过'}**；
- M0-PSD-PM 全候选晋级：**{'通过' if gates['raw10_M0_PSD_PM_full_candidate_promoted'] else '未通过'}**。

无论候选是否晋级，原始 10 机 M0-3EG 都是本轮之后的基本坐标模型；7 聚类不再作为默认主模型。

![原始10机模型链](figures/fig1_raw10_model_chain.svg)

## 方法

M0-3EG 保留二阶 swing 平衡并增加节点归一化暂态电势状态。D 与完整 10 机有效图上的 K 由高阶同源训练轨迹的积分弱形式一次求解，暂态参数只由验证闭环选择；这里的 D、K 是有效约化系数，不宣称等于设备或物理线路真值。

M0-GMD 在冻结 M0-3EG 后，对训练残差做 POD，随后在图特征和稳定一阶记忆库上进行组稀疏积分求解。它发现的是低秩残差模态，因此预测能力较强，但边级解释有限。

M0-GMD-PM 再对 observed Pm 与预测速度拟合零总功率子空间上的 Hurwitz ODE；部署只读 Pm_clear，不读取未来 Pm。Observed-Pm 结果只作为条件重积分敏感性，不是在线因果预测。

新开发的 M0-PSD 不先把残差压缩为 POD 模态，而是在 45 条完整有效候选边上直接发现三类结构：角度势能梯度、非负静态耗散、非负被动速度记忆。所有系数一次非负积分 ridge 求解；记忆满足 `tau*z_dot=nu-z`，其储能 `0.5*d*tau*z^2` 与机械功率合并后严格留下 `-d*z^2`。候选还必须通过 COI 商空间局部刚度、训练/验证定义域势能、链式法则、内部力平衡和步长收敛审计。它仍是 M0-3EG 的端口结构残差，**不是完整 Lyapunov 证书，也不命名为 M2A-LYAP**。

## 数据与证据边界

训练与验证均只保留至少 100 个真实采样点的轨迹，因此没有把“只有几个点”的记录堆入弱方程。所有尺度、稀疏阈值、时间常数和 Pm ODE 均只使用 train/val；标签、Pe、电压、事件时间和测试样本未参与拟合或选择。

按数据源的 H=100 结果：

{sources}

两种数据源若误差仍明显分层，这说明它们共享网络拓扑并不等于共享完全相同的高阶闭合动力学；这也是结构支持必须做跨源复核、不能只凭合并残差最小化宣称“发现真实方程”的原因。

## 解释与限制

1. 迁移到 10 机消除了聚类分区的人为先验，但会增加完整图的 45 条边和共线性；GMD 用低秩模态处理这一点，PSD 则用非负端口结构与组稀疏门控处理。
2. M0-PSD 中被选中的边是 Kron 后的**有效发电机相互作用**，不能直接映射为 39 节点输电线路。
3. F1 单 seed 只能决定下一阶段的建模主线，不能形成五折或测试泛化结论。
4. 稳定性分类不在本轮选择目标内；本轮先锁定重积分精度、零失败和物理审计。

## 历史审计摘要

{historical['conclusion']}

接受范围：原始 10 机 validation-only 主线及门控后的结构发现结论。下一步只有晋级候选才值得扩展到其余 folds；test 仍保持锁定。
"""
    (result_dir / "report.md").write_text(report, encoding="utf-8")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>原始10机M0物理结构发现</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f3f4f5;color:#202124;line-height:1.7;margin:0}}
main{{max-width:1080px;margin:26px auto;background:white;padding:46px 58px;box-shadow:0 2px 14px #0001}}
h1{{border-bottom:3px solid #267278;padding-bottom:12px}}pre{{white-space:pre-wrap;font:inherit}}img{{max-width:100%;border:1px solid #ddd}}
@media(max-width:720px){{main{{margin:0;padding:24px}}}}</style></head><body><main>
<h1>原始 10 机 M0 主线与物理结构发现</h1><img src="figures/fig1_raw10_model_chain.svg" alt="模型链结果">
<pre>{html.escape(report)}</pre></main></body></html>"""
    (result_dir / "report.html").write_text(document, encoding="utf-8")


def _run_diagnostic(
    *,
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    result_dir: Path,
    run_id: str,
) -> None:
    if (
        config.get("test_access") != "forbidden"
        or config.get("representation") != "generator_10"
        or config.get("training_pm_mode") != "observed"
    ):
        raise RuntimeError("raw ten-machine discovery requires locked test and observed training Pm")
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold=str(config["fold"]),
    )
    train_ids = [
        event_id
        for event_id in split.train_ids
        if store.metadata(event_id).valid_length
        >= int(config["minimum_train_steps"])
    ]
    val_ids = [
        event_id
        for event_id in split.val_ids
        if store.metadata(event_id).valid_length
        >= int(config["minimum_validation_steps"])
    ]
    if not train_ids or not val_ids:
        raise RuntimeError("no eligible raw ten-machine train/validation samples")
    request = ModelDataRequest.initial_state_rollout("observed")
    inertia = store.dataset_metadata.M_prior
    train = [
        adapt_event(
            store.load_event(event_id),
            inertia_prior=inertia,
            request=request,
            representation="generator_10",
        )
        for event_id in train_ids
    ]
    val = [
        adapt_event(
            store.load_event(event_id),
            inertia_prior=inertia,
            request=request,
            representation="generator_10",
        )
        for event_id in val_ids
    ]
    for sample in train + val:
        audit_model_sample(sample)

    checkpoints = result_dir / "checkpoints"
    checkpoints.mkdir()
    models, fit_artifacts = _fit_chain(train, val, config, checkpoints)
    for name, table in fit_artifacts["tables"].items():
        _write_csv(result_dir / name, _flatten_candidate_rows(table))
    _json(result_dir / "m0_3eg_threshold_policy.json", fit_artifacts["threshold_policy"])

    historical = _historical_audit(root)
    _json(result_dir / "historical_representation_audit.json", historical)
    selection_marker = {
        "run_id": run_id,
        "selection_complete": True,
        "fold": config["fold"],
        "seed": config["seed"],
        "representation": "generator_10",
        "minimum_real_steps": int(config["minimum_train_steps"]),
        "train_cases": len(train),
        "validation_cases": len(val),
        "test_ids_read": [],
        "labels_used_for_fit_or_selection": False,
        "measured_Pe_or_voltage_used": False,
        "observed_Pm_used_for_train_and_conditional_sensitivity": True,
        "future_observed_Pm_used_by_causal_deployment_models": False,
        "selected": {
            "M0-3EG": {
                "alpha": models["M0-3EG"].alpha,
                "tau": models["M0-3EG"].tau,
                "activation_threshold": models["M0-3EG"].activation_threshold,
                "anchor_ridge": models["M0-3EG"].anchor.selected_ridge,
            },
            "M0-GMD": {
                "modal_rank": models["M0-GMD"].diagnostics["selected_modal_rank"],
                "memory_order": models["M0-GMD"].diagnostics["selected_memory_order"],
                "features": models["M0-GMD"].diagnostics["selected_feature_names"],
            },
            "M0-GMD-PM": models["M0-GMD-PM"].diagnostics["selected_candidate"],
            "M0-PSD": {
                "promoted": models["M0-PSD"].diagnostics["promoted"],
                "support": models["M0-PSD"].diagnostics["selected_support"],
                "fit_taus": models["M0-PSD"].diagnostics["selected_fit_taus"],
            },
            "M0-PSD-PM": models["M0-PSD-PM"].diagnostics["selected_candidate"],
        },
    }
    _json(result_dir / "selection_marker.json", selection_marker)

    horizons = [int(value) for value in config["report_horizons"]]
    metric_rows = _evaluate(models, val, store, horizons)
    aggregate = _aggregate(metric_rows)
    source_rows = _source_aggregate(metric_rows, max(horizons))
    bootstraps = {
        "GMD_vs_3EG_observed": _paired_fault_line_bootstrap(
            metric_rows,
            "M0-3EG-observed",
            "M0-GMD-observed",
            max(horizons),
            int(config["bootstrap_draws"]),
            int(config["seed"]),
        ),
        "GMD_PM_vs_GMD_ZOH": _paired_fault_line_bootstrap(
            metric_rows,
            "M0-GMD-ZOH",
            "M0-GMD-PM",
            max(horizons),
            int(config["bootstrap_draws"]),
            int(config["seed"]) + 1,
        ),
        "PSD_vs_GMD_observed": _paired_fault_line_bootstrap(
            metric_rows,
            "M0-GMD-observed",
            "M0-PSD-observed",
            max(horizons),
            int(config["bootstrap_draws"]),
            int(config["seed"]) + 2,
        ),
    }
    assessment = _assess(config, models, aggregate, bootstraps)

    causality_audit = _causal_schedule_invariance_audit(
        models, val[0], min(max(horizons), len(val[0]["training_context"]["targets"]["t"]))
    )
    _json(result_dir / "causality_invariance_audit.json", causality_audit)

    convergence_rows = []
    for source in sorted({_source(sample) for sample in val}):
        sample = next(sample for sample in val if _source(sample) == source)
        target = sample["training_context"]["targets"]
        steps = max(horizons)
        time_grid = np.asarray(target["t"][:steps])
        static = sample["model_inputs"]["static"]
        for method, model, policy in (
            ("M0-3EG-observed", models["M0-3EG"], "observed"),
            ("M0-GMD-observed", models["M0-GMD"], "observed"),
            ("M0-GMD-PM", models["M0-GMD-PM"], "causal_pm_ode"),
            ("M0-PSD-observed", models["M0-PSD"], "observed"),
            ("M0-PSD-PM", models["M0-PSD-PM"], "causal_pm_ode"),
        ):
            convergence_rows.append(
                {
                    "source_dataset": source,
                    "event_id": sample["event_id"],
                    "method": method,
                    **model.convergence_audit(
                        static,
                        time_grid,
                        _route_exogenous(sample, steps, policy),
                    ),
                }
            )

    edges = tuple(tuple(pair) for pair in train[0]["training_context"]["edges"])
    parameter_rows = _parameter_rows(models, edges)
    data_rows = [
        {
            "role": role,
            "event_id": event_id,
            "source_dataset": store.metadata(event_id).source_dataset,
            "fault_line": store.metadata(event_id).fault_line,
            "valid_length": store.metadata(event_id).valid_length,
            "quality_status": store.metadata(event_id).quality_status,
        }
        for role, ids in (("train", train_ids), ("val", val_ids))
        for event_id in ids
    ]
    _write_csv(result_dir / "dataset_support.csv", data_rows)
    _write_csv(result_dir / "per_case_metrics.csv", metric_rows)
    _write_csv(result_dir / "aggregate_metrics.csv", aggregate)
    _write_csv(result_dir / "source_metrics_h100.csv", source_rows)
    _write_csv(result_dir / "integration_convergence.csv", convergence_rows)
    _write_csv(result_dir / "learned_parameters_and_structure.csv", parameter_rows)
    _json(result_dir / "paired_fault_line_bootstrap.json", bootstraps)
    _json(result_dir / "scientific_assessment.json", assessment)
    _json(
        result_dir / "physics_audits.json",
        {name: model.physics_audit() for name, model in models.items()},
    )
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "dataset_revision": store.dataset_metadata.dataset_revision,
            "dataset_build_id": store.dataset_metadata.build_id,
            "fold": config["fold"],
            "seed": config["seed"],
            "representation": "generator_10",
            "train_ids_used": train_ids,
            "val_ids_used": val_ids,
            "test_ids_read": [],
            "labels_used_for_fit_selection_or_thresholds": False,
            "observed_Pm_is_conditional_noncausal_sensitivity": True,
            "causal_models_initialize_from_Pm_clear": True,
            "minimum_real_steps": int(config["minimum_train_steps"]),
            "config_sha256": _sha256(config_path),
        },
    )
    _make_figure(result_dir, aggregate, parameter_rows, assessment)
    _write_report(result_dir, run_id, assessment, historical, source_rows)

    audits = {name: model.physics_audit() for name, model in models.items()}
    process_gates = {
        "test_data_unopened": True,
        "labels_unused": True,
        "only_at_least_100_point_trajectories_used": all(
            int(row["valid_length"]) >= int(config["minimum_train_steps"])
            for row in data_rows
        ),
        "raw_generator_10_representation": True,
        "selection_marker_written_before_metric_assessment": True,
        "historical_representation_audit_complete": historical[
            "accepted_round6_representation"
        ]
        == "clustered_7",
        "all_model_physics_audits_pass": all(
            audit.get("status") == "pass" for audit in audits.values()
        ),
        "all_rollouts_completed": all(
            row["status"] == "completed" for row in metric_rows
        ),
        "all_integration_convergence_audits_pass": all(
            row["convergent"] for row in convergence_rows
        ),
        "coi_and_power_drift_below_1e_10": max(
            row.get("coi_drift", 0.0)
            for row in metric_rows
            if row["status"] == "completed"
        )
        < 1e-10,
        "observed_Pm_claim_bounded": True,
        "causal_Pm_models_ignore_future_schedule": causality_audit["status"]
        == "PASS",
        "PSD_not_misnamed_as_LYAP": audits["M0-PSD"].get(
            "full_Lyapunov_certificate_claim"
        )
        is False,
        "scientific_candidate_may_be_negative": True,
    }
    process = {
        "status": (
            "PASS"
            if all(
                value
                for key, value in process_gates.items()
                if key != "scientific_candidate_may_be_negative"
            )
            else "FAIL"
        ),
        "gates": process_gates,
        "scientific_assessment": assessment,
    }
    _json(result_dir / "diagnostic_process_acceptance.json", process)
    if process["status"] != "PASS":
        raise RuntimeError("raw ten-machine physical-discovery process gate failed")
    artifacts = {
        path.relative_to(result_dir).as_posix(): _sha256(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)


def _run_adversarial_audit(
    *, root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic = _latest_result(
        root, str(config["diagnostic_experiment"]), "diagnostic"
    )
    process = json.loads(
        (diagnostic / "diagnostic_process_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    lineage = json.loads((diagnostic / "lineage.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (diagnostic / "selection_marker.json").read_text(encoding="utf-8")
    )
    historical = json.loads(
        (diagnostic / "historical_representation_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assessment = json.loads(
        (diagnostic / "scientific_assessment.json").read_text(encoding="utf-8")
    )
    metrics = _read_csv(diagnostic / "per_case_metrics.csv")
    causality = json.loads(
        (diagnostic / "causality_invariance_audit.json").read_text(encoding="utf-8")
    )
    gates = {
        "artifact_hashes_match": not _verify_artifact_hashes(diagnostic),
        "diagnostic_process_passed": process.get("status") == "PASS",
        "test_data_unopened": not lineage.get("test_ids_read"),
        "labels_unused": not lineage.get("labels_used_for_fit_selection_or_thresholds"),
        "raw10_lineage_locked": lineage.get("representation") == "generator_10",
        "short_trajectories_excluded": lineage.get("minimum_real_steps") == 100,
        "historical_clustered7_claim_has_hashed_evidence": (
            historical.get("accepted_round6_representation") == "clustered_7"
            and len(historical.get("evidence_files", {})) >= 5
        ),
        "observed_Pm_not_called_causal": assessment["claim_boundary"][
            "observed_Pm_is_conditional"
        ],
        "causal_pm_routes_present": {
            row["method"]
            for row in metrics
            if row["input_policy"] == "causal_pm_ode"
        }
        == {"M0-GMD-PM", "M0-PSD-PM"},
        "causal_pm_future_schedule_counterfactual_passed": (
            causality.get("status") == "PASS"
            and causality.get("future_observed_schedule_is_forbidden_at_deployment")
            is True
            and {row.get("method") for row in causality.get("rows", [])}
            == {"M0-GMD-PM", "M0-PSD-PM"}
            and all(
                float(row.get("max_abs_prediction_error", float("inf"))) <= 1e-14
                for row in causality.get("rows", [])
            )
        ),
        "PSD_claim_is_gated_not_forced": (
            assessment["gates"]["raw10_M0_PSD_physical_structure_promoted"]
            == bool(
                assessment["PSD_structure"]["promoted_by_internal_validation"]
                and assessment["PSD_structure"]["support"]["active_edges"] > 0
                and assessment["PSD_structure"]["cross_source_edge_support_jaccard"]
                is not None
                and assessment["PSD_structure"]["cross_source_edge_support_jaccard"]
                >= float(config["minimum_cross_source_support_jaccard"])
                and assessment["relative_changes"][
                    "M0_PSD_degradation_vs_M0_GMD_observed"
                ]
                <= float(config["maximum_physical_discovery_degradation_vs_gmd"])
                and assessment["PSD_structure"]["physics_audit"]["status"] == "pass"
            )
        ),
        "no_physical_line_or_machine_parameter_claim": (
            not assessment["claim_boundary"]["physical_line_recovery_claim"]
            and not assessment["claim_boundary"]["machine_parameter_recovery_claim"]
        ),
        "no_LYAP_name_without_certificate": not assessment["claim_boundary"][
            "full_Lyapunov_certificate_claim"
        ],
        "single_fold_boundary_explicit": not assessment["claim_boundary"][
            "five_fold_generalization_claim"
        ],
        "selection_complete_before_audit": bool(selection.get("selection_complete")),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "status": status,
        "decision": (
            "ACCEPTABLE_WITH_GATED_NEGATIVE_RESULTS_ALLOWED"
            if status == "PASS"
            else "REJECT"
        ),
        "run_id": run_id,
        "diagnostic_result": diagnostic.name,
        "gates": gates,
        "adversarial_questions": [
            "Were accepted historical 7-node results confused with later raw-10 diagnostics?",
            "Did short trajectories enter the weak equation stack?",
            "Was observed future Pm presented as a causal deployment input?",
            "Was a POD modal closure overinterpreted as edge topology discovery?",
            "Was effective complete-graph support overinterpreted as physical transmission lines?",
            "Was M0-PSD promoted despite cross-source support disagreement or accuracy loss?",
            "Was a port-structured residual mislabeled as a full Lyapunov-certified model?",
        ],
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# 原始 10 机物理结构发现的对抗性审查\n\n"
        f"**决定：{review['decision']}**\n\n"
        + "\n".join(
            f"- {'PASS' if value else 'FAIL'} `{key}`"
            for key, value in gates.items()
        )
        + "\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("adversarial audit rejected raw ten-machine study")


def _run_acceptance(
    *, root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic = _latest_result(
        root, str(config["diagnostic_experiment"]), "diagnostic"
    )
    adversarial = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_audit"
    )
    process = json.loads(
        (diagnostic / "diagnostic_process_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    review = json.loads(
        (adversarial / "adversarial_review.json").read_text(encoding="utf-8")
    )
    assessment = json.loads(
        (diagnostic / "scientific_assessment.json").read_text(encoding="utf-8")
    )
    gates = {
        "diagnostic_process_passed": process.get("status") == "PASS",
        "diagnostic_hashes_match": not _verify_artifact_hashes(diagnostic),
        "adversarial_review_passed": review.get("status") == "PASS",
        "test_remained_locked": not assessment["claim_boundary"]["test_opened"],
        "bounded_claims": (
            not assessment["claim_boundary"]["five_fold_generalization_claim"]
            and not assessment["claim_boundary"]["physical_line_recovery_claim"]
            and not assessment["claim_boundary"]["full_Lyapunov_certificate_claim"]
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    accepted_models = ["M0-3EG"]
    model_gates = assessment["gates"]
    if model_gates["raw10_M0_GMD_promoted_over_M0_3EG"]:
        accepted_models.append("M0-GMD")
    if model_gates["raw10_M0_GMD_PM_causal_port_promoted_over_ZOH"]:
        accepted_models.append("M0-GMD-PM")
    if model_gates["raw10_M0_PSD_physical_structure_promoted"]:
        accepted_models.append("M0-PSD")
    if model_gates["raw10_M0_PSD_PM_full_candidate_promoted"]:
        accepted_models.append("M0-PSD-PM")
    acceptance = {
        "status": status,
        "decision": "ACCEPT" if status == "PASS" else "REJECT",
        "scope": (
            "F1/seed324 validation-only raw generator_10 mainline; test unopened; "
            "scientific candidate gates may return negative"
        ),
        "run_id": run_id,
        "diagnostic_result": diagnostic.name,
        "adversarial_result": adversarial.name,
        "accepted_model_roles": accepted_models,
        "scientific_gates": model_gates,
        "gates": gates,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# 原始 10 机主线最终验收\n\n"
        f"**决定：{acceptance['decision']}**\n\n"
        f"已接受角色：{', '.join(accepted_models)}。\n\n"
        "范围严格限于 F1、seed324、validation-only；test 未读取。"
        "未晋级的结构候选保留为可审计负结果，不覆盖基础 10 机 M0-3EG。\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("acceptance rejected raw ten-machine study")


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    config_name = str(
        experiment.get(
            "config", "paper_ten_machine_physical_discovery_f1_seed324.toml"
        )
    )
    config_path = root / "configs" / "experiments" / config_name
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if stage == "diagnostic":
        _run_diagnostic(
            root=root,
            config=config,
            config_path=config_path,
            result_dir=result_dir,
            run_id=run_id,
        )
        return
    if stage == "adversarial_audit":
        _run_adversarial_audit(
            root=root, config=config, result_dir=result_dir, run_id=run_id
        )
        return
    if stage == "acceptance":
        _run_acceptance(
            root=root, config=config, result_dir=result_dir, run_id=run_id
        )
        return
    raise RuntimeError(
        "ten-machine physical discovery implements diagnostic, adversarial_audit and acceptance"
    )
