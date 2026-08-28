"""Single-corpus clustering decision followed by one prunable model fit."""

from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data import (
    CanonicalStore,
    ModelDataRequest,
    adapt_event,
    fit_stratification_policy,
    load_split,
    stratify_samples,
)
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation import (
    DmdMode,
    cross_cluster_modal_consensus,
    exact_dmd_modes,
    match_dmd_modes,
)
from kron_reduce.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kron_reduce.evaluation.stability_region import missed_instability_rate
from kron_reduce.models import (
    PrunableEnergyModel,
    RunContext,
    StructureSelectionResult,
    block_registry,
)
from kron_reduce.physics import aggregate_model_sample, reconstruct_cluster_state
from kron_reduce.physics.coherency import wilson_interval
from kron_reduce.selection import ClusterThresholds, decide_clustering

WORKFLOW = "data_level_structure_selection_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_plain(value), indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


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
                    key: json.dumps(_plain(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else _plain(value)
                    for key, value in row.items()
                }
            )


def _write_hashes(result_dir: Path) -> None:
    _json(
        result_dir / "artifact_hashes.json",
        {
            path.name: _sha256(path)
            for path in sorted(result_dir.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )


def _pruning_rows(trace: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for round_index, round_row in enumerate(trace):
        for trial in round_row.get("trials", []):
            horizon = trial["summary"].get("horizon_standardized_loss", {})
            rows.append(
                {
                    "round": round_index,
                    "active_before": round_row["active_before"],
                    "removed_block": trial["removed_block"],
                    "accepted": trial["removed_block"]
                    == round_row.get("accepted_removal"),
                    "admissible": trial["admissible"],
                    "reference_loss": round_row["reference_loss"],
                    "candidate_loss": trial["summary"]["standardized_loss"],
                    "loss_1s": horizon.get("1s"),
                    "loss_5s": horizon.get("5s"),
                    "loss_10s": horizon.get("10s"),
                    "failure_rate": trial["summary"]["failure_rate"],
                    "modal_noninferior": trial["modal_noninferior"],
                    "physics_status": trial["physics"]["status"],
                }
            )
    return rows


def _verify_hashes(result_dir: Path) -> list[str]:
    expected = json.loads((result_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        name
        for name, digest in expected.items()
        if not (result_dir / name).is_file() or _sha256(result_dir / name) != digest
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    valid = [
        path
        for path in sorted(root.glob(f"results/{experiment}-{stage}-*"))
        if not (path / "RUN_FAILED.json").exists()
    ]
    if not valid:
        raise FileNotFoundError(f"no completed {stage} result for {experiment}")
    return valid[-1]


def _validate_scope(config: dict[str, Any]) -> None:
    expected = {
        "fold": "F1",
        "seed": 324,
        "pm_mode": "zoh",
        "test_access": "forbidden",
        "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden",
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"data-level workflow scope violation: {mismatch}")


def _load_data(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold=str(config["fold"]),
    )
    request = ModelDataRequest.initial_state_rollout("zoh")
    inertia = store.dataset_metadata.M_prior

    def adapt(ids: Sequence[str]) -> list[dict[str, Any]]:
        output = []
        for event_id in ids:
            if store.metadata(event_id).valid_length < int(config["minimum_model_steps"]):
                continue
            sample = adapt_event(
                store.load_event(event_id),
                inertia_prior=inertia,
                request=request,
                representation="generator_10",
            )
            audit_model_sample(sample)
            output.append(sample)
        return output

    return {"store": store, "split": split, "train": adapt(split.train_ids)}


def _context(config: dict[str, Any], samples: Sequence[dict[str, Any]]) -> RunContext:
    return RunContext(
        seed=int(config["seed"]),
        horizon_steps=int(config["selection_horizon_steps"]),
        max_cases=len(samples),
        ridge_values=tuple(float(value) for value in config["ridge_values"]),
        integral_windows=tuple(int(value) for value in config["integral_windows"]),
        fit_max_steps=int(config["fit_max_steps"]),
        weak_max_windows_per_case_horizon=int(config["weak_max_windows_per_case_horizon"]),
        max_internal_step=float(config["max_internal_step"]),
    )


def _hypothesis(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "registered_before_response_metrics": True,
        "algorithm_unit": "one supplied trajectory corpus",
        "selection_output": "one clustering decision and one final model",
        "cluster_score": "maximum normalized information/input/mode/coupling loss",
        "model_root": "B2 integral weak form",
        "optional_blocks": [
            "NONLINEAR_POTENTIAL",
            "EDGE_DAMPING",
            "NODE_HIGH_ORDER",
            "EDGE_MEMORY",
            "RESONANT_MEMORY",
        ],
        "selection_rule": (
            "fit the supported full block model, then remove/refit one whole block at a time; "
            "accept a removal only under 2% rollout and modal noninferiority"
        ),
        "posthoc_strata_role": "frozen-model validation only",
        "stability_scope": "fixed post-clear input and topology",
        "claim_boundaries": config["claim_boundaries"],
        "test_ids_read": [],
        "other_folds_read": [],
    }


def _roles_from_decision(
    samples: Sequence[dict[str, Any]], roles: dict[str, tuple[str, ...]]
) -> dict[str, list[dict[str, Any]]]:
    def group(sample: dict[str, Any]) -> str:
        metadata = sample["training_context"]["metadata"]
        return str(
            metadata.get("fault_line")
            or metadata.get("independent_group")
            or sample["event_id"]
        )

    return {
        role: [sample for sample in samples if group(sample) in set(groups)]
        for role, groups in roles.items()
    }


def _selected_view(
    samples: Sequence[dict[str, Any]], decision: Any
) -> list[dict[str, Any]]:
    if not decision.use_clustering:
        return list(samples)
    return [
        aggregate_model_sample(
            sample,
            decision.partition,
            f"data_level_clustered_{decision.n_clusters}",
        )
        for sample in samples
    ]


def _lift_modes(modes: Sequence[dict[str, Any]], decision: Any, mass: np.ndarray) -> list[dict[str, Any]]:
    if not decision.use_clustering:
        return [dict(mode) for mode in modes]
    output = []
    for mode in modes:
        real = reconstruct_cluster_state(np.asarray(mode["shape_real"])[None, :], mass, decision.partition)[0]
        imag = reconstruct_cluster_state(np.asarray(mode["shape_imag"])[None, :], mass, decision.partition)[0]
        complex_shape = real + 1j * imag
        norm = np.sqrt(np.sum(np.abs(complex_shape) ** 2 * mass))
        if norm > 0:
            complex_shape = complex_shape / norm
        output.append(
            {
                **mode,
                "shape_real": np.real(complex_shape).tolist(),
                "shape_imag": np.imag(complex_shape).tolist(),
                "lifted_to_original_coordinates": True,
            }
        )
    return output


def _evaluate(
    model: PrunableEnergyModel,
    selected_samples: Sequence[dict[str, Any]],
    original_samples: Sequence[dict[str, Any]],
    decision: Any,
    horizons: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original = {sample["event_id"]: sample for sample in original_samples}
    rows = []
    aggregates = []
    for horizon in horizons:
        horizon_rows = []
        for sample in selected_samples:
            source = original[sample["event_id"]]
            target = source["training_context"]["targets"]
            selected_target = sample["training_context"]["targets"]
            steps = min(int(horizon), len(selected_target["t"]))
            row: dict[str, Any] = {
                "event_id": sample["event_id"],
                "horizon_steps": int(horizon),
                "completed_steps": int(steps),
            }
            try:
                delta, omega = model.rollout(
                    sample["model_inputs"]["static"],
                    np.asarray(selected_target["t"][:steps]),
                    {
                        "Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]),
                        "interpolation": "zoh",
                    },
                )
                if decision.use_clustering:
                    mass = np.asarray(source["model_inputs"]["static"]["mass"], dtype=float)
                    delta = reconstruct_cluster_state(delta, mass, decision.partition)
                    omega = reconstruct_cluster_state(omega, mass, decision.partition)
                metrics = trajectory_metrics(
                    np.asarray(target["delta"][:steps]),
                    np.asarray(target["omega"][:steps]),
                    delta,
                    omega,
                )
                row.update(metrics)
                truth_delta = np.asarray(target["delta"][:steps], dtype=float)
                truth_omega = np.asarray(target["omega"][:steps], dtype=float)
                delta_scale = max(float(np.std(truth_delta)), 1e-12)
                omega_scale = max(float(np.std(truth_omega)), 1e-12)
                row["normalized_rmse"] = float(
                    np.sqrt(
                        0.5
                        * (
                            (float(row["delta_rmse"]) / delta_scale) ** 2
                            + (float(row["omega_rmse"]) / omega_scale) ** 2
                        )
                    )
                )
                mass = np.asarray(source["model_inputs"]["static"]["mass"], dtype=float)
                row["maximum_coi_drift"] = float(
                    max(np.max(np.abs(delta @ mass)), np.max(np.abs(omega @ mass)))
                )
            except (FloatingPointError, ValueError) as error:
                row.update({"rmse": float("nan"), "failure": type(error).__name__})
            rows.append(row)
            horizon_rows.append(row)
        aggregate = aggregate_trajectory_metrics(horizon_rows)
        normalized = [
            float(row["normalized_rmse"])
            for row in horizon_rows
            if np.isfinite(row.get("normalized_rmse", float("nan")))
        ]
        aggregate["mean_normalized_rmse"] = (
            float(np.mean(normalized)) if normalized else float("nan")
        )
        aggregate["maximum_normalized_rmse"] = (
            float(np.max(normalized)) if normalized else float("nan")
        )
        aggregates.append({"horizon_steps": int(horizon), **aggregate})
    return rows, aggregates


def _posthoc_slices(
    samples: Sequence[dict[str, Any]],
    per_case: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    policy, _ = fit_stratification_policy(
        list(samples),
        report_horizons=tuple(int(value) for value in config["report_horizons"]),
        minimum_structure_steps=int(config["minimum_model_steps"]),
        hankel_delays=int(config["hankel_delays"]),
        hankel_relative_floor=float(config["hankel_relative_floor"]),
        maximum_hankel_condition=float(config["maximum_hankel_condition"]),
    )
    strata = {row["event_id"]: row for row in stratify_samples(list(samples), policy)}
    coupling_scores = {}
    for sample in samples:
        target = sample["training_context"]["targets"]
        mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=float)
        incidence = np.asarray(sample["model_inputs"]["static"]["incidence"], dtype=float)
        delta = np.asarray(target["delta"], dtype=float)
        relative = delta - delta[[0]]
        edge_activity = float(np.sqrt(np.mean((relative @ incidence) ** 2)))
        state_activity = float(np.sqrt(np.mean(relative**2 * mass[None, :])))
        coupling_scores[sample["event_id"]] = edge_activity / max(state_activity, 1e-12)
    finite_scores = np.asarray(
        [value for value in coupling_scores.values() if np.isfinite(value)], dtype=float
    )
    coupling_edges = (
        np.quantile(finite_scores, (1.0 / 3.0, 2.0 / 3.0))
        if finite_scores.size >= 3
        else np.asarray([float("inf"), float("inf")])
    )
    for event_id, row in strata.items():
        value = coupling_scores[event_id]
        row["coupling_strength_score"] = float(value)
        row["coupling_strength_level"] = (
            "C-low"
            if value <= coupling_edges[0]
            else "C-mid"
            if value <= coupling_edges[1]
            else "C-high"
        )
    rows = []
    keys = ("amplitude_level", "richness_level", "coupling_strength_level")
    for key in keys:
        values = sorted({row[key] for row in strata.values()})
        for value in values:
            selected_ids = {event_id for event_id, row in strata.items() if row[key] == value}
            for horizon in sorted({int(row["horizon_steps"]) for row in per_case}):
                subset = [
                    row
                    for row in per_case
                    if row["event_id"] in selected_ids and int(row["horizon_steps"]) == horizon
                ]
                if subset:
                    rows.append(
                        {
                            "slice_key": key,
                            "slice_value": value,
                            "horizon_steps": horizon,
                            "n_events": len(subset),
                            "model_frozen_before_slicing": True,
                            "retrained": False,
                            "rerouted": False,
                            **aggregate_trajectory_metrics(subset),
                        }
                    )
    return rows


def _integration_convergence(
    model: PrunableEnergyModel,
    sample: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    static = sample["model_inputs"]["static"]
    target = sample["training_context"]["targets"]
    steps = min(max(int(value) for value in config["report_horizons"]), len(target["t"]))
    time = np.asarray(target["t"][:steps], dtype=float)
    pm = np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps], dtype=float)
    base_step = float(config["max_internal_step"])
    trajectories = []
    for divisor in (1.0, 2.0, 4.0):
        delta, omega, _ = model.rollout_augmented(
            static,
            time,
            {"Pm": pm, "interpolation": "zoh"},
            max_step=base_step / divisor,
        )
        trajectories.append(np.concatenate((delta, omega), axis=1))
    coarse_error = float(np.sqrt(np.mean((trajectories[0] - trajectories[1]) ** 2)))
    fine_error = float(np.sqrt(np.mean((trajectories[1] - trajectories[2]) ** 2)))
    ratio = coarse_error / max(fine_error, np.finfo(float).eps)
    roundoff = fine_error <= float(config["rk4_roundoff_error"])
    passed = bool(
        roundoff
        or float(config["rk4_order_ratio_lower"])
        <= ratio
        <= float(config["rk4_order_ratio_upper"])
    )
    return {
        "status": "pass" if passed else "fail",
        "event_id": sample["event_id"],
        "coarse_step": base_step,
        "coarse_to_half_rmse": coarse_error,
        "half_to_quarter_rmse": fine_error,
        "error_ratio": float(ratio),
        "expected_order_ratio": 16.0,
        "roundoff_limited": roundoff,
    }


def _modal_audit(
    model_modes: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    modes_by_group: dict[str, list[DmdMode]] = {}
    for sample in samples:
        target = sample["training_context"]["targets"]
        time = np.asarray(target["t"], dtype=float)
        mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=float)
        metadata = sample["training_context"].get("metadata", {})
        group = str(metadata.get("fault_line") or sample["event_id"])
        empirical = exact_dmd_modes(
            np.asarray(target["omega"], dtype=float) * np.sqrt(mass)[None, :],
            float(np.median(np.diff(time))),
            rank=min(2 * len(mass), len(time) - 1),
        )
        modes_by_group.setdefault(group, []).extend(empirical)
    consensus = cross_cluster_modal_consensus(
        modes_by_group,
        minimum_support_fraction=0.80,
    )
    reference = [row.mode for row in consensus]
    model_as_dmd = []
    mass = np.asarray(samples[0]["model_inputs"]["static"]["mass"], dtype=float)
    for row in model_modes:
        shape = (
            np.asarray(row["shape_real"], dtype=float)
            + 1j * np.asarray(row["shape_imag"], dtype=float)
        ) * np.sqrt(mass)
        norm = float(np.linalg.norm(shape))
        if norm > 0.0:
            shape = shape / norm
        model_as_dmd.append(
            DmdMode(
                frequency_hz=float(row["frequency_hz"]),
                damping_ratio=float(row["damping_ratio"]),
                decay_rate=float(row["decay_rate"]),
                strength=1.0,
                shape_real=tuple(float(value) for value in np.real(shape)),
                shape_imag=tuple(float(value) for value in np.imag(shape)),
            )
        )
    matches = match_dmd_modes(reference, model_as_dmd, max_relative_frequency_error=0.5)
    return {
        "empirical_consensus": [row.as_dict() for row in consensus],
        "matches": matches,
        "matched_modes": len(matches),
        "maximum_relative_frequency_error": max(
            [float(row["relative_frequency_error"]) for row in matches], default=float("nan")
        ),
        "maximum_absolute_damping_error": max(
            [float(row["absolute_damping_ratio_error"]) for row in matches],
            default=float("nan"),
        ),
        "minimum_mac": min([float(row["mac"]) for row in matches], default=float("nan")),
        "fit_and_selection_data_only": True,
        "empirical_modes_are_response_descriptors": True,
    }


def _run_hypothesis(
    config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "hypothesis_registry.json", _hypothesis(config, run_id))
    _json(result_dir / "lineage.json", {"run_id": run_id, "stage": "hypothesis_registration", "test_ids_read": []})
    _write_hashes(result_dir)


def _run_controlled(
    root: Path, config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    data = _load_data(root, config)
    samples = data["train"]
    thresholds = ClusterThresholds(
        information=float(config["cluster_information_budget"]),
        input=float(config["cluster_input_budget"]),
        mode=float(config["cluster_mode_budget"]),
        spectral_gap_ratio=float(config["cluster_spectral_gap_ratio"]),
        state=float(config["cluster_state_budget"]),
        coupling=float(config["cluster_coupling_budget"]),
        closure=float(config["cluster_closure_budget"]),
        bootstrap_support=float(config["cluster_bootstrap_support"]),
        confidence=float(config["cluster_confidence"]),
    )
    decision = decide_clustering(
        samples,
        thresholds=thresholds,
        seed=int(config["seed"]),
        bootstrap_draws=int(config["cluster_bootstrap_draws"]),
        max_steps=int(config["cluster_max_steps"]),
        delays=int(config["hankel_delays"]),
    )
    original_roles = _roles_from_decision(samples, decision.roles)
    selected_all = _selected_view(samples, decision)
    selected_roles = _roles_from_decision(selected_all, decision.roles)
    fit = selected_roles["fit"]
    selection = selected_roles["selection"]
    audit = selected_roles["audit"]
    context = _context(config, fit)

    base = PrunableEnergyModel(active_blocks=(), max_internal_step=float(config["max_internal_step"]))
    base.fit(fit, selection, context)
    proposal = PrunableEnergyModel.propose_candidates(
        fit,
        base.node_damping,
        base.coupling,
        max_steps=int(config["fit_max_steps"]),
        correlation_threshold=float(config["candidate_correlation_threshold"]),
    )
    model = PrunableEnergyModel(
        active_blocks=proposal["active_blocks"],
        memory_taus=proposal["memory_taus"],
        resonances=proposal["resonances"],
        node_tau=float(config["node_high_order_tau"]),
        potential_scale=float(config["potential_scale"]),
        max_internal_step=float(config["max_internal_step"]),
    )
    model.fit(fit, selection, context)
    model.refine_closed_loop(
        selection,
        context,
        maximum_cases=int(config["closed_loop_refine_cases"]),
        maximum_evaluations=int(config["closed_loop_refine_evaluations"]),
    )
    model = model.prune(
        fit,
        selection,
        context,
        tolerance=float(config["pruning_noninferiority_tolerance"]),
        closed_loop_refine_cases=int(config["pruning_refine_cases"]),
        closed_loop_refine_evaluations=int(config["pruning_refine_evaluations"]),
    )
    selected_ridge = model.selected_ridge if model.selected_ridge is not None else 1e-6
    final_context = RunContext(
        **{
            **context.__dict__,
            "max_cases": len(fit) + len(selection),
            "ridge_values": (float(selected_ridge),),
        }
    )
    final_model = PrunableEnergyModel(
        active_blocks=model.active_blocks,
        memory_taus=model.memory_taus,
        resonances=model.resonances,
        node_tau=model.node_tau,
        potential_scale=model.potential_scale,
        max_internal_step=model.max_internal_step,
    )
    final_model.fit(fit + selection, selection, final_context)
    final_model.refine_closed_loop(
        fit + selection,
        final_context,
        maximum_cases=int(config["closed_loop_refine_cases"]),
        maximum_evaluations=int(config["closed_loop_refine_evaluations"]),
    )
    final_model.pruning_trace = model.pruning_trace

    per_case, aggregate = _evaluate(
        final_model,
        audit,
        original_roles["audit"],
        decision,
        tuple(int(value) for value in config["report_horizons"]),
    )
    posthoc = _posthoc_slices(
        original_roles["audit"], per_case, config
    ) if original_roles["audit"] else []
    static = (fit + selection)[0]["model_inputs"]["static"]
    selected_coordinate_modes = final_model.modes(static)
    modal_audit = _modal_audit(selected_coordinate_modes, fit + selection)
    modes = _lift_modes(
        selected_coordinate_modes,
        decision,
        np.asarray(samples[0]["model_inputs"]["static"]["mass"], dtype=float),
    )
    margins = []
    maximum_stability_cases = int(config["maximum_stability_cases"])
    stability_cases = audit if maximum_stability_cases <= 0 else audit[:maximum_stability_cases]
    for sample in stability_cases:
        true_stable = data["store"].metadata(sample["event_id"]).outcome_label == "stable"
        try:
            margins.append(
                {
                    "event_id": sample["event_id"],
                    "true_stable": true_stable,
                    **final_model.stability_margin(
                        sample["model_inputs"]["static"],
                        angle_limit=float(config["stability_angle_limit"]),
                    ),
                }
            )
        except (ValueError, RuntimeError) as error:
            margins.append(
                {
                    "event_id": sample["event_id"],
                    "true_stable": true_stable,
                    "certified": False,
                    "predicted_stable": False,
                    "error": str(error),
                }
            )
    physics = final_model.physics_audit(static)
    supply_errors = [
        final_model.supply_identity_error(sample["model_inputs"]["static"])
        for sample in audit
    ]
    physics["maximum_numeric_supply_identity_error"] = max(supply_errors, default=0.0)
    physics["numeric_supply_identity_pass"] = bool(
        physics["maximum_numeric_supply_identity_error"]
        <= float(config["supply_identity_tolerance"])
    )
    if not physics["numeric_supply_identity_pass"]:
        physics["status"] = "fail"
    integration = (
        _integration_convergence(final_model, audit[0], config)
        if audit
        else {"status": "not_evaluated", "reason": "empty_audit_set"}
    )
    if margins:
        true_stable = np.asarray([row["true_stable"] for row in margins], dtype=bool)
        predicted_stable = np.asarray(
            [bool(row.get("predicted_stable", False)) for row in margins], dtype=bool
        )
        stability_metrics = missed_instability_rate(true_stable, predicted_stable)
        unstable_count = int(np.sum(~true_stable))
        false_stable_count = int(np.sum((~true_stable) & predicted_stable))
        _, fsr_upper = (
            wilson_interval(false_stable_count, unstable_count)
            if unstable_count
            else (float("nan"), float("nan"))
        )
        stability_metrics.update(
            {
                "FSR_95_upper": float(fsr_upper),
                "false_stable_certificates": false_stable_count,
                "unstable_cases": unstable_count,
                "certificate_coverage": float(
                    np.mean([bool(row.get("certified", False)) for row in margins])
                ),
                "evaluated_after_model_freeze": True,
                "labels_used_for_training_or_selection": False,
            }
        )
    else:
        stability_metrics = {"status": "not_evaluated", "reason": "empty_audit_set"}
    stability = {
        "scope": "fixed post-clear input and topology",
        "physics": physics,
        "integration": integration,
        "case_margins": margins,
        "classification": stability_metrics,
        "supply_identity_tolerance": float(config["supply_identity_tolerance"]),
    }
    model_path = result_dir / "selected_model.npz"
    final_model.save(model_path)
    selection_result = StructureSelectionResult(
        cluster_decision=decision.to_dict(),
        active_blocks=final_model.selected_blocks,
        pruning_trace=tuple(final_model.pruning_trace),
        rollout_metrics={
            "aggregate": aggregate,
            "posthoc_slices": posthoc,
            "maximum_coi_drift": max(
                [float(row["maximum_coi_drift"]) for row in per_case if "maximum_coi_drift" in row],
                default=0.0,
            ),
            "zero_integration_failures": all(
                float(row.get("failure_rate", 1.0)) == 0.0 for row in aggregate
            ),
        },
        modes=tuple(modes),
        stability=stability,
        model_path=model_path.name,
    )

    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "cluster_decision.json", decision.to_dict())
    _json(result_dir / "partition.json", {"partition": decision.to_dict()["partition"], "frozen": True})
    _json(
        result_dir / "candidate_blocks.json",
        {
            **proposal,
            "block_specs": {
                name: spec.to_dict() for name, spec in block_registry().items()
            },
        },
    )
    _json(result_dir / "pruning_trace.json", final_model.pruning_trace)
    _csv(result_dir / "pruning_trace.csv", _pruning_rows(final_model.pruning_trace))
    _csv(result_dir / "audit_per_case_metrics.csv", per_case)
    _csv(result_dir / "audit_aggregate_metrics.csv", aggregate)
    _csv(result_dir / "posthoc_slice_metrics.csv", posthoc)
    _json(
        result_dir / "modal_report.json",
        {"modes": modes, "data_support_audit": modal_audit, "model_frozen": True},
    )
    _json(result_dir / "stability_certificate.json", stability)
    _json(result_dir / "structure_selection_result.json", selection_result.to_dict())
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "controlled_experiment",
            "algorithm_unit": "one corpus",
            "number_of_cluster_decisions": 1,
            "number_of_final_models": 1,
            "posthoc_strata_used_for_selection": False,
            "frozen_validation_read": False,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)


def _run_residual(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    if _verify_hashes(controlled):
        raise RuntimeError("controlled artifacts changed before residual diagnosis")
    result = json.loads((controlled / "structure_selection_result.json").read_text(encoding="utf-8"))
    diagnosis = {
        "run_id": run_id,
        "status": "COMPLETED",
        "clustered": bool(result["cluster_decision"]["use_clustering"]),
        "selected_partition": result["cluster_decision"]["partition"],
        "active_blocks": result["active_blocks"],
        "removed_blocks": [
            row.get("accepted_removal")
            for row in result["pruning_trace"]
            if row.get("accepted_removal") is not None
        ],
        "model_count": 1,
        "posthoc_slices_retrained": False,
        "posthoc_slices_rerouted": False,
        "test_ids_read": [],
        "other_folds_read": [],
    }
    _json(result_dir / "residual_diagnosis.json", diagnosis)
    _json(result_dir / "lineage.json", {"run_id": run_id, "stage": "residual_diagnosis", "controlled_result": controlled.name, "test_ids_read": []})
    _write_hashes(result_dir)


def _run_adversarial(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    if _verify_hashes(controlled) or _verify_hashes(residual):
        raise RuntimeError("parent evidence changed before adversarial review")
    lineage = json.loads((controlled / "lineage.json").read_text(encoding="utf-8"))
    decision = json.loads((controlled / "cluster_decision.json").read_text(encoding="utf-8"))
    result = json.loads((controlled / "structure_selection_result.json").read_text(encoding="utf-8"))
    checks = {
        "one_cluster_decision": lineage["number_of_cluster_decisions"] == 1,
        "one_final_model": lineage["number_of_final_models"] == 1,
        "decision_covers_original_nodes": sorted(
            node for group in decision["partition"] for node in group
        ) == list(range(decision["n_original"])),
        "B2_backbone_retained": result["active_blocks"][0] == "B2_BACKBONE",
        "posthoc_strata_not_used_for_selection": not lineage["posthoc_strata_used_for_selection"],
        "fixed_input_policy": config["pm_mode"] == "zoh",
        "test_locked": lineage["test_ids_read"] == [],
        "other_folds_locked": lineage["other_folds_read"] == [],
    }
    review = {"run_id": run_id, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "test_ids_read": [], "other_folds_read": []}
    _json(result_dir / "adversarial_review.json", review)
    _json(result_dir / "lineage.json", {"run_id": run_id, "stage": "adversarial_review", "controlled_result": controlled.name, "residual_result": residual.name, "test_ids_read": []})
    _write_hashes(result_dir)
    if review["status"] != "PASS":
        raise RuntimeError("data-level adversarial review failed")


def _run_acceptance(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    adversarial = _latest_result(root, str(config["adversarial_experiment"]), "adversarial_review")
    if any(_verify_hashes(path) for path in (controlled, residual, adversarial)):
        raise RuntimeError("evidence changed before acceptance")
    review = json.loads((adversarial / "adversarial_review.json").read_text(encoding="utf-8"))
    result = json.loads((controlled / "structure_selection_result.json").read_text(encoding="utf-8"))
    gates = {
        "adversarial_review": review["status"] == "PASS",
        "single_model": bool(result["active_blocks"]),
        "cluster_decision_present": "use_clustering" in result["cluster_decision"],
        "zero_integration_failures": result["rollout_metrics"]["zero_integration_failures"],
        "coi_drift": result["rollout_metrics"]["maximum_coi_drift"] <= 1e-10,
        "energy_and_curvature": result["stability"]["physics"]["status"] == "pass",
        "numeric_supply_identity": result["stability"]["physics"][
            "numeric_supply_identity_pass"
        ],
        "rk4_convergence": result["stability"]["integration"]["status"] == "pass",
        "no_false_stable_certificate": result["stability"]["classification"].get(
            "false_stable_certificates", 1
        )
        == 0,
        "FSR_95_upper": result["stability"]["classification"].get(
            "FSR_95_upper", float("inf")
        )
        <= float(config["stability_fsr_upper_target"]),
        "test_locked": review["test_ids_read"] == [],
        "other_folds_locked": review["other_folds_read"] == [],
    }
    acceptance = {
        "run_id": run_id,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "terminal_state": "DATA_LEVEL_MODEL_SELECTED" if all(gates.values()) else "IMPLEMENTATION_AUDIT_FAILED",
        "gates": gates,
        "clustered": result["cluster_decision"]["use_clustering"],
        "n_clusters": result["cluster_decision"]["n_clusters"],
        "active_blocks": result["active_blocks"],
        "model_count": 1,
        "posthoc_strata_role": "validation only",
        "stability_fsr_upper_target": float(config["stability_fsr_upper_target"]),
        "test_ids_read": [],
        "other_folds_read": [],
        "merge_gate_required": True,
    }
    _json(result_dir / "acceptance.json", acceptance)
    _json(result_dir / "lineage.json", {"run_id": run_id, "stage": "acceptance", "controlled_result": controlled.name, "test_ids_read": []})
    _write_hashes(result_dir)
    if acceptance["status"] != "PASS":
        raise RuntimeError("data-level acceptance failed")


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
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
        raise ValueError(f"unsupported data-level stage: {stage}")
