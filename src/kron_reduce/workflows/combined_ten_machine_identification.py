"""Frozen ten-machine, source-balanced, instance-prunable identification workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation import (
    DmdMode,
    cross_cluster_modal_consensus,
    exact_dmd_modes,
    match_dmd_modes,
)
from kron_reduce.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kron_reduce.models import (
    CandidateBlockInstance,
    PrunableEnergyModel,
    RunContext,
    StructureSelectionResult,
    block_registry,
)
from kron_reduce.physics.coherency import wilson_interval

WORKFLOW = "combined_ten_machine_identification_v1"
HIGHER_ORDER_SOURCE = "ieee39_highorder_dataset"
DENSE_SOURCE = "ieee39_near_critical_unstable_v1_dense"

HIGHER_ORDER_LINES = {
    "fit": {"Line_10", "Line_11", "Line_14", "Line_17", "Line_23", "Line_27", "Line_28", "Line_30", "Line_32", "Line_33"},
    "selection": {"Line_16", "Line_29", "Line_3"},
    "internal_audit": {"Line_20", "Line_25", "Line_31"},
}
DENSE_LINES = {
    "fit": {"Line_2", "Line_42", "Line_44", "Line_45"},
    "selection": {"Line_37"},
    "internal_audit": {"Line_43", "Line_46"},
}


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


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_hashes(result_dir: Path) -> None:
    _json(
        result_dir / "artifact_hashes.json",
        {
            path.relative_to(result_dir).as_posix(): _sha256(path)
            for path in sorted(result_dir.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )


def combined_role(
    source_dataset: str,
    fault_line: str,
    valid_length: int,
    canonical_role: str,
    *,
    minimum_steps: int = 1001,
) -> str:
    """Return the preregistered role without reading a trajectory or label."""

    if canonical_role == "train" and int(valid_length) < int(minimum_steps):
        return "postfreeze_safety_train"
    if canonical_role == "val" and int(valid_length) < int(minimum_steps):
        return "postfreeze_safety_validation"
    if canonical_role == "val":
        return "external_validation"
    if canonical_role != "train":
        raise ValueError(f"combined workflow cannot assign role {canonical_role!r}")
    mapping = HIGHER_ORDER_LINES if source_dataset == HIGHER_ORDER_SOURCE else DENSE_LINES
    if source_dataset not in {HIGHER_ORDER_SOURCE, DENSE_SOURCE}:
        raise ValueError(f"unexpected combined source: {source_dataset}")
    matches = [role for role, lines in mapping.items() if str(fault_line) in lines]
    if len(matches) != 1:
        raise ValueError(f"fault line has no unique preregistered role: {source_dataset}/{fault_line}")
    return matches[0]


def validate_role_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identification = [
        row for row in rows if row["role"] in {"fit", "selection", "internal_audit"}
    ]
    seen: dict[tuple[str, str], str] = {}
    for row in identification:
        key = (str(row["source_dataset"]), str(row["fault_line"]))
        role = str(row["role"])
        if key in seen and seen[key] != role:
            raise ValueError(f"fault-line role leakage: {key}: {seen[key]} vs {role}")
        seen[key] = role
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["role"])] = counts.get(str(row["role"]), 0) + 1
    expected = {
        "fit": 110,
        "selection": 25,
        "internal_audit": 43,
        "external_validation": 34,
        "postfreeze_safety_train": 31,
        "postfreeze_safety_validation": 9,
    }
    exact = all(counts.get(role, 0) == count for role, count in expected.items())
    return {
        "counts": counts,
        "expected_counts": expected,
        "exact_expected_counts": exact,
        "fault_line_role_leakage": False,
        "identification_events": len(identification),
        "short_safety_events": counts.get("postfreeze_safety_train", 0)
        + counts.get("postfreeze_safety_validation", 0),
    }


def _verify_upstream(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result_dir = root / "results" / str(config["upstream_cluster_result"])
    hashes = result_dir / "artifact_hashes.json"
    if not hashes.is_file():
        raise FileNotFoundError(f"missing frozen cluster evidence: {hashes}")
    actual_hash = _sha256(hashes)
    expected_hash = str(config["upstream_cluster_artifact_hash"]).lower()
    if actual_hash != expected_hash:
        raise RuntimeError("frozen cluster artifact hash changed")
    recorded = json.loads(hashes.read_text(encoding="utf-8"))
    changed = [
        name
        for name, digest in recorded.items()
        if not (result_dir / name).is_file() or _sha256(result_dir / name) != digest
    ]
    if changed:
        raise RuntimeError(f"frozen cluster evidence changed: {changed}")
    summary = json.loads((result_dir / "validation_summary.json").read_text(encoding="utf-8"))
    decision = summary["primary_decision"]
    if summary["verdict"] != "VALIDATED_REFUSAL" or decision["use_clustering"]:
        raise RuntimeError("upstream evidence does not freeze the original ten-machine coordinates")
    if int(decision["n_clusters"]) != 10:
        raise RuntimeError("upstream refusal did not retain all ten machines")
    return {
        "result_dir": result_dir.name,
        "artifact_hash": actual_hash,
        "verdict": summary["verdict"],
        "cluster_decision": decision,
    }


def _data_manifest(store: CanonicalStore, split: Any, minimum_steps: int) -> list[dict[str, Any]]:
    rows = []
    for canonical_role, ids in (("train", split.train_ids), ("val", split.val_ids)):
        for event_id in ids:
            metadata = store.metadata(event_id)
            rows.append(
                {
                    "event_id": event_id,
                    "source_dataset": metadata.source_dataset,
                    "fault_line": metadata.fault_line,
                    "valid_length": metadata.valid_length,
                    "canonical_role": canonical_role,
                    "role": combined_role(
                        metadata.source_dataset,
                        metadata.fault_line,
                        metadata.valid_length,
                        canonical_role,
                        minimum_steps=minimum_steps,
                    ),
                    "label_read_for_identification": False,
                    "future_Pm_used": False,
                }
            )
    return rows


def _adapt(store: CanonicalStore, event_ids: Sequence[str]) -> list[dict[str, Any]]:
    request = ModelDataRequest.initial_state_rollout("zoh")
    inertia = store.dataset_metadata.M_prior
    output = []
    for event_id in event_ids:
        sample = adapt_event(
            store.load_event(event_id),
            inertia_prior=inertia,
            request=request,
            representation="generator_10",
        )
        sample["training_context"].pop("label", None)
        audit_model_sample(sample)
        output.append(sample)
    return output


def _context(config: Mapping[str, Any], samples: Sequence[dict[str, Any]]) -> RunContext:
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


def _evaluate(
    model: PrunableEnergyModel,
    samples: Sequence[dict[str, Any]],
    horizons: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        time = np.asarray(target["t"], dtype=float)
        pm = np.asarray(sample["model_inputs"]["exogenous"]["Pm"], dtype=float)
        metadata = sample["training_context"]["metadata"]
        for horizon in horizons:
            stop = min(len(time), int(np.searchsorted(time - time[0], horizon, side="right")))
            row: dict[str, Any] = {
                "event_id": sample["event_id"],
                "source_dataset": metadata["source_dataset"],
                "fault_line": metadata["fault_line"],
                "requested_horizon_s": float(horizon),
                "evaluated_horizon_s": float(time[stop - 1] - time[0]) if stop else 0.0,
            }
            if stop < 2:
                row["failure"] = "insufficient_horizon"
                rows.append(row)
                continue
            try:
                delta, omega = model.rollout(
                    static,
                    time[:stop],
                    {"Pm": pm[:stop], "interpolation": "zoh"},
                )
                metrics = trajectory_metrics(
                    np.asarray(target["delta"][:stop]),
                    np.asarray(target["omega"][:stop]),
                    delta,
                    omega,
                )
                row.update(metrics)
                delta_scale = max(float(np.std(target["delta"][:stop])), 1e-12)
                omega_scale = max(float(np.std(target["omega"][:stop])), 1e-12)
                row["normalized_rmse"] = float(
                    np.sqrt(
                        0.5
                        * (
                            (float(metrics["delta_rmse"]) / delta_scale) ** 2
                            + (float(metrics["omega_rmse"]) / omega_scale) ** 2
                        )
                    )
                )
                mass = np.asarray(static["mass"], dtype=float)
                row["maximum_coi_drift"] = float(
                    max(np.max(np.abs(delta @ mass)), np.max(np.abs(omega @ mass)))
                )
            except (FloatingPointError, ValueError) as error:
                row["failure"] = type(error).__name__
            rows.append(row)
    aggregate = []
    sources = sorted({str(row["source_dataset"]) for row in rows})
    for source in ["combined", *sources]:
        for horizon in horizons:
            local = [
                row
                for row in rows
                if row["requested_horizon_s"] == float(horizon)
                and (source == "combined" or row["source_dataset"] == source)
            ]
            metrics = aggregate_trajectory_metrics(local)
            normalized = [
                float(row["normalized_rmse"])
                for row in local
                if np.isfinite(row.get("normalized_rmse", float("nan")))
            ]
            aggregate.append(
                {
                    "source_dataset": source,
                    "requested_horizon_s": float(horizon),
                    **metrics,
                    "mean_normalized_rmse": float(np.mean(normalized)) if normalized else float("nan"),
                    "maximum_coi_drift": max(
                        [float(row.get("maximum_coi_drift", 0.0)) for row in local],
                        default=float("nan"),
                    ),
                }
            )
    return rows, aggregate


def _model_dmd_modes(model_modes: Sequence[dict[str, Any]], mass: np.ndarray) -> list[DmdMode]:
    output = []
    for row in model_modes:
        shape = (
            np.asarray(row["shape_real"], dtype=float)
            + 1j * np.asarray(row["shape_imag"], dtype=float)
        ) * np.sqrt(mass)
        norm = float(np.linalg.norm(shape))
        if norm > 0:
            shape = shape / norm
        output.append(
            DmdMode(
                frequency_hz=float(row["frequency_hz"]),
                damping_ratio=float(row["damping_ratio"]),
                decay_rate=float(row["decay_rate"]),
                strength=1.0,
                shape_real=tuple(float(value) for value in np.real(shape)),
                shape_imag=tuple(float(value) for value in np.imag(shape)),
            )
        )
    return output


def _modal_audit(
    model_modes: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if not samples:
        return {"status": "UNSUPPORTED", "reason": "empty cohort"}
    by_group: dict[str, list[DmdMode]] = {}
    for sample in samples:
        target = sample["training_context"]["targets"]
        time = np.asarray(target["t"], dtype=float)
        mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=float)
        if len(time) < 8:
            continue
        modes = exact_dmd_modes(
            np.asarray(target["omega"], dtype=float) * np.sqrt(mass)[None, :],
            float(np.median(np.diff(time))),
            rank=min(2 * len(mass), len(time) - 1),
        )
        by_group.setdefault(str(sample["training_context"]["metadata"]["fault_line"]), []).extend(modes)
    if len(by_group) < 2:
        return {
            "status": "UNSUPPORTED",
            "reason": "fewer than two independent fault lines cannot establish a repeatable mode",
            "empirical_consensus": [],
            "supports_modal_claim": False,
            "independent_fault_lines": len(by_group),
        }
    consensus = cross_cluster_modal_consensus(by_group, minimum_support_fraction=0.80)
    if not consensus:
        return {
            "status": "UNSUPPORTED",
            "reason": "no empirical mode reached 80% fault-line bootstrap support",
            "empirical_consensus": [],
            "supports_modal_claim": False,
        }
    mass = np.asarray(samples[0]["model_inputs"]["static"]["mass"], dtype=float)
    matches = match_dmd_modes(
        [row.mode for row in consensus],
        _model_dmd_modes(model_modes, mass),
        max_relative_frequency_error=0.5,
    )
    frequency = max([float(row["relative_frequency_error"]) for row in matches], default=float("inf"))
    damping = max([float(row["absolute_damping_ratio_error"]) for row in matches], default=float("inf"))
    mac = min([float(row["mac"]) for row in matches], default=0.0)
    passed = bool(
        len(matches) >= len(consensus)
        and frequency <= 0.05
        and damping <= 0.05
        and mac >= 0.8
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "supports_modal_claim": passed,
        "empirical_consensus": [row.as_dict() for row in consensus],
        "matches": matches,
        "maximum_relative_frequency_error": frequency,
        "maximum_absolute_damping_error": damping,
        "minimum_mac": mac,
        "thresholds": {"relative_frequency": 0.05, "absolute_damping": 0.05, "minimum_mac": 0.8},
    }


def _selection_supported_modes(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only source-local modes repeated across independent selection lines."""

    output: list[dict[str, Any]] = []
    sources = sorted(
        {str(sample["training_context"]["metadata"]["source_dataset"]) for sample in samples}
    )
    for source in sources:
        local = [
            sample
            for sample in samples
            if sample["training_context"]["metadata"]["source_dataset"] == source
        ]
        by_group: dict[str, list[DmdMode]] = {}
        for sample in local:
            target = sample["training_context"]["targets"]
            time = np.asarray(target["t"], dtype=float)
            mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=float)
            if len(time) < 8:
                continue
            modes = exact_dmd_modes(
                np.asarray(target["omega"], dtype=float) * np.sqrt(mass)[None, :],
                float(np.median(np.diff(time))),
                rank=min(2 * len(mass), len(time) - 1),
            )
            fault_line = str(sample["training_context"]["metadata"]["fault_line"])
            by_group.setdefault(fault_line, []).extend(modes)
        if len(by_group) < 2:
            continue
        consensus = cross_cluster_modal_consensus(by_group, minimum_support_fraction=0.80)
        for row in consensus:
            mode = row.mode.as_dict()
            mode["source_dataset"] = source
            mode["support_fraction"] = float(row.support_fraction)
            mode["independent_fault_lines"] = int(row.total_clusters)
            output.append(mode)
    return output


def _modal_report(
    model: PrunableEnergyModel,
    static: dict[str, np.ndarray],
    internal: Sequence[dict[str, Any]],
    external: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    modes = model.modes(static)

    def cohorts(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sources = sorted(
            {str(sample["training_context"]["metadata"]["source_dataset"]) for sample in samples}
        )
        return {
            "combined": _modal_audit(modes, samples),
            **{
                source: _modal_audit(
                    modes,
                    [
                        sample
                        for sample in samples
                        if sample["training_context"]["metadata"]["source_dataset"] == source
                    ],
                )
                for source in sources
            },
        }

    return {"model_modes": modes, "internal": cohorts(internal), "external": cohorts(external)}


def _integration_convergence(
    model: PrunableEnergyModel, sample: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    target = sample["training_context"]["targets"]
    time = np.asarray(target["t"][: min(1001, len(target["t"]))], dtype=float)
    pm = np.asarray(sample["model_inputs"]["exogenous"]["Pm"][: len(time)], dtype=float)
    trajectories = []
    for divisor in (1.0, 2.0, 4.0):
        delta, omega, _ = model.rollout_augmented(
            sample["model_inputs"]["static"],
            time,
            {"Pm": pm, "interpolation": "zoh"},
            max_step=float(config["max_internal_step"]) / divisor,
        )
        trajectories.append(np.concatenate((delta, omega), axis=1))
    coarse = float(np.sqrt(np.mean((trajectories[0] - trajectories[1]) ** 2)))
    fine = float(np.sqrt(np.mean((trajectories[1] - trajectories[2]) ** 2)))
    ratio = coarse / max(fine, np.finfo(float).eps)
    roundoff = fine <= float(config["rk4_roundoff_error"])
    passed = bool(
        roundoff
        or float(config["rk4_order_ratio_lower"])
        <= ratio
        <= float(config["rk4_order_ratio_upper"])
    )
    return {
        "status": "pass" if passed else "fail",
        "event_id": sample["event_id"],
        "coarse_to_half_rmse": coarse,
        "half_to_quarter_rmse": fine,
        "error_ratio": ratio,
        "roundoff_limited": roundoff,
    }


def _safety_audit(
    model: PrunableEnergyModel,
    samples: Sequence[dict[str, Any]],
    store: CanonicalStore,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for sample in samples:
        metadata = store.metadata(sample["event_id"])
        try:
            margin = model.stability_margin(
                sample["model_inputs"]["static"],
                angle_limit=float(config["stability_angle_limit"]),
            )
            false_certificate = bool(
                metadata.outcome_label != "stable" and float(margin.get("margin", -np.inf)) > 0.0
            )
            rows.append(
                {
                    "event_id": sample["event_id"],
                    "source_dataset": metadata.source_dataset,
                    "true_outcome": metadata.outcome_label,
                    "false_stable_certificate": false_certificate,
                    **margin,
                }
            )
        except (ValueError, RuntimeError) as error:
            rows.append(
                {
                    "event_id": sample["event_id"],
                    "source_dataset": metadata.source_dataset,
                    "true_outcome": metadata.outcome_label,
                    "false_stable_certificate": False,
                    "certified": False,
                    "error": str(error),
                }
            )
    unstable = [row for row in rows if row["true_outcome"] != "stable"]
    false_count = sum(bool(row["false_stable_certificate"]) for row in unstable)
    _, upper = wilson_interval(false_count, len(unstable)) if unstable else (float("nan"), float("nan"))
    return {
        "status": "PASS" if false_count == 0 else "FAIL",
        "cases": rows,
        "unstable_cases": len(unstable),
        "false_stable_certificates": false_count,
        "FSR": false_count / max(1, len(unstable)),
        "FSR_95_upper": float(upper),
        "level_6_claimed": False,
        "reason_level_6_not_claimed": "40 unstable cases cannot establish a 5% 95%-upper FSR bound",
        "evaluated_after_structure_freeze": True,
        "labels_used_for_identification_or_selection": False,
    }


def _pruning_rows(trace: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for round_index, round_row in enumerate(trace, start=1):
        for trial in round_row.get("trials", []):
            degradation = trial.get("source_degradation", {})
            rows.append(
                {
                    "round": round_index,
                    "active_before": round_row.get("active_before", []),
                    "removed_instance": trial.get("removed_instance"),
                    "removed_family": trial.get("removed_family"),
                    "admissible": trial.get("admissible", False),
                    "modal_noninferior": trial.get("modal_noninferior", False),
                    "maximum_point_degradation": degradation.get("maximum_point_degradation"),
                    "maximum_bootstrap_upper": degradation.get("maximum_bootstrap_upper"),
                    "accepted": round_row.get("accepted_removal") == trial.get("removed_instance"),
                }
            )
    return rows


def _physics_audit(
    model: PrunableEnergyModel,
    static: dict[str, np.ndarray],
    audit_samples: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    physics = model.physics_audit(static)
    supply = [model.supply_identity_error(sample["model_inputs"]["static"]) for sample in audit_samples]
    physics["maximum_numeric_supply_identity_error"] = max(supply, default=0.0)
    physics["numeric_supply_identity_pass"] = bool(
        physics["maximum_numeric_supply_identity_error"] <= float(config["supply_identity_tolerance"])
    )
    if not physics["numeric_supply_identity_pass"]:
        physics["status"] = "fail"
    return physics


def _cohort_status(
    aggregate: Sequence[Mapping[str, Any]],
    degradation: Mapping[str, Any],
    modal: Mapping[str, Any],
    maximum_coi_drift: float,
) -> str:
    zero_failures = all(float(row.get("failure_rate", 1.0)) == 0.0 for row in aggregate)
    modal_pass = all(row.get("status") != "FAIL" for row in modal.values())
    return (
        "PASS"
        if zero_failures
        and bool(degradation.get("passes", False))
        and modal_pass
        and maximum_coi_drift <= 1e-10
        else "FAIL"
    )


def _run_scaffold(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    upstream = _verify_upstream(root, config)
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(
        result_dir / "registration.json",
        {
            "run_id": run_id,
            "workflow": WORKFLOW,
            "coordinate_system": "original_generator_10",
            "upstream_cluster_evidence": upstream,
            "single_shared_model": True,
            "candidate_bootstrap_draws": int(config["candidate_bootstrap_draws"]),
            "pruning_bootstrap_draws": int(config["pruning_bootstrap_draws"]),
            "test_data_access": False,
            "future_Pm_access": False,
            "labels_used_for_identification": False,
        },
    )
    _write_hashes(result_dir)


def _run_identification(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    upstream = _verify_upstream(root, config)
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold=str(config["fold"]),
    )
    manifest = _data_manifest(store, split, int(config["minimum_model_steps"]))
    role_audit = validate_role_manifest(manifest)
    if not role_audit["exact_expected_counts"]:
        raise RuntimeError(f"combined corpus counts changed: {role_audit}")
    by_role = {
        role: [str(row["event_id"]) for row in manifest if row["role"] == role]
        for role in role_audit["expected_counts"]
    }
    samples = {role: _adapt(store, ids) for role, ids in by_role.items()}
    fit = samples["fit"]
    selection = samples["selection"]
    internal = samples["internal_audit"]
    external = samples["external_validation"]
    safety_samples = samples["postfreeze_safety_train"] + samples["postfreeze_safety_validation"]
    context = _context(config, fit)
    identification_error: str | None = None
    final_model: PrunableEnergyModel | None = None
    proposal: dict[str, Any] = {}
    b2: PrunableEnergyModel | None = None
    pruning_trace: list[dict[str, Any]] = []
    selection_supported_modes: list[dict[str, Any]] = []
    try:
        b2 = PrunableEnergyModel(
            active_blocks=(),
            max_internal_step=float(config["max_internal_step"]),
            enforce_identifiability=True,
        )
        b2.fit(fit, selection, context)
        proposal = PrunableEnergyModel.propose_candidates(
            fit,
            b2.node_damping,
            b2.coupling,
            max_steps=int(config["fit_max_steps"]),
            correlation_threshold=float(config["candidate_correlation_threshold"]),
            bootstrap_draws=int(config["candidate_bootstrap_draws"]),
            seed=int(config["seed"]),
        )
        full = PrunableEnergyModel(
            block_instances=[
                CandidateBlockInstance.from_dict(row) for row in proposal["candidate_instances"]
            ],
            node_tau=float(config["node_high_order_tau"]),
            potential_scale=float(config["potential_scale"]),
            max_internal_step=float(config["max_internal_step"]),
            enforce_identifiability=True,
        )
        full.fit(fit, selection, context)
        full.refine_closed_loop(
            selection,
            context,
            maximum_cases=int(config["closed_loop_refine_cases"]),
            maximum_evaluations=int(config["closed_loop_refine_evaluations"]),
        )
        selection_supported_modes = _selection_supported_modes(selection)
        selected = full.prune(
            fit,
            selection,
            context,
            tolerance=float(config["pruning_noninferiority_tolerance"]),
            closed_loop_refine_cases=int(config["pruning_refine_cases"]),
            closed_loop_refine_evaluations=int(config["pruning_refine_evaluations"]),
            bootstrap_draws=int(config["pruning_bootstrap_draws"]),
            seed=int(config["seed"]),
            required_modal_modes=selection_supported_modes,
        )
        pruning_trace = list(selected.pruning_trace)
        selected_ridge = float(selected.selected_ridge or 1e-6)
        final_context = RunContext(
            **{
                **context.__dict__,
                "max_cases": len(fit) + len(selection),
                "ridge_values": (selected_ridge,),
            }
        )
        final_model = PrunableEnergyModel(
            block_instances=selected.active_instances,
            node_tau=selected.node_tau,
            potential_scale=selected.potential_scale,
            max_internal_step=selected.max_internal_step,
            enforce_identifiability=True,
        )
        final_model.fit(fit + selection, selection, final_context)
        final_model.refine_closed_loop(
            fit + selection,
            final_context,
            maximum_cases=int(config["closed_loop_refine_cases"]),
            maximum_evaluations=int(config["closed_loop_refine_evaluations"]),
        )
        final_model.pruning_trace = pruning_trace
        b2_final = PrunableEnergyModel(
            active_blocks=(),
            max_internal_step=float(config["max_internal_step"]),
            enforce_identifiability=True,
        )
        b2_final.fit(fit + selection, selection, final_context)
        b2 = b2_final
    except (ValueError, RuntimeError, FloatingPointError) as error:
        identification_error = str(error)

    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _csv(result_dir / "dataset_manifest.csv", manifest)
    _json(result_dir / "role_audit.json", role_audit)
    _json(result_dir / "upstream_cluster_evidence.json", upstream)
    if final_model is None or b2 is None:
        summary = {
            "run_id": run_id,
            "terminal_state": "IDENTIFICATION_FAILED",
            "error": identification_error,
            "upstream_cluster_hash": upstream["artifact_hash"],
            "test_ids_read": [],
            "test_trajectories_read": 0,
        }
        _json(result_dir / "integral_residual_candidates.json", proposal)
        _json(result_dir / "validation_summary.json", summary)
        _json(result_dir / "lineage.json", summary)
        _write_hashes(result_dir)
        return

    horizons = tuple(float(value) for value in config["rollout_horizons_s"])
    internal_rows, internal_aggregate = _evaluate(final_model, internal, horizons)
    external_rows, external_aggregate = _evaluate(final_model, external, horizons)
    internal_degradation = PrunableEnergyModel.source_stratified_degradation(
        b2,
        final_model,
        internal,
        _context(config, internal),
        tolerance=float(config["pruning_noninferiority_tolerance"]),
        bootstrap_draws=int(config["pruning_bootstrap_draws"]),
        seed=int(config["seed"]),
    )
    external_degradation = PrunableEnergyModel.source_stratified_degradation(
        b2,
        final_model,
        external,
        _context(config, external),
        tolerance=float(config["pruning_noninferiority_tolerance"]),
        bootstrap_draws=int(config["pruning_bootstrap_draws"]),
        seed=int(config["seed"]) + 1,
    )
    static = (fit + selection)[0]["model_inputs"]["static"]
    modal = _modal_report(final_model, static, internal, external)
    modal["selection_supported_modes"] = selection_supported_modes
    modal["selection_support_rule"] = {
        "minimum_fault_lines": 2,
        "minimum_fault_line_support_fraction": 0.80,
        "source_local": True,
    }
    physics = _physics_audit(final_model, static, internal + external, config)
    integration = _integration_convergence(final_model, internal[0], config)
    safety = _safety_audit(final_model, safety_samples, store, config)
    maximum_internal_coi = max(
        [float(row.get("maximum_coi_drift", 0.0)) for row in internal_rows], default=float("inf")
    )
    maximum_external_coi = max(
        [float(row.get("maximum_coi_drift", 0.0)) for row in external_rows], default=float("inf")
    )
    internal_status = _cohort_status(
        internal_aggregate,
        internal_degradation,
        modal["internal"],
        maximum_internal_coi,
    )
    if physics["status"] != "pass" or integration["status"] != "pass":
        internal_status = "FAIL"
    external_status = _cohort_status(
        external_aggregate,
        external_degradation,
        modal["external"],
        maximum_external_coi,
    )
    if safety["status"] != "PASS":
        external_status = "FAIL"
    if internal_status != "PASS":
        terminal_state = "IDENTIFICATION_FAILED"
    elif external_status != "PASS":
        terminal_state = "EXTERNAL_VALIDATION_FAILED"
    elif final_model.active_instances:
        terminal_state = "PRUNABLE_MODEL_SELECTED"
    else:
        terminal_state = "B2_SUFFICIENT"

    model_path = result_dir / "frozen_model.npz"
    b2_path = result_dir / "b2_baseline_model.npz"
    final_model.save(model_path)
    b2.save(b2_path)
    candidate_ranks = {
        row.instance_id: int(row.port_rank)
        for row in final_model.active_instances
        if row.port_rank > 0
    }
    stability = {
        "scope": "fixed post-clear input and topology",
        "physics": physics,
        "integration": integration,
        "safety": safety,
    }
    result = StructureSelectionResult(
        cluster_decision=upstream["cluster_decision"],
        active_blocks=final_model.selected_blocks,
        pruning_trace=tuple(pruning_trace),
        rollout_metrics={
            "internal": internal_aggregate,
            "external": external_aggregate,
            "internal_b2_noninferiority": internal_degradation,
            "external_b2_noninferiority": external_degradation,
            "maximum_coi_drift": max(maximum_internal_coi, maximum_external_coi),
            "zero_integration_failures": all(
                float(row.get("failure_rate", 1.0)) == 0.0
                for row in [*internal_aggregate, *external_aggregate]
            ),
        },
        modes=tuple(modal["model_modes"]),
        stability=stability,
        model_path=model_path.name,
        upstream_cluster_hash=upstream["artifact_hash"],
        source_balancing={
            "rule": "equal_source_then_equal_fault_line_then_equal_event_then_equal_window",
            "source_is_model_input": False,
        },
        active_block_instances=tuple(row.to_dict() for row in final_model.active_instances),
        candidate_port_ranks=candidate_ranks,
        internal_audit_status=internal_status,
        external_audit_status=external_status,
        terminal_state=terminal_state,
    )
    _json(
        result_dir / "b2_baseline.json",
        {"diagnostics": b2.diagnostics, "model_path": b2_path.name},
    )
    _json(
        result_dir / "integral_residual_candidates.json",
        {
            **proposal,
            "block_specs": {name: value.to_dict() for name, value in block_registry().items()},
        },
    )
    _json(
        result_dir / "low_rank_ports.json",
        {
            "candidates": [
                row for row in proposal.get("candidate_instances", []) if int(row.get("port_rank", 0)) > 0
            ],
            "active": [
                row.to_dict() for row in final_model.active_instances if row.port_rank > 0
            ],
        },
    )
    _json(result_dir / "pruning_trace.json", pruning_trace)
    _csv(result_dir / "pruning_trace.csv", _pruning_rows(pruning_trace))
    _csv(result_dir / "internal_rollout_metrics.csv", internal_rows)
    _csv(result_dir / "internal_rollout_aggregate.csv", internal_aggregate)
    _csv(result_dir / "external_rollout_metrics.csv", external_rows)
    _csv(result_dir / "external_rollout_aggregate.csv", external_aggregate)
    _json(result_dir / "modal_report.json", modal)
    _json(result_dir / "stability_certificate.json", stability)
    _json(result_dir / "structure_selection_result.json", result.to_dict())
    summary = {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "terminal_state": terminal_state,
        "coordinate_system": "original_generator_10",
        "upstream_cluster_hash": upstream["artifact_hash"],
        "active_blocks": list(final_model.selected_blocks),
        "active_block_instances": [row.instance_id for row in final_model.active_instances],
        "internal_audit_status": internal_status,
        "external_audit_status": external_status,
        "safety_status": safety["status"],
        "false_stable_certificates": safety["false_stable_certificates"],
        "FSR_95_upper": safety["FSR_95_upper"],
        "level_6_claimed": False,
        "model_count": 1,
        "source_routing": False,
        "test_ids_read": [],
        "test_trajectories_read": 0,
        "future_observed_Pm_used": False,
    }
    _json(result_dir / "validation_summary.json", summary)
    _json(
        result_dir / "lineage.json",
        {
            **summary,
            "fit_event_count": len(fit),
            "selection_event_count": len(selection),
            "internal_audit_event_count": len(internal),
            "external_validation_event_count": len(external),
            "postfreeze_safety_event_count": len(safety_samples),
            "labels_first_read_stage": "postfreeze_safety_audit",
            "structure_changed_after_freeze": False,
        },
    )
    _write_hashes(result_dir)


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "fold": "F1",
        "seed": 324,
        "pm_mode": "zoh",
        "test_access": "forbidden",
        "other_folds_access": "forbidden",
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise RuntimeError(f"combined ten-machine scope violation: {mismatch}")
    if stage == "scaffold":
        _run_scaffold(root, config_path, config, result_dir, run_id)
    elif stage == "identification_run":
        _run_identification(root, config_path, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported combined ten-machine stage: {stage}")
