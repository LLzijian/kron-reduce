"""Read-only P1 module-effect and fit-only identifiability execution."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest, validate_p1_samples
from kron_reduce.evaluation.module_contribution import (
    DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS,
    DeletionCell,
    DeletionDegradation,
    EffectCI,
    ModuleContributionEvidence,
    ModuleContributionStatus,
    classify_module_contribution,
)
from kron_reduce.evaluation.p1_effect_statistics import (
    summarize_nonnegative_effect_rows,
)
from kron_reduce.evaluation.p1_effect_statistics import (
    view_key as effect_view_key,
)
from kron_reduce.evaluation.p1_module_identifiability import (
    audit_saved_model_identifiability,
)
from kron_reduce.evaluation.p1_paired_statistics import view_key as paired_view_key
from kron_reduce.models.base import RunContext
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel
from kron_reduce.workflows.p1_counterfactual_execution import (
    _new_output,
    _seal,
    verify_counterfactual_artifacts,
)
from kron_reduce.workflows.p1_five_arm_execution import (
    _read_json,
    _write_json,
    verify_five_arm_bundle,
)

_COUNTER_FIELDS = {
    "candidate_delta_rmse",
    "candidate_omega_rmse",
    "combined_change",
    "delta_change",
    "event_id",
    "fault_line",
    "horizon_seconds",
    "kind",
    "mode",
    "omega_change",
    "reference_delta_rmse",
    "reference_omega_rmse",
    "removed_instance_ids",
    "source_dataset",
    "status",
    "variant_id",
}
_COUNTER_FAILED_FIELDS = {
    "event_id",
    "failure",
    "fault_line",
    "horizon_seconds",
    "kind",
    "mode",
    "removed_instance_ids",
    "source_dataset",
    "status",
    "variant_id",
}
_PHYSICAL_FIELDS = {
    "absent_module_semantics",
    "dissipation_mean",
    "energy_scale",
    "event_id",
    "fault_line",
    "horizon_seconds",
    "identity_error_max",
    "identity_error_normalized",
    "independent_support_verdict",
    "kind",
    "mode",
    "parameter_norm",
    "port_abs_mean",
    "port_normalized_abs_mean",
    "power_scale",
    "pure_dissipation_protocol_conflict",
    "remaining_parameter_norm",
    "remaining_port_abs_mean",
    "remaining_storage_mean",
    "removed_instance_ids",
    "selected_modules_present",
    "source_dataset",
    "state_source",
    "status",
    "storage_applicable",
    "storage_mean",
    "storage_normalized_mean",
    "storage_q95",
    "sum_port_abs_mean",
    "variant_id",
}
_PHYSICAL_FAILED_FIELDS = _COUNTER_FAILED_FIELDS | {"state_source"}
_TUPLE_CONTEXT_FIELDS = {
    "ridge_values",
    "integral_windows",
    "sindy_thresholds",
    "m2a_residual_scales",
    "graph_prune_thresholds",
}
_RAW_SUPPLY_IDENTITY_LIMIT = 1e-8


def verify_module_support_artifacts(
    directory: Path, expected_manifest_sha256: str
) -> dict[str, str]:
    """Verify the immutable support-audit tree and non-promotional boundary."""
    directory = Path(directory)
    hashes = verify_counterfactual_artifacts(directory, expected_manifest_sha256)
    if set(hashes) != {
        "effect_statistics.json",
        "identifiability.json",
        "module_support.json",
        "summary.json",
    }:
        raise ValueError("invalid module-support artifact inventory")
    summary = _read_json(directory / "summary.json")
    if (
        summary.get("schema") != "kron_reduce.p1_module_support_execution.v1"
        or summary.get("scientific_acceptance") is not False
        or summary.get("scientific_verdict") is not None
        or summary.get("model_unit_gate_passed") is not False
        or summary.get("promotion_decision") != "KEEP_BASELINE"
        or summary.get("selection_data_reopened") is not False
        or summary.get("test_or_other_fold_data_used") is not False
        or summary.get("thresholds") != DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS.as_dict()
        or summary.get("raw_supply_identity_limit") != _RAW_SUPPLY_IDENTITY_LIMIT
    ):
        raise ValueError("invalid module-support summary boundary")
    return hashes


def _csv_rows(
    path: Path, completed_fields: set[str], failed_fields: set[str]
) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular CSV: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        allowed = {
            frozenset(completed_fields),
            frozenset(completed_fields | {"failure"}),
            frozenset(failed_fields),
        }
        if frozenset(fields) not in allowed:
            raise ValueError(f"unexpected CSV schema: {path.name}")
        rows = list(reader)
    if not rows or any(set(row) != fields for row in rows):
        raise ValueError(f"empty or malformed CSV: {path.name}")
    return rows


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real finite number") from error
    if not np.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be a real finite number")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(result) != str(value):
        raise ValueError(f"{name} must be an integer")
    return result


def _boolean(value: str, name: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"{name} must use an explicit boolean spelling")
    return value == "True"


def _removed(value: str) -> list[str]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("removed_instance_ids must be strict JSON") from error
    if (
        not isinstance(result, list)
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("removed_instance_ids must be unique module IDs")
    return result


def _metadata(manifest: P1RoleManifest, role: str) -> dict[str, tuple[str, str]]:
    return {
        row.event_id: (row.source_dataset, row.fault_line)
        for row in manifest.authorize(role, manifest.ids(role))
    }


def _validate_counter_rows(
    rows: list[dict[str, str]], variants: Sequence[dict], manifest: P1RoleManifest
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    metadata = _metadata(manifest, "internal_audit")
    expected_variants = {row["variant_id"]: row for row in variants}
    output: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    numeric = {
        "candidate_delta_rmse",
        "candidate_omega_rmse",
        "combined_change",
        "delta_change",
        "omega_change",
        "reference_delta_rmse",
        "reference_omega_rmse",
    }
    for row in rows:
        event, variant = row["event_id"], expected_variants.get(row["variant_id"])
        horizon = _integer(row["horizon_seconds"], "horizon")
        if (
            event not in metadata
            or horizon not in {1, 5, 10}
            or variant is None
            or row["mode"] not in {"zero", "refit"}
            or row["status"] not in {"completed", "failed"}
            or (row["source_dataset"], row["fault_line"]) != metadata[event]
            or row["kind"] != variant["kind"]
            or _removed(row["removed_instance_ids"]) != variant["removed_instance_ids"]
        ):
            raise ValueError("counterfactual metric metadata mismatch")
        key = (row["variant_id"], row["mode"], event, horizon)
        if key in output:
            raise ValueError("duplicate counterfactual metric cell")
        parsed: dict[str, Any] = dict(row)
        if row["status"] == "completed":
            if any(name not in row for name in numeric) or row.get("failure", ""):
                raise ValueError("completed counterfactual cell has incomplete fields")
            parsed.update({name: _number(row[name], name, nonnegative=True) for name in numeric})
        elif not row.get("failure") or any(row.get(name, "") for name in numeric):
            raise ValueError("failed counterfactual cell is malformed")
        output[key] = parsed
    required = {
        (variant["variant_id"], mode, event, horizon)
        for variant in variants
        for mode in ("zero", "refit")
        for event in metadata
        for horizon in (1, 5, 10)
    }
    if set(output) != required:
        raise ValueError("counterfactual metrics do not cover the complete role")
    return output


def _validate_physical_rows(
    rows: list[dict[str, str]], variants: Sequence[dict], manifest: P1RoleManifest
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    metadata = _metadata(manifest, "internal_audit")
    expected_variants = {row["variant_id"]: row for row in variants}
    output: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    common_numeric = {
        "dissipation_mean",
        "energy_scale",
        "identity_error_max",
        "identity_error_normalized",
        "parameter_norm",
        "port_abs_mean",
        "port_normalized_abs_mean",
        "power_scale",
        "storage_mean",
        "storage_normalized_mean",
        "storage_q95",
        "sum_port_abs_mean",
    }
    remaining_numeric = {
        "remaining_parameter_norm",
        "remaining_port_abs_mean",
        "remaining_storage_mean",
    }
    completed_only = _PHYSICAL_FIELDS - (_PHYSICAL_FAILED_FIELDS - {"failure"})
    for row in rows:
        event, variant = row["event_id"], expected_variants.get(row["variant_id"])
        horizon = _integer(row["horizon_seconds"], "horizon")
        if (
            event not in metadata
            or horizon not in {1, 5, 10}
            or variant is None
            or row["mode"] not in {"zero", "refit", "reference"}
            or row["status"] not in {"completed", "failed"}
            or (row["source_dataset"], row["fault_line"]) != metadata[event]
            or row["kind"] != variant["kind"]
            or _removed(row["removed_instance_ids"]) != variant["removed_instance_ids"]
        ):
            raise ValueError("module physical metadata mismatch")
        key = (row["variant_id"], row["mode"], event, horizon)
        if key in output:
            raise ValueError("duplicate module physical cell")
        parsed: dict[str, Any] = dict(row)
        if row["status"] == "completed":
            required_numeric = common_numeric | (
                remaining_numeric if row["mode"] != "reference" else set()
            )
            if any(name not in row for name in required_numeric) or row.get("failure", ""):
                raise ValueError("completed physical cell has incomplete fields")
            parsed.update(
                {name: _number(row[name], name, nonnegative=True) for name in required_numeric}
            )
            parsed["storage_applicable"] = _boolean(row["storage_applicable"], "storage_applicable")
            parsed["pure_dissipation_protocol_conflict"] = _boolean(
                row["pure_dissipation_protocol_conflict"],
                "pure_dissipation_protocol_conflict",
            )
            if row["independent_support_verdict"] != "NOT_EVALUATED":
                raise ValueError("physical row prematurely declares module support")
            if row["mode"] == "reference":
                if (
                    row["state_source"] != "full_reference_prediction"
                    or any(row.get(name, "") for name in remaining_numeric)
                    or row.get("selected_modules_present", "")
                    or row.get("absent_module_semantics", "")
                ):
                    raise ValueError("reference physical row has intervention-only fields")
            else:
                selected = _integer(row["selected_modules_present"], "selected_modules_present")
                expected_selected = (
                    len(variant["removed_instance_ids"]) if row["mode"] == "zero" else 0
                )
                expected_semantics = (
                    "all_selected_instances_present"
                    if row["mode"] == "zero" or not variant["removed_instance_ids"]
                    else "structurally_deleted_module_has_zero_storage_and_port"
                )
                if (
                    row["state_source"] != f"{row['mode']}_prediction"
                    or selected != expected_selected
                    or row["absent_module_semantics"] != expected_semantics
                ):
                    raise ValueError("intervention physical semantics mismatch")
                parsed["selected_modules_present"] = selected
        else:
            expected_state_source = (
                "full_reference_prediction"
                if row["mode"] == "reference"
                else f"{row['mode']}_prediction"
            )
            if (
                not row.get("failure")
                or any(row.get(name, "") for name in completed_only)
                or row["state_source"] != expected_state_source
            ):
                raise ValueError("failed physical cell is malformed")
        output[key] = parsed
    required = {
        (variant["variant_id"], mode, event, horizon)
        for variant in variants
        for mode in ("zero", "refit", "reference")
        for event in metadata
        for horizon in (1, 5, 10)
    }
    if set(output) != required:
        raise ValueError("module physical rows do not cover the complete role")
    return output


def _validate_numerical_rows(
    document: dict, manifest: P1RoleManifest
) -> dict[tuple[str, int], dict[str, Any]]:
    if (
        set(document) != {"rows", "numerical_gate_verdict"}
        or document["numerical_gate_verdict"] != "NOT_EVALUATED"
        or not isinstance(document["rows"], list)
    ):
        raise ValueError("invalid numerical-error document")
    metadata = _metadata(manifest, "internal_audit")
    output = {}
    completed_fields = {
        "event_id",
        "source_dataset",
        "fault_line",
        "horizon_seconds",
        "status",
        "step_01_to_005_change",
        "step_005_to_0025_change",
        "convergence_ratio",
        "ratio_unavailable_reason",
    }
    failed_fields = {
        "event_id",
        "source_dataset",
        "fault_line",
        "horizon_seconds",
        "status",
        "reason",
    }
    for row in document["rows"]:
        if not isinstance(row, dict) or row.get("status") not in {"completed", "failed"}:
            raise ValueError("invalid numerical-error row")
        fields = completed_fields if row["status"] == "completed" else failed_fields
        if set(row) != fields:
            raise ValueError("invalid numerical-error row schema")
        event = row["event_id"]
        horizon = row["horizon_seconds"]
        if (
            event not in metadata
            or isinstance(horizon, bool)
            or horizon not in {1, 5, 10}
            or (row["source_dataset"], row["fault_line"]) != metadata[event]
            or (event, horizon) in output
        ):
            raise ValueError("numerical-error metadata mismatch or duplicate")
        parsed = dict(row)
        if row["status"] == "completed":
            for name in ("step_01_to_005_change", "step_005_to_0025_change"):
                parsed[name] = _number(row[name], name, nonnegative=True)
            ratio = row["convergence_ratio"]
            parsed["convergence_ratio"] = (
                None if ratio is None else _number(ratio, "convergence_ratio", nonnegative=True)
            )
            if (ratio is None) != (row["ratio_unavailable_reason"] == "near_roundoff"):
                raise ValueError("numerical convergence reason mismatch")
        output[event, horizon] = parsed
    required = {(event, horizon) for event in metadata for horizon in (1, 5, 10)}
    if set(output) != required:
        raise ValueError("numerical-error rows do not cover the complete role")
    return output


def _numerical_views(
    rows: dict[tuple[str, int], dict[str, Any]], manifest: P1RoleManifest
) -> dict[str, dict[str, Any]]:
    metadata = _metadata(manifest, "internal_audit")
    sources = sorted({source for source, _ in metadata.values()})
    views = {}
    for horizon in (1, 5, 10):
        for source in (*sources, None):
            selected = [
                rows[event, horizon]
                for event, (owner, _) in metadata.items()
                if source is None or owner == source
            ]
            failed = sorted(row["event_id"] for row in selected if row["status"] == "failed")
            view: dict[str, Any] = {
                "source_dataset": source,
                "horizon_seconds": horizon,
                "event_ids": sorted(row["event_id"] for row in selected),
                "failed_event_ids": failed,
            }
            if failed:
                view.update(status="UNAVAILABLE_FAILED", numerical_q95=None)
            else:
                values = np.asarray([row["step_01_to_005_change"] for row in selected], dtype=float)
                with np.errstate(over="raise", invalid="raise"):
                    q95 = float(np.quantile(values, 0.95))
                if not np.isfinite(q95) or q95 < 0:
                    raise ValueError("invalid numerical-error quantile")
                ratios = [row["convergence_ratio"] for row in selected]
                view.update(
                    status="COMPLETED",
                    numerical_q95=q95,
                    rk4_ratio_available=all(value is not None for value in ratios),
                    rk4_ratio_gate_passed=bool(
                        all(value is not None and 8 <= value <= 32 for value in ratios)
                    ),
                )
            views[effect_view_key(source, horizon)] = view
    return views


def _effect_rows(
    variant: dict,
    counter: dict[tuple[str, str, str, int], dict[str, Any]],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    manifest: P1RoleManifest,
    field: str,
) -> list[dict[str, Any]]:
    metadata = _metadata(manifest, "internal_audit")
    output = []
    for event, (source, line) in metadata.items():
        for horizon in (1, 5, 10):
            if field == "combined_change":
                row = counter[variant["variant_id"], "zero", event, horizon]
            else:
                row = physical[variant["variant_id"], "reference", event, horizon]
            output.append(
                {
                    "event_id": event,
                    "source_dataset": source,
                    "fault_line": line,
                    "horizon_seconds": horizon,
                    "status": row["status"],
                    "value": row[field] if row["status"] == "completed" else None,
                }
            )
    return output


def _effect_ci(view: dict[str, Any]) -> EffectCI:
    if view.get("status") != "COMPLETED" or view.get("ci95") is None:
        raise ValueError("effect interval is unavailable")
    return EffectCI(view["point_estimate"], view["ci95"][0], view["ci95"][1])


def _deletion_cells(
    inference: dict,
    sources: Sequence[str],
    primary_source: str,
    primary_horizon: int,
) -> tuple[DeletionCell, ...]:
    cells = []
    for source in sources:
        for horizon in (1, 5, 10):
            for output in ("delta", "omega"):
                view = inference["views"].get(paired_view_key(source, horizon, output))
                if (
                    not isinstance(view, dict)
                    or view.get("status") != "COMPLETED"
                    or view.get("relative_CI") is None
                    or view.get("relative_degradation") is None
                ):
                    raise ValueError("delete/refit interval is unavailable")
                degradation = DeletionDegradation(
                    view["relative_degradation"],
                    view["relative_CI"][0],
                    view["relative_CI"][1],
                )
                cells.append(
                    DeletionCell(
                        f"{source}|{horizon}|{output}",
                        degradation,
                        primary=source == primary_source and horizon == primary_horizon,
                        source=source,
                        horizon=horizon,
                        output=output,
                    )
                )
    return tuple(cells)


def _overall_status(views: Sequence[dict[str, Any]], identifiable: bool) -> str:
    statuses = [row.get("module_contribution_status") for row in views]
    if ModuleContributionStatus.INDEPENDENTLY_SUPPORTED.value in statuses:
        return ModuleContributionStatus.INDEPENDENTLY_SUPPORTED.value
    if not identifiable:
        return ModuleContributionStatus.UNIDENTIFIABLE.value
    if any(
        status is None or status == ModuleContributionStatus.UNRESOLVED.value for status in statuses
    ):
        return ModuleContributionStatus.UNRESOLVED.value
    if ModuleContributionStatus.REPLACEABLE.value in statuses:
        return ModuleContributionStatus.REPLACEABLE.value
    if statuses and all(
        status == ModuleContributionStatus.ZERO_EFFECT.value for status in statuses
    ):
        return ModuleContributionStatus.ZERO_EFFECT.value
    return ModuleContributionStatus.UNRESOLVED.value


def _support_views(
    variant: dict,
    effect: dict[str, dict],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    identifiability: dict,
    inference: dict,
    numerical: dict[str, dict[str, Any]],
    manifest: P1RoleManifest,
) -> tuple[list[dict[str, Any]], str]:
    metadata = _metadata(manifest, "internal_audit")
    sources = sorted({source for source, _ in metadata.values()})
    identifier = variant["removed_instance_ids"][0]
    ident = next(
        (row for row in identifiability["module_results"] if row["instance_id"] == identifier),
        None,
    )
    if ident is None:
        raise ValueError("singleton intervention lacks its identifiability block")
    rows = []
    for source in sources:
        for horizon in (1, 5, 10):
            key = effect_view_key(source, horizon)
            direct = effect["direct_prediction"]["views"][key]
            storage = effect["storage"]["views"][key]
            port = effect["port"]["views"][key]
            numeric = numerical[key]
            selected_physical = [
                physical[variant["variant_id"], "reference", event, horizon]
                for event, (owner, _) in metadata.items()
                if owner == source
            ]
            header: dict[str, Any] = {
                "source_dataset": source,
                "horizon_seconds": horizon,
                "instance_id": identifier,
                "scientific_acceptance": False,
            }
            if any(row["status"] != "completed" for row in selected_physical):
                rows.append({**header, "status": "UNAVAILABLE_FAILED_PHYSICS"})
                continue
            if any(row["pure_dissipation_protocol_conflict"] for row in selected_physical):
                rows.append({**header, "status": "PROTOCOL_NOT_APPLICABLE_PURE_DISSIPATION"})
                continue
            if any(view.get("status") != "COMPLETED" for view in (direct, storage, port, numeric)):
                rows.append({**header, "status": "UNDETERMINED_EFFECT_OR_NUMERICS"})
                continue
            raw_identity = max(row["identity_error_max"] for row in selected_physical)
            normalized_identity = max(row["identity_error_normalized"] for row in selected_physical)
            if raw_identity > _RAW_SUPPLY_IDENTITY_LIMIT:
                rows.append(
                    {
                        **header,
                        "status": "UNDETERMINED_SUPPLY_IDENTITY_GATE",
                        "supply_identity_error_max": raw_identity,
                    }
                )
                continue
            if ident["condition_number"] is None or ident["profile_loss_increase"] is None:
                rows.append(
                    {
                        **header,
                        "status": "PARAMETER_NOT_IDENTIFIABLE",
                        "module_contribution_status": ModuleContributionStatus.UNIDENTIFIABLE.value,
                    }
                )
                continue
            try:
                cells = _deletion_cells(inference, sources, source, horizon)
                evidence = ModuleContributionEvidence(
                    direct_prediction_effect_ci=_effect_ci(direct),
                    storage_effect_ci=_effect_ci(storage),
                    port_effect_ci=_effect_ci(port),
                    deletion_cells=cells,
                    sigma_ratio=ident["sigma_ratio"],
                    condition_number=ident["condition_number"],
                    profile_loss_increase=ident["profile_loss_increase"],
                    numerical_q95=numeric["numerical_q95"],
                    supply_identity_error=normalized_identity,
                )
                result = classify_module_contribution(
                    evidence, DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS
                )
                rows.append(
                    {
                        **header,
                        "status": "COMPLETED",
                        "module_contribution_status": result.status.value,
                        "tau_y": result.tau_y,
                        "tau_phys": result.tau_phys,
                        "supply_identity_error_max": raw_identity,
                        "supply_identity_error_normalized_max": normalized_identity,
                        "rk4_ratio_gate_passed": numeric["rk4_ratio_gate_passed"],
                        "evidence": evidence.as_dict(),
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        **header,
                        "status": "UNDETERMINED_DELETION_INTERVAL",
                        "reason": str(error),
                    }
                )
    return rows, _overall_status(rows, ident["parameter_identifiability_passed"])


def _joint_status(
    variant: dict,
    effect: dict[str, dict],
    physical: dict[tuple[str, str, str, int], dict[str, Any]],
    inference: dict,
    numerical: dict[str, dict[str, Any]],
    manifest: P1RoleManifest,
) -> dict[str, Any]:
    metadata = _metadata(manifest, "internal_audit")
    sources = sorted({source for source, _ in metadata.values()})
    zero_all, delete_all, refuted, unavailable = True, True, False, []
    cells = []
    for source in sources:
        for horizon in (1, 5, 10):
            key = effect_view_key(source, horizon)
            views = [
                effect[name]["views"][key] for name in ("direct_prediction", "storage", "port")
            ]
            numeric = numerical[key]
            selected = [
                physical[variant["variant_id"], "reference", event, horizon]
                for event, (owner, _) in metadata.items()
                if owner == source
            ]
            if (
                any(view.get("status") != "COMPLETED" for view in views)
                or numeric.get("status") != "COMPLETED"
                or any(row["status"] != "completed" for row in selected)
                or any(row["identity_error_max"] > _RAW_SUPPLY_IDENTITY_LIMIT for row in selected)
                or any(row["pure_dissipation_protocol_conflict"] for row in selected)
            ):
                unavailable.append({"source_dataset": source, "horizon_seconds": horizon})
                continue
            thresholds = DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS
            tau_y = max(
                thresholds.prediction_floor,
                thresholds.prediction_numerical_multiplier * numeric["numerical_q95"],
            )
            tau_phys = max(
                thresholds.physical_floor,
                thresholds.physical_numerical_multiplier
                * max(row["identity_error_normalized"] for row in selected),
            )
            tolerances = (tau_y, tau_phys, tau_phys)
            zero_all &= all(
                view["ci95"][1] <= tolerance
                for view, tolerance in zip(views, tolerances, strict=True)
            )
            refuted |= any(
                view["ci95"][0] > tolerance
                for view, tolerance in zip(views, tolerances, strict=True)
            )
            for output in ("delta", "omega"):
                view = inference["views"].get(paired_view_key(source, horizon, output))
                if (
                    not isinstance(view, dict)
                    or view.get("status") != "COMPLETED"
                    or view.get("relative_CI") is None
                ):
                    unavailable.append(
                        {"source_dataset": source, "horizon_seconds": horizon, "output": output}
                    )
                    continue
                upper, lower = view["relative_CI"][1], view["relative_CI"][0]
                delete_all &= upper <= thresholds.deletion_degradation
                refuted |= lower > thresholds.deletion_degradation
                cells.append(
                    {
                        "source_dataset": source,
                        "horizon_seconds": horizon,
                        "output": output,
                        "relative_degradation": view["relative_degradation"],
                        "relative_CI": view["relative_CI"],
                    }
                )
    if refuted:
        status = "REFUTED"
    elif unavailable:
        status = "UNRESOLVED"
    elif zero_all and delete_all:
        status = "PROVEN"
    else:
        status = "UNRESOLVED"
    evidence_complete = not unavailable
    return {
        "status": status,
        "zero_effect_all_registered_views": zero_all if evidence_complete else None,
        "delete_refit_noninferior_all_registered_cells": (
            delete_all if evidence_complete else None
        ),
        "refutation_boundary_crossed": refuted,
        "evidence_complete": evidence_complete,
        "unavailable_cells": unavailable,
        "deletion_cells": cells,
        "scientific_acceptance": False,
    }


def _context(record: Mapping[str, Any]) -> RunContext:
    values = dict(record)
    for name in _TUPLE_CONTEXT_FIELDS:
        if name in values:
            if not isinstance(values[name], list):
                raise ValueError("saved fit context tuple field must be a JSON list")
            values[name] = tuple(values[name])
    try:
        return RunContext(**values)
    except TypeError as error:
        raise ValueError("saved fit context does not match RunContext") from error


def evaluate_saved_module_support(
    base_directory: Path,
    base_manifest_sha256: str,
    counterfactual_directory: Path,
    counterfactual_manifest_sha256: str,
    evaluation_directory: Path,
    evaluation_manifest_sha256: str,
    manifest: P1RoleManifest,
    fit_samples: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Compute component verdicts without reopening selection or internal trajectories."""
    validate_p1_samples(fit_samples, manifest, "fit")
    base_directory, counterfactual_directory, evaluation_directory, output_dir = map(
        Path,
        (
            base_directory,
            counterfactual_directory,
            evaluation_directory,
            output_dir,
        ),
    )
    base = verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
    counterfactual_hashes = verify_counterfactual_artifacts(
        counterfactual_directory, counterfactual_manifest_sha256
    )
    verify_counterfactual_artifacts(evaluation_directory, evaluation_manifest_sha256)
    evaluation_summary = _read_json(evaluation_directory / "summary.json")
    if (
        evaluation_summary.get("base_manifest_sha256") != base_manifest_sha256
        or evaluation_summary.get("counterfactual_manifest_sha256")
        != counterfactual_manifest_sha256
        or evaluation_summary.get("role_manifest_sha256") != manifest.source_sha256
        or evaluation_summary.get("role") != "internal_audit"
        or evaluation_summary.get("scientific_verdict") is not None
        or evaluation_summary.get("promotion_decision") != "KEEP_BASELINE"
    ):
        raise ValueError("counterfactual evaluation is not bound to the frozen inputs")
    spec = _read_json(base_directory / "core3.json")
    variants = [
        {
            "variant_id": f"single_{index:03d}",
            "kind": "single",
            "removed_instance_ids": [name],
        }
        for index, name in enumerate(
            sorted(row["instance_id"] for row in spec["active_block_instances"])
        )
    ]
    variants.append(
        {
            "variant_id": "joint",
            "kind": "joint",
            "removed_instance_ids": sorted(
                row["instance_id"] for row in spec["active_block_instances"]
            ),
        }
    )
    counterfactual_plan = _read_json(counterfactual_directory / "plan.json")
    expected_counterfactual_files = {"plan.json"} | {
        f"{variant['variant_id']}_{suffix}"
        for variant in variants
        for suffix in (
            "zero.npz",
            "zero.json",
            "refit.npz",
            "refit.json",
            "lineage.json",
        )
    }
    if (
        set(counterfactual_hashes) != expected_counterfactual_files
        or counterfactual_plan.get("schema") != "kron_reduce.p1_counterfactual_models.v1"
        or counterfactual_plan.get("base_manifest_sha256") != base_manifest_sha256
        or counterfactual_plan.get("base_core_sha256") != base["arms"][2]["checkpoint_hash"]
        or counterfactual_plan.get("role_manifest_sha256") != manifest.source_sha256
        or counterfactual_plan.get("common_contract_hash") != base["common_contract_hash"]
        or counterfactual_plan.get("fit_scales") != base["fit_scales"]
        or counterfactual_plan.get("variants") != variants
        or counterfactual_plan.get("reference_arm") != 3
        or counterfactual_plan.get("scientific_verdict") is not None
        or counterfactual_plan.get("promotion_decision") != "KEEP_BASELINE"
    ):
        raise ValueError("counterfactual bundle is not bound to the frozen base")
    counter = _validate_counter_rows(
        _csv_rows(
            evaluation_directory / "module_counterfactual_metrics.csv",
            _COUNTER_FIELDS,
            _COUNTER_FAILED_FIELDS,
        ),
        variants,
        manifest,
    )
    physical = _validate_physical_rows(
        _csv_rows(
            evaluation_directory / "module_energy_port_audit.csv",
            _PHYSICAL_FIELDS,
            _PHYSICAL_FAILED_FIELDS,
        ),
        variants,
        manifest,
    )
    numerical_rows = _validate_numerical_rows(
        _read_json(evaluation_directory / "numerical_error.json"), manifest
    )
    numerical = _numerical_views(numerical_rows, manifest)
    paired = _read_json(evaluation_directory / "module_paired_statistics.json")
    if set(paired) != {"variants"} or not isinstance(paired["variants"], list):
        raise ValueError("invalid paired-statistics inventory")
    paired_lookup = {}
    expected_pairs = {
        (variant["variant_id"], mode) for variant in variants for mode in ("zero", "refit")
    }
    expected_variants = {variant["variant_id"]: variant for variant in variants}
    for row in paired["variants"]:
        if not isinstance(row, dict) or set(row) != {
            "variant_id",
            "kind",
            "removed_instance_ids",
            "mode",
            "inference",
        }:
            raise ValueError("invalid paired-statistics variant schema")
        variant = expected_variants.get(row["variant_id"])
        key = (row["variant_id"], row["mode"])
        if (
            variant is None
            or key not in expected_pairs
            or key in paired_lookup
            or row["kind"] != variant["kind"]
            or row["removed_instance_ids"] != variant["removed_instance_ids"]
            or not isinstance(row["inference"], dict)
        ):
            raise ValueError("paired-statistics variant metadata mismatch")
        paired_lookup[key] = row["inference"]
    if set(paired_lookup) != expected_pairs:
        raise ValueError("paired-statistics variants are missing or duplicate")

    lineage = _read_json(base_directory / "fit_lineage.json")
    core = PrunableEnergyModel.load(base_directory / "core3.npz")
    current = lineage.get("current")
    if not isinstance(current, dict) or not isinstance(current.get("context"), dict):
        raise TypeError("saved fit lineage lacks the current context")
    context = _context(current["context"])
    selected_ridge = _number(
        core.diagnostics.get("selected_ridge"), "selected_ridge", nonnegative=True
    )
    _new_output(
        output_dir,
        (base_directory, counterfactual_directory, evaluation_directory),
    )
    try:
        identifiability = audit_saved_model_identifiability(
            core,
            fit_samples,
            manifest,
            context,
            selected_ridge=selected_ridge,
        )
        effects: dict[str, dict[str, dict]] = {}
        for variant in variants:
            rows = {
                "direct_prediction": summarize_nonnegative_effect_rows(
                    _effect_rows(variant, counter, physical, manifest, "combined_change"),
                    manifest,
                    "internal_audit",
                    "direct_prediction_change",
                ),
                "storage": summarize_nonnegative_effect_rows(
                    _effect_rows(variant, counter, physical, manifest, "storage_normalized_mean"),
                    manifest,
                    "internal_audit",
                    "normalized_storage_mean",
                ),
                "port": summarize_nonnegative_effect_rows(
                    _effect_rows(
                        variant,
                        counter,
                        physical,
                        manifest,
                        "port_normalized_abs_mean",
                    ),
                    manifest,
                    "internal_audit",
                    "normalized_port_abs_mean",
                ),
            }
            effects[variant["variant_id"]] = rows
        module_rows = []
        for variant in (row for row in variants if row["kind"] == "single"):
            inference = paired_lookup.get((variant["variant_id"], "refit"))
            if inference is None:
                raise ValueError("singleton delete/refit statistics are missing")
            views, status = _support_views(
                variant,
                effects[variant["variant_id"]],
                physical,
                identifiability,
                inference,
                numerical,
                manifest,
            )
            module_rows.append(
                {
                    **variant,
                    "instance_id": variant["removed_instance_ids"][0],
                    "module_contribution_status": status,
                    "support_views": views,
                    "scientific_acceptance": False,
                }
            )
        joint = next(row for row in variants if row["kind"] == "joint")
        joint_inference = paired_lookup.get(("joint", "refit"))
        if joint_inference is None:
            raise ValueError("joint delete/refit statistics are missing")
        joint_result = _joint_status(
            joint,
            effects["joint"],
            physical,
            joint_inference,
            numerical,
            manifest,
        )
        _write_json(output_dir / "effect_statistics.json", effects)
        _write_json(output_dir / "identifiability.json", identifiability)
        _write_json(
            output_dir / "module_support.json",
            {"modules": module_rows, "joint_b2": joint_result},
        )
        summary = {
            "schema": "kron_reduce.p1_module_support_execution.v1",
            "base_manifest_sha256": base_manifest_sha256,
            "counterfactual_manifest_sha256": counterfactual_manifest_sha256,
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "role_manifest_sha256": manifest.source_sha256,
            "fit_role": "fit",
            "effect_role": "internal_audit",
            "module_count": len(module_rows),
            "module_statuses": {
                row["instance_id"]: row["module_contribution_status"] for row in module_rows
            },
            "joint_b2_status": joint_result["status"],
            "thresholds": DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS.as_dict(),
            "raw_supply_identity_limit": _RAW_SUPPLY_IDENTITY_LIMIT,
            "selection_data_reopened": False,
            "test_or_other_fold_data_used": False,
            "scientific_acceptance": False,
            "scientific_verdict": None,
            "model_unit_gate_passed": False,
            "promotion_decision": "KEEP_BASELINE",
        }
        _write_json(output_dir / "summary.json", summary)
        verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
        verify_counterfactual_artifacts(counterfactual_directory, counterfactual_manifest_sha256)
        verify_counterfactual_artifacts(evaluation_directory, evaluation_manifest_sha256)
        digest = _seal(output_dir)
        return {
            **summary,
            "directory": str(output_dir.resolve()),
            "artifact_manifest_sha256": digest,
        }
    except Exception as error:
        _write_json(
            output_dir / "FAILED.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_verdict": None,
                "promotion_decision": "KEEP_BASELINE",
            },
        )
        raise
