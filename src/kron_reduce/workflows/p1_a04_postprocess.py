"""Evidence-bound post-processing for the P1-A04 controlled experiment.

P1 accepts a reproducible diagnosis, not the candidate model.  Every function
in this module is read-only with respect to its parent runs and fails closed on
missing samples, altered manifests, leakage, or ambiguous statistical support.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest
from kron_reduce.evaluation.p1_paired_statistics import summarize_paired_rows, view_key
from kron_reduce.workflows import p1_a03_real_f1_controlled_chain as parent
from kron_reduce.workflows.p1_counterfactual_execution import (
    _seal,
    paired_score_rows,
    verify_counterfactual_artifacts,
)
from kron_reduce.workflows.p1_five_arm_execution import (
    _read_json,
    _write_json,
    verify_five_arm_bundle,
)
from kron_reduce.workflows.p1_modal_observability_execution import (
    verify_modal_observability_artifacts,
)
from kron_reduce.workflows.p1_module_support_execution import (
    verify_module_support_artifacts,
)
from project_harness.evidence import (
    merge_evidence_commit,
    sha256_file,
    verify_artifact_directory,
)
from project_harness.loop import load_state

_HORIZONS = (1, 5, 10)
_OUTPUTS = ("delta", "omega")
_MODULE_STATUSES = {
    "INDEPENDENTLY_SUPPORTED",
    "UNIDENTIFIABLE",
    "UNRESOLVED",
    "REPLACEABLE",
    "ZERO_EFFECT",
}
_VIEW_STATUSES = {
    "COMPLETED",
    "PARAMETER_NOT_IDENTIFIABLE",
    "UNAVAILABLE_FAILED_PHYSICS",
    "PROTOCOL_NOT_APPLICABLE_PURE_DISSIPATION",
    "UNDETERMINED_EFFECT_OR_NUMERICS",
    "UNDETERMINED_SUPPLY_IDENTITY_GATE",
    "UNDETERMINED_DELETION_INTERVAL",
}
_ADVERSARIAL_CHECK_NAMES = {
    "exact_three_questions",
    "all_machine_checks_pass",
    "all_claim_boundaries_nonempty",
    "registered_time_grid_complete",
    "retry_read_time_only",
    "test_and_other_folds_locked",
    "forbidden_channels_not_used",
    "single_model_without_source_routing",
    "future_pm_not_main_input",
    "observed_pm_confound_disclosed",
    "tail_metrics_not_omitted",
    "no_premature_verdict_or_level6",
    "model_not_promoted",
}


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _history_evidence(root: Path, workflow: str, stage: str) -> dict[str, Any]:
    matches = [
        event.get("evidence")
        for event in load_state(root, workflow).get("history", [])
        if event.get("action") == "advance" and event.get("to") == stage
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError(f"expected exactly one recorded {stage!r} evidence object")
    return matches[0]


def _report_merge_commit(root: Path, workflow: str) -> str:
    commit = merge_evidence_commit(_history_evidence(root, workflow, "merge_gate"))
    if commit is None:
        raise RuntimeError("merge gate lacks its approved commit")
    return commit


def _bound_result(root: Path, workflow: str, stage: str) -> tuple[Path, dict[str, Any]]:
    evidence = _history_evidence(root, workflow, stage)
    run_id = evidence.get("result_dir")
    binding = evidence.get("artifact_manifest")
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or not isinstance(binding, dict)
        or binding.get("run_id") != run_id
        or binding.get("path") != "artifact_hashes.json"
        or not isinstance(binding.get("sha256"), str)
    ):
        raise RuntimeError(f"{stage} lacks an exact result/manifest binding")
    directory = root / "results" / run_id
    if (directory / "RUN_FAILED.json").exists() or (directory / "FAILED.json").exists():
        raise RuntimeError(f"{stage} is a failed result")
    verify_artifact_directory(
        directory,
        expected_manifest_sha256=binding["sha256"],
        strict=True,
    )
    return directory, evidence


def _role_manifest(root: Path, config: Mapping[str, Any]) -> P1RoleManifest:
    candidate = root / "results" / str(config["candidate_run_id"])
    verify_artifact_directory(
        candidate,
        expected_manifest_sha256=str(config["candidate_artifact_manifest_sha256"]),
        strict=True,
    )
    return P1RoleManifest.from_csv(
        candidate / str(config["role_manifest_file"]),
        fold=str(config["fold"]),
        expected_sha256=str(config["role_manifest_sha256"]),
    )


def _verified_controlled(
    root: Path, workflow: str, config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], P1RoleManifest, dict[str, Any]]:
    controlled, evidence = _bound_result(root, workflow, "controlled_experiment")
    manifest = _role_manifest(root, config)
    lineage = _read_json(controlled / "lineage.json")
    children = lineage.get("child_artifacts")
    if not isinstance(children, dict) or set(children) != set(parent.EXPECTED_CHILDREN):
        raise RuntimeError("controlled run must bind all six child artifacts")

    def child_digest(name: str) -> str:
        binding = children[name]
        if (
            not isinstance(binding, dict)
            or binding.get("path") != f"{name}/artifact_hashes.json"
            or not isinstance(binding.get("sha256"), str)
        ):
            raise RuntimeError(f"invalid controlled child binding: {name}")
        return binding["sha256"]

    bundle = verify_five_arm_bundle(
        controlled / "five_arm_fit", child_digest("five_arm_fit"), manifest
    )
    for name in (
        "five_arm_internal_evaluation",
        "counterfactual_models",
        "counterfactual_evaluation",
    ):
        verify_counterfactual_artifacts(controlled / name, child_digest(name))
    verify_module_support_artifacts(
        controlled / "module_support", child_digest("module_support")
    )
    verify_modal_observability_artifacts(
        controlled / "modal_observability", child_digest("modal_observability")
    )
    return controlled, evidence, manifest, bundle


def _metric_payload(arm: Mapping[str, Any]) -> dict[str, Any]:
    metrics = arm.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("each arm must contain one internal-audit metric payload")
    return metrics


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_ids: set[str],
    source: str | None,
    horizon: int,
    output: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("horizon_seconds") == horizon
        and (source is None or row.get("source_dataset") == source)
    ]
    selected_ids = {str(row.get("event_id")) for row in selected}
    if selected_ids != expected_ids or len(selected) != len(expected_ids):
        raise ValueError("registered event population is missing or duplicated")
    if any(row.get("status") != "completed" for row in selected):
        raise ValueError("a registered trajectory failed integration")

    sse = 0.0
    sst = 0.0
    coi = 0.0
    rmse_rows: list[tuple[str, float]] = []
    for row in selected:
        cell = row.get("outputs", {}).get(output)
        if not isinstance(cell, dict):
            raise TypeError("trajectory output cell is missing")
        count = cell.get("count_per_node")
        expected_count = horizon * 100 + 1
        if isinstance(count, bool) or count != expected_count:
            raise ValueError("trajectory output uses an unregistered time horizon")
        errors = np.asarray(cell.get("sse_by_node"), dtype=float)
        sums = np.asarray(cell.get("target_sum_by_node"), dtype=float)
        squares = np.asarray(cell.get("target_sum_squares_by_node"), dtype=float)
        if (
            errors.shape != (10,)
            or sums.shape != (10,)
            or squares.shape != (10,)
            or not np.isfinite(np.concatenate((errors, sums, squares))).all()
            or np.any(errors < 0.0)
        ):
            raise ValueError("invalid retained trajectory sufficient statistics")
        local_sst = squares - sums * sums / count
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(squares))
        if np.any(local_sst < -tolerance):
            raise ValueError("negative centered target sum of squares")
        sse += float(np.sum(errors))
        sst += float(np.sum(np.maximum(local_sst, 0.0)))
        coi = max(coi, _finite(cell.get("coi_drift"), "coi_drift", nonnegative=True))
        rmse_rows.append(
            (
                str(row["event_id"]),
                _finite(cell.get("rmse"), "rmse", nonnegative=True),
            )
        )
    if sst <= 0.0:
        raise ValueError("pooled R2 is undefined because target variance is zero")
    tail_count = max(1, math.ceil(0.1 * len(rmse_rows)))
    tail = sorted(rmse_rows, key=lambda item: (-item[1], item[0]))[:tail_count]
    return {
        "event_count": len(selected),
        "failed_event_count": 0,
        "pooled_r2": 1.0 - sse / sst,
        "pooled_sse": sse,
        "pooled_sst_per_event_node_centered": sst,
        "mean_rmse": float(np.mean([value for _, value in rmse_rows])),
        "worst_10pct_mean_rmse": float(np.mean([value for _, value in tail])),
        "worst_10pct_event_ids": [event_id for event_id, _ in tail],
        "maximum_coi_drift": coi,
    }


def _five_arm_diagnostics(
    raw: Mapping[str, Any], manifest: P1RoleManifest, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    arms = raw.get("arms")
    if raw.get("role") != "internal_audit" or not isinstance(arms, list) or len(arms) != 5:
        raise ValueError("expected exactly five internal-audit arms")
    keyed: dict[int, dict[str, Any]] = {}
    expected_ids = set(manifest.ids("internal_audit"))
    metadata = {
        record.event_id: (record.source_dataset, record.fault_line)
        for record in manifest.authorize("internal_audit", manifest.ids("internal_audit"))
    }
    sources = sorted({source for source, _ in metadata.values()})
    expected_bindings = {int(row["arm"]): row for row in bundle["arms"]}
    diagnostics: dict[str, Any] = {"arms": {}, "comparisons": {}}
    for entry in arms:
        if not isinstance(entry, dict) or isinstance(entry.get("arm"), bool):
            raise TypeError("invalid arm record")
        arm = int(entry["arm"])
        if arm in keyed or arm not in {1, 2, 3, 4, 5}:
            raise ValueError("five-arm identities are missing or duplicated")
        if entry.get("binding") != expected_bindings[arm]:
            raise ValueError("evaluation arm is not bound to the frozen fit bundle")
        metric = _metric_payload(entry)
        if (
            metric.get("role") != "internal_audit"
            or metric.get("role_manifest_sha256") != manifest.source_sha256
            or metric.get("event_ids") != list(manifest.ids("internal_audit"))
            or metric.get("failed_event_ids") != []
            or metric.get("bootstrap_executed") is not False
            or metric.get("scientific_acceptance") is not False
        ):
            raise ValueError("five-arm metric boundary or population changed")
        rows = metric.get("rows")
        if not isinstance(rows, list) or len(rows) != len(expected_ids) * len(_HORIZONS):
            raise ValueError("five-arm rows are incomplete")
        for row in rows:
            event_id = str(row.get("event_id"))
            if event_id not in metadata or (
                row.get("source_dataset"),
                row.get("fault_line"),
            ) != metadata[event_id]:
                raise ValueError("five-arm row metadata differs from the role manifest")
        aggregates: dict[str, Any] = {}
        for source in (*sources, None):
            scoped_ids = {
                event_id
                for event_id, (owner, _) in metadata.items()
                if source is None or owner == source
            }
            source_key = "ALL" if source is None else source
            aggregates[source_key] = {
                str(horizon): {
                    output: _aggregate_rows(
                        rows,
                        expected_ids=scoped_ids,
                        source=source,
                        horizon=horizon,
                        output=output,
                    )
                    for output in _OUTPUTS
                }
                for horizon in _HORIZONS
            }
        diagnostics["arms"][str(arm)] = {
            "binding": entry["binding"],
            "balanced_loss": _finite(
                metric.get("balanced_loss"), "balanced_loss", nonnegative=True
            ),
            "aggregates": aggregates,
        }
        keyed[arm] = metric

    if set(keyed) != {1, 2, 3, 4, 5}:
        raise ValueError("five-arm inventory is incomplete")
    for name, reference, candidate in (
        ("structure_1_to_2", 1, 2),
        ("balanced_loss_2_to_3", 2, 3),
        ("observed_pm_3_to_4", 3, 4),
        ("causal_closure_3_to_5", 3, 5),
    ):
        diagnostics["comparisons"][name] = summarize_paired_rows(
            paired_score_rows(keyed[reference], keyed[candidate]),
            manifest,
            "internal_audit",
        )
    diagnostics["arm2_arm3_metrics_identical"] = keyed[2] == keyed[3]
    diagnostics["independent_fault_lines"] = {
        source: len({line for owner, line in metadata.values() if owner == source})
        for source in sources
    }
    return diagnostics


def _fit_diagnostics(controlled: Path) -> dict[str, Any]:
    lineage = _read_json(controlled / "five_arm_fit" / "fit_lineage.json")
    current = lineage.get("current", {}).get("result", {}).get("details", {})
    baseline = lineage.get("b2", {}).get("result", {}).get("details", {})
    calibrations = lineage.get("calibrations")
    if not isinstance(calibrations, list) or len(calibrations) != 3:
        raise ValueError("three registered closed-loop calibrations are required")
    saved: list[dict[str, Any]] = []
    for row in calibrations:
        if not isinstance(row, dict):
            raise TypeError("calibration lineage row must be an object")
        saved.append(
            {
                "loss": str(row.get("loss")),
                "status": str(row.get("status")),
                "solver_success": row.get("solver_success"),
                "solver_nfev": int(row.get("solver_nfev")),
                "baseline_loss": _finite(
                    row.get("baseline_loss"), "calibration baseline loss", nonnegative=True
                ),
                "candidate_loss": _finite(
                    row.get("candidate_loss"), "calibration candidate loss", nonnegative=True
                ),
                "initial_core_fingerprint": str(row.get("initial_core_fingerprint")),
                "final_core_fingerprint": str(row.get("final_core_fingerprint")),
            }
        )
    b2_weak = _finite(baseline.get("weak_residual_rmse"), "B2 weak residual", nonnegative=True)
    current_weak = _finite(
        current.get("weak_residual_rmse"), "current weak residual", nonnegative=True
    )
    return {
        "b2": {
            key: baseline.get(key)
            for key in (
                "solver_success",
                "identifiability_pass",
                "n_scalar_equations",
                "n_parameters",
                "effective_rank",
                "condition_number",
                "weak_residual_rmse",
            )
        },
        "current": {
            key: current.get(key)
            for key in (
                "solver_success",
                "identifiability_pass",
                "n_scalar_equations",
                "n_parameters",
                "effective_rank",
                "condition_number",
                "weak_residual_rmse",
                "block_parameter_norms",
                "instance_parameter_norms",
            )
        },
        "weak_residual_relative_improvement": (b2_weak - current_weak) / b2_weak,
        "calibrations": saved,
        "all_calibrations_rejected_after_solver_failure": all(
            row["solver_success"] is False
            and row["status"] == "calibration_rejected"
            and row["initial_core_fingerprint"] == row["final_core_fingerprint"]
            for row in saved
        ),
    }


def _module_diagnostics(controlled: Path) -> dict[str, Any]:
    summary = _read_json(controlled / "module_support" / "summary.json")
    support = _read_json(controlled / "module_support" / "module_support.json")
    identifiability = _read_json(controlled / "module_support" / "identifiability.json")
    effects = _read_json(controlled / "module_support" / "effect_statistics.json")
    counter = _read_json(controlled / "counterfactual_evaluation" / "summary.json")
    modules = support.get("modules")
    if not isinstance(modules, list) or len(modules) != 7:
        raise ValueError("exactly seven module interventions are required")
    if counter.get("variant_count") != 8 or counter.get("failed_counterfactual_rows") != 0:
        raise ValueError("counterfactual intervention inventory is incomplete")
    identifiers: set[str] = set()
    view_counts: Counter[str] = Counter()
    compact_modules: list[dict[str, Any]] = []
    for module in modules:
        identifier = module.get("instance_id")
        status = module.get("module_contribution_status")
        views = module.get("support_views")
        if (
            not isinstance(identifier, str)
            or identifier in identifiers
            or status not in _MODULE_STATUSES
            or not isinstance(views, list)
            or len(views) != 6
        ):
            raise ValueError("module contribution record is malformed")
        identifiers.add(identifier)
        local = Counter(str(view.get("status")) for view in views)
        if any(name not in _VIEW_STATUSES for name in local):
            raise ValueError("unknown module support-view status")
        view_counts.update(local)
        compact_modules.append(
            {
                "instance_id": identifier,
                "module_contribution_status": status,
                "support_view_status_counts": dict(sorted(local.items())),
            }
        )
    ident_rows = identifiability.get("module_results")
    if not isinstance(ident_rows, list) or {
        row.get("instance_id") for row in ident_rows if isinstance(row, dict)
    } != identifiers:
        raise ValueError("module identifiability rows do not bind all interventions")
    ident_compact = [
        {
            key: row.get(key)
            for key in (
                "instance_id",
                "family",
                "weak_parameter_norm",
                "all_module_columns_active",
                "conditional_rank_gate_passed",
                "profile_gate_passed",
                "parameter_identifiability_passed",
                "sigma_ratio",
                "condition_number",
                "profile_loss_increase",
            )
        }
        for row in ident_rows
    ]
    joint = support.get("joint_b2")
    if not isinstance(joint, dict) or joint.get("status") not in {"PROVEN", "REFUTED", "UNRESOLVED"}:
        raise ValueError("joint B2 audit is missing")
    if summary.get("module_statuses") != {
        row["instance_id"]: row["module_contribution_status"] for row in compact_modules
    }:
        raise ValueError("module summary disagrees with detailed support records")
    return {
        "modules": compact_modules,
        "module_status_counts": dict(
            sorted(Counter(row["module_contribution_status"] for row in compact_modules).items())
        ),
        "support_view_status_counts": dict(sorted(view_counts.items())),
        "identifiability": ident_compact,
        "joint_b2": joint,
        "effect_statistics_manifest_bound": isinstance(effects, dict) and len(effects) == 8,
        "counterfactual_summary": counter,
    }


def _modal_diagnostics(controlled: Path) -> dict[str, Any]:
    summary = _read_json(controlled / "modal_observability" / "modal_audit_summary.json")
    if (
        summary.get("frequency_source") != "shared_equation_poles"
        or summary.get("trajectory_dmd_role") != "local_observability_only"
        or summary.get("source_labels_entered_model_or_pole_extraction") is not False
        or summary.get("observed_or_future_pm_read") is not False
        or summary.get("scientific_verdict") is not None
    ):
        raise ValueError("shared-pole/local-observability semantics changed")
    pole_statuses = summary.get("pole_statuses")
    view_counts = summary.get("view_status_counts")
    if (
        not isinstance(pole_statuses, dict)
        or len(pole_statuses) != summary.get("equation_reportable_pole_count")
        or not isinstance(view_counts, dict)
        or sum(view_counts.values()) != 2 * len(pole_statuses)
    ):
        raise ValueError("modal identity or view counts are incomplete")
    return {
        key: summary[key]
        for key in (
            "equation_reportable_pole_count",
            "frequency_source",
            "trajectory_dmd_role",
            "pole_statuses",
            "view_status_counts",
            "event_count",
            "event_pole_cells",
            "event_rank_match_cells",
        )
    }


def _access_diagnostics(
    controlled: Path, manifest: P1RoleManifest, config: Mapping[str, Any]
) -> dict[str, Any]:
    access = _read_json(controlled / "data_access_audit.json")
    controlled_summary = _read_json(controlled / "controlled_experiment_summary.json")
    expected_counts = {
        role: len(manifest.ids(role)) for role in ("fit", "selection", "internal_audit")
    }
    checks = {
        "role_counts_match": access.get("role_counts") == expected_counts,
        "role_hashes_match": access.get("role_id_hashes") == manifest.role_id_hashes,
        "test_locked": access.get("test_ids_read") == [],
        "other_folds_locked": access.get("other_folds_read") == [],
        "labels_not_read": access.get("labels_read") is False,
        "pe_voltage_not_read": access.get("pe_or_voltage_read") is False,
        "future_pm_not_main_input": access.get("future_pm_used_as_main_model_input") is False,
        "source_metadata_not_routing": access.get("source_or_fault_metadata_used_for_routing")
        is False,
        "one_model": config.get("model_count") == 1,
        "source_routing_disabled": config.get("source_routing") is False,
        "level6_not_claimed": controlled_summary.get("level_6_claimed") is False,
        "controlled_model_count_is_one": controlled_summary.get("five_arm_count") == 5
        and config.get("model_count") == 1,
        "controlled_not_promoted": controlled_summary.get("promotion_decision")
        == "KEEP_BASELINE"
        and controlled_summary.get("scientific_verdict") is None,
    }
    if not all(checks.values()):
        raise RuntimeError("controlled data-access or single-model boundary failed")
    return {"checks": checks, "role_counts": expected_counts}


def build_diagnosis(
    controlled: Path,
    manifest: P1RoleManifest,
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _read_json(controlled / "five_arm_internal_evaluation" / "raw_metrics.json")
    five_arm = _five_arm_diagnostics(raw, manifest, bundle)
    fit = _fit_diagnostics(controlled)
    modules = _module_diagnostics(controlled)
    modal = _modal_diagnostics(controlled)
    access = _access_diagnostics(controlled, manifest, config)

    arm2 = five_arm["arms"]["2"]
    arm3 = five_arm["arms"]["3"]
    arm5 = five_arm["arms"]["5"]
    arm2_equals_arm3 = (
        arm2["binding"]["checkpoint_hash"] == arm3["binding"]["checkpoint_hash"]
        and arm2["balanced_loss"] == arm3["balanced_loss"]
        and five_arm["comparisons"]["balanced_loss_2_to_3"]["views"]
        [view_key(None, 10, "omega")]["absolute_difference"]
        == 0.0
        and five_arm["arm2_arm3_metrics_identical"] is True
    )
    all_view = lambda name: five_arm["comparisons"][name]["views"][
        view_key(None, 10, "omega")
    ]
    highorder_view = lambda name: five_arm["comparisons"][name]["views"][
        view_key("ieee39_highorder_dataset", 10, "omega")
    ]
    near_lines = five_arm["independent_fault_lines"].get(
        "ieee39_near_critical_unstable_v1_dense"
    )
    causal_overall = all_view("causal_closure_3_to_5")
    causal_highorder = highorder_view("causal_closure_3_to_5")
    arm5_r2 = arm5["aggregates"]["ALL"]["10"]["omega"]["pooled_r2"]
    causal_overall_degradation = causal_overall.get("relative_degradation")
    causal_overall_improves = (
        isinstance(causal_overall_degradation, (int, float))
        and not isinstance(causal_overall_degradation, bool)
        and math.isfinite(float(causal_overall_degradation))
        and float(causal_overall_degradation) < 0.0
    )
    integration_failures = sum(
        cell[output]["failed_event_count"]
        for arm in five_arm["arms"].values()
        for cell in arm["aggregates"]["ALL"].values()
        for output in _OUTPUTS
    )
    maximum_coi = max(
        cell[output]["maximum_coi_drift"]
        for arm in five_arm["arms"].values()
        for cell in arm["aggregates"]["ALL"].values()
        for output in _OUTPUTS
    )
    module_counts = modules["module_status_counts"]
    pole_counts = Counter(modal["pole_statuses"].values())
    joint_status = modules["joint_b2"]["status"]
    supported_count = module_counts.get("INDEPENDENTLY_SUPPORTED", 0)
    if joint_status == "REFUTED" and supported_count == 0:
        q1_status = "ANSWERED_WHOLE_MODEL_DIFFERS_INDIVIDUAL_SUPPORT_ABSENT"
        q1_answer = (
            "The jointly deleted model is statistically worse in at least one registered cell, "
            "so the whole candidate does not collapse to B2.  However, no named module has "
            "independent predictive support; the observed module-status counts are "
            f"{dict(sorted(module_counts.items()))}. The candidate therefore cannot be promoted "
            "as a validated module-union model."
        )
    else:
        q1_status = "ANSWERED_WITH_BOUND_MODULE_STATUS"
        q1_answer = (
            f"The joint B2 audit is {joint_status}; the seven registered module interventions "
            f"have status counts {dict(sorted(module_counts.items()))}. These outcomes are kept "
            "separate because a whole-model difference is not individual-module support."
        )
    all_calibrations_rejected = fit["all_calibrations_rejected_after_solver_failure"]
    weak_improvement = fit["weak_residual_relative_improvement"]
    if (
        integration_failures == 0
        and maximum_coi <= 1e-10
        and all_calibrations_rejected
        and causal_overall_improves
        and causal_overall.get("status") == "UNDETERMINED"
        and arm5_r2 <= 0.0
    ):
        q2_status = "ANSWERED_WITH_UNRESOLVED_DECOMPOSITION"
        q2_answer = (
            "Numerical integration and COI preservation are not the cause. A causal mechanical-"
            "power closure improves the registered point error, but 10 s pooled speed R2 remains "
            f"{arm5_r2:.6g}. The weak local objective improves only {weak_improvement:.3%}, all "
            "three closed-loop calibrations are rejected after solver failure, and the overall "
            "fault-line interval is unavailable. The evidence identifies an unresolved mixture "
            "of forcing/dynamic closure and objective-calibration mismatch, not one proven cause."
        )
        root_cause_status = "UNRESOLVED_MIXED_CLOSURE_AND_OBJECTIVE_CALIBRATION"
    else:
        q2_status = "ANSWERED_WITH_BOUND_CAUSE_EVIDENCE"
        q2_answer = (
            f"The registered evidence has {integration_failures} integration failures, maximum "
            f"COI drift {maximum_coi:.6g}, causal-closure 10 s pooled speed R2 {arm5_r2:.6g}, "
            f"and weak-residual relative improvement {weak_improvement:.3%}. Calibration and "
            "paired-interval statuses are reported without converting an unavailable interval "
            "into a causal conclusion."
        )
        root_cause_status = "BOUND_EVIDENCE_REPORTED"
    if pole_counts == {"NOT_OBSERVED": modal["equation_reportable_pole_count"]}:
        q3_status = "ANSWERED_METHOD_CORRECT_CLAIM_UNSUPPORTED"
        q3_answer = (
            "The implementation follows the v8 semantics: poles come from the one frozen shared "
            "equation and trajectory DMD only measures excitation-local observability. On IEEE39, "
            f"none of the {modal['equation_reportable_pole_count']} reportable poles is observed; "
            f"view statuses are {modal['view_status_counts']}. No identified-frequency claim is available."
        )
    else:
        q3_status = "ANSWERED_WITH_BOUND_MODAL_SUPPORT"
        q3_answer = (
            "The implementation follows the v8 shared-equation-pole and local-DMD semantics. "
            f"Pole statuses are {dict(sorted(pole_counts.items()))}; view statuses are "
            f"{modal['view_status_counts']}. Only eligible supported poles may become claims."
        )

    questions = {
        "Q1_modules_change_prediction_or_degenerate_to_B2": {
            "answer_status": q1_status,
            "answer": q1_answer,
            "machine_checks": {
                "seven_interventions_present": len(modules["modules"]) == 7,
                "eight_counterfactual_variants_present": modules["counterfactual_summary"].get(
                    "variant_count"
                )
                == 8,
                "no_counterfactual_failures": modules["counterfactual_summary"].get(
                    "failed_counterfactual_rows"
                )
                == 0,
                "joint_b2_has_registered_status": joint_status
                in {"PROVEN", "REFUTED", "UNRESOLVED"},
                "all_module_statuses_registered": sum(module_counts.values()) == 7
                and set(module_counts).issubset(_MODULE_STATUSES),
                "effect_statistics_bound": modules["effect_statistics_manifest_bound"] is True,
            },
            "evidence": modules,
            "claim_boundary": (
                "Joint-B2 refutation is not individual-module support. Nonzero coefficients are "
                "not independent contributions; zero coefficients and failed profile gates remain "
                "negative evidence."
            ),
        },
        "Q2_why_long_horizon_speed_error_is_large": {
            "answer_status": q2_status,
            "answer": q2_answer,
            "machine_checks": {
                "all_five_arms_present": set(five_arm["arms"]) == {"1", "2", "3", "4", "5"},
                "integration_failure_count_valid": isinstance(integration_failures, int)
                and integration_failures >= 0,
                "maximum_coi_finite": math.isfinite(maximum_coi),
                "weak_fit_statistics_finite": math.isfinite(weak_improvement),
                "three_calibrations_present": len(fit["calibrations"]) == 3,
                "arm2_arm3_reversion_audited": isinstance(arm2_equals_arm3, bool),
                "causal_highorder_interval_status_valid": causal_highorder.get("status")
                in {"COMPLETED", "UNDETERMINED", "UNAVAILABLE_FAILED"},
                "causal_overall_interval_status_valid": causal_overall.get("status")
                in {"COMPLETED", "UNDETERMINED", "UNAVAILABLE_FAILED"},
                "causal_overall_point_effect_finite": isinstance(
                    causal_overall_degradation, (int, float)
                )
                and not isinstance(causal_overall_degradation, bool)
                and math.isfinite(float(causal_overall_degradation)),
                "independent_line_count_recorded": isinstance(near_lines, int)
                and near_lines >= 0,
                "causal_10s_speed_r2_finite": math.isfinite(arm5_r2),
            },
            "evidence": {
                "five_arm": five_arm,
                "fit": fit,
                "integration_failure_count": integration_failures,
                "maximum_coi_drift": maximum_coi,
                "arm2_equals_arm3": arm2_equals_arm3,
                "arm4_interpretation": (
                    "frozen-core input sensitivity only; core3 was fitted under Pm_clear ZOH, "
                    "so observed Pm(t) is not a refitted conditional upper bound"
                ),
                "root_cause_status": root_cause_status,
            },
            "claim_boundary": (
                "Point improvements do not replace a missing overall confidence interval. The "
                "observed-Pm arm is noncausal and parameter-confounded, and the causal closure has "
                "not achieved positive 10 s speed R2."
            ),
        },
        "Q3_modal_audit_shared_poles_and_local_observability": {
            "answer_status": q3_status,
            "answer": q3_answer,
            "machine_checks": {
                "frequency_source_is_shared_equation": modal["frequency_source"]
                == "shared_equation_poles",
                "dmd_is_local_observability_only": modal["trajectory_dmd_role"]
                == "local_observability_only",
                "pole_status_count_complete": sum(pole_counts.values())
                == modal["equation_reportable_pole_count"],
                "pole_statuses_registered": set(pole_counts).issubset(
                    {"OBSERVED", "NOT_OBSERVED"}
                ),
                "view_statuses_registered": set(modal["view_status_counts"]).issubset(
                    {"SUPPORTED", "UNSUPPORTED", "UNDETERMINED"}
                ),
            },
            "evidence": modal,
            "claim_boundary": (
                "Internal equation poles are not identified IEEE39 oscillation frequencies unless "
                "an eligible excitation view supplies the preregistered support."
            ),
        },
    }
    for question in questions.values():
        checks = question["machine_checks"]
        if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
            raise RuntimeError("a P1 diagnostic question is incomplete or contradicted")
    return {
        "schema": "kron_reduce.p1_a04_residual_diagnosis.v1",
        "diagnostic_target": "three_master_P1_questions",
        "diagnosis_complete": True,
        "questions": questions,
        "access": access,
        "scientific_verdict": None,
        "promotion_decision": "KEEP_BASELINE",
        "level_6_claimed": False,
    }


def _seal_stage(
    result_dir: Path, lineage: Mapping[str, Any], stage_evidence: Mapping[str, Any]
) -> None:
    _write_json(result_dir / "lineage.json", dict(lineage))
    _write_json(result_dir / "stage_evidence.json", dict(stage_evidence))
    digest = _seal(result_dir)
    verify_artifact_directory(result_dir, expected_manifest_sha256=digest, strict=True)


def run_residual_diagnosis(
    *,
    root: Path,
    workflow: str,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled, controlled_evidence, manifest, bundle = _verified_controlled(
        root, workflow, config
    )
    diagnosis = build_diagnosis(controlled, manifest, bundle, config)
    diagnosis.update(
        {
            "run_id": run_id,
            "controlled_run": controlled.name,
            "controlled_manifest": controlled_evidence["artifact_manifest"],
        }
    )
    _write_json(result_dir / "residual_diagnosis.json", diagnosis)
    _seal_stage(
        result_dir,
        {
            "schema": "kron_reduce.p1_a04_residual_lineage.v1",
            "run_id": run_id,
            "stage": "residual_diagnosis",
            "controlled_run": controlled.name,
            "controlled_manifest": controlled_evidence["artifact_manifest"],
        },
        {
            "process_status": "PASS",
            "promotion_decision": "KEEP_BASELINE",
            "level_6_claimed": False,
            "diagnosis_complete": True,
        },
    )


def _adversarial_checks(
    diagnosis: Mapping[str, Any],
    retry_grid: Mapping[str, Any],
    retry_access: Mapping[str, Any],
    controlled_access: Mapping[str, Any],
) -> dict[str, bool]:
    questions = diagnosis.get("questions")
    return {
        "exact_three_questions": isinstance(questions, dict)
        and set(questions)
        == {
            "Q1_modules_change_prediction_or_degenerate_to_B2",
            "Q2_why_long_horizon_speed_error_is_large",
            "Q3_modal_audit_shared_poles_and_local_observability",
        },
        "all_machine_checks_pass": isinstance(questions, dict)
        and all(
            isinstance(question.get("machine_checks"), dict)
            and bool(question["machine_checks"])
            and all(value is True for value in question["machine_checks"].values())
            for question in questions.values()
        ),
        "all_claim_boundaries_nonempty": isinstance(questions, dict)
        and all(
            isinstance(question.get("claim_boundary"), str)
            and bool(question["claim_boundary"].strip())
            for question in questions.values()
        ),
        "registered_time_grid_complete": retry_grid.get("event_count") == 178
        and retry_grid.get("shifted_event_count") == 69
        and retry_grid.get("no_extrapolation_verified") is True,
        "retry_read_time_only": retry_access.get("only_postclear_time_read") is True
        and retry_access.get("raw_state_or_pm_read") is False,
        "test_and_other_folds_locked": retry_access.get("test_ids_read") == []
        and retry_access.get("other_folds_read") == []
        and controlled_access.get("test_ids_read") == []
        and controlled_access.get("other_folds_read") == [],
        "forbidden_channels_not_used": controlled_access.get("labels_read") is False
        and controlled_access.get("pe_or_voltage_read") is False,
        "single_model_without_source_routing": controlled_access.get(
            "source_or_fault_metadata_used_for_routing"
        )
        is False,
        "future_pm_not_main_input": controlled_access.get(
            "future_pm_used_as_main_model_input"
        )
        is False,
        "observed_pm_confound_disclosed": isinstance(questions, dict)
        and "not a refitted conditional upper bound"
        in questions["Q2_why_long_horizon_speed_error_is_large"]["evidence"][
            "arm4_interpretation"
        ],
        "tail_metrics_not_omitted": isinstance(questions, dict)
        and all(
            "worst_10pct_mean_rmse" in output
            and len(output.get("worst_10pct_event_ids", []))
            == max(1, math.ceil(0.1 * output.get("event_count", 0)))
            for arm in questions["Q2_why_long_horizon_speed_error_is_large"]["evidence"][
                "five_arm"
            ]["arms"].values()
            for horizon in arm["aggregates"]["ALL"].values()
            for output in horizon.values()
        ),
        "no_premature_verdict_or_level6": diagnosis.get("scientific_verdict") is None
        and diagnosis.get("level_6_claimed") is False,
        "model_not_promoted": diagnosis.get("promotion_decision") == "KEEP_BASELINE",
    }


def _review_passes(review: Mapping[str, Any]) -> bool:
    checks = review.get("checks")
    return (
        review.get("status") == "PASS"
        and isinstance(checks, dict)
        and set(checks) == _ADVERSARIAL_CHECK_NAMES
        and all(value is True for value in checks.values())
    )


def run_adversarial_review(
    *,
    root: Path,
    workflow: str,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    retry, retry_evidence = _bound_result(root, workflow, "retry_registered")
    controlled, controlled_evidence, manifest, bundle = _verified_controlled(
        root, workflow, config
    )
    residual, residual_evidence = _bound_result(root, workflow, "residual_diagnosis")
    stored = _read_json(residual / "residual_diagnosis.json")
    recomputed = build_diagnosis(controlled, manifest, bundle, config)
    stored_core = {
        key: stored[key]
        for key in (
            "schema",
            "diagnostic_target",
            "diagnosis_complete",
            "questions",
            "access",
            "scientific_verdict",
            "promotion_decision",
            "level_6_claimed",
        )
    }
    if stored_core != recomputed:
        raise RuntimeError("sealed residual diagnosis is not reproducible from its parent")
    checks = _adversarial_checks(
        stored,
        _read_json(retry / "time_grid_audit.json"),
        _read_json(retry / "data_access_audit.json"),
        _read_json(controlled / "data_access_audit.json"),
    )
    if set(checks) != _ADVERSARIAL_CHECK_NAMES or not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"P1-A04 adversarial review failed closed: {failed}")
    review = {
        "schema": "kron_reduce.p1_a04_adversarial_review.v1",
        "run_id": run_id,
        "status": "PASS",
        "checks": checks,
        "attacks": {
            "B2_degeneracy": "joint deletion refutation is kept separate from module support",
            "mean_masks_tail": "all five arms retain worst-10% means and five event identities",
            "baseline_unfairness": "all arms share the registered role, horizon, integrator, and contract",
            "loss_effect_overclaim": "arm2/arm3 equality is attributed to rejected calibration",
            "observed_pm_overclaim": "arm4 is frozen-core sensitivity, not a valid upper bound",
            "modal_overinterpretation": "unobserved shared poles are not frequency claims",
            "label_or_future_leakage": "registered access audit is clean",
        },
        "scientific_verdict": None,
        "promotion_decision": "KEEP_BASELINE",
        "level_6_claimed": False,
    }
    _write_json(result_dir / "adversarial_review.json", review)
    _seal_stage(
        result_dir,
        {
            "schema": "kron_reduce.p1_a04_adversarial_lineage.v1",
            "run_id": run_id,
            "stage": "adversarial_review",
            "parents": {
                "retry_registered": retry_evidence["artifact_manifest"],
                "controlled_experiment": controlled_evidence["artifact_manifest"],
                "residual_diagnosis": residual_evidence["artifact_manifest"],
            },
        },
        {
            "process_status": "PASS",
            "promotion_decision": "KEEP_BASELINE",
            "level_6_claimed": False,
            "adversarial_review_status": "PASS",
        },
    )


def _rollback_target(root: Path, config: Mapping[str, Any]) -> str:
    report = _read_json(root / str(config["previous_failure_report"]))
    target = report.get("rollback_target")
    if not isinstance(target, str) or not target:
        raise RuntimeError("registered prior failure lacks a rollback target")
    return target


def run_acceptance(
    *,
    root: Path,
    workflow: str,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled, controlled_evidence, _, _ = _verified_controlled(root, workflow, config)
    residual, residual_evidence = _bound_result(root, workflow, "residual_diagnosis")
    adversarial, adversarial_evidence = _bound_result(root, workflow, "adversarial_review")
    diagnosis = _read_json(residual / "residual_diagnosis.json")
    review = _read_json(adversarial / "adversarial_review.json")
    verdict = (
        "ACCEPT"
        if diagnosis.get("diagnosis_complete") is True and _review_passes(review)
        else "RETRY"
    )
    questions = diagnosis["questions"]
    paper_claims = [question["answer"] for question in questions.values()]
    acceptance = {
        "schema": "kron_reduce.p1_a04_acceptance.v1",
        "run_id": run_id,
        "process_status": "PASS",
        "scientific_verdict": verdict,
        "promotion_decision": "KEEP_BASELINE",
        "rollback_target": _rollback_target(root, config),
        "claim_boundary": (
            "Accepted P1 diagnosis on F1 fit/selection/internal-audit evidence only. The candidate "
            "model is not promoted; there is no test/F2-F5 performance, identified-frequency, "
            "stability-certificate, or Level 6 claim."
        ),
        "paper_usable_claims": paper_claims if verdict == "ACCEPT" else [],
        "question_answers": {
            name: {
                "answer_status": question["answer_status"],
                "answer": question["answer"],
                "claim_boundary": question["claim_boundary"],
            }
            for name, question in questions.items()
        },
        "parents": {
            "controlled_experiment": controlled_evidence["artifact_manifest"],
            "residual_diagnosis": residual_evidence["artifact_manifest"],
            "adversarial_review": adversarial_evidence["artifact_manifest"],
        },
        "controlled_run": controlled.name,
        "level_6_claimed": False,
    }
    _write_json(result_dir / "acceptance.json", acceptance)
    _seal_stage(
        result_dir,
        {
            "schema": "kron_reduce.p1_a04_acceptance_lineage.v1",
            "run_id": run_id,
            "stage": "acceptance",
            "parents": acceptance["parents"],
        },
        {
            key: acceptance[key]
            for key in (
                "process_status",
                "scientific_verdict",
                "promotion_decision",
                "rollback_target",
                "claim_boundary",
                "paper_usable_claims",
            )
        },
    )


def run_rollback_report(
    *, root: Path, workflow: str, result_dir: Path, run_id: str
) -> None:
    acceptance_dir, acceptance_evidence = _bound_result(root, workflow, "acceptance")
    acceptance = _read_json(acceptance_dir / "acceptance.json")
    if acceptance.get("scientific_verdict") not in {"REJECT", "RETRY"}:
        raise RuntimeError("rollback_report requires a rejected or retry acceptance")
    residual_dir, residual_evidence = _bound_result(root, workflow, "residual_diagnosis")
    rollback = {
        "schema": "kron_reduce.p1_a04_rollback_report.v1",
        "run_id": run_id,
        "process_status": "PASS",
        "scientific_verdict": acceptance["scientific_verdict"],
        "promotion_decision": "KEEP_BASELINE",
        "rollback_target": acceptance["rollback_target"],
        "negative_evidence_run": residual_dir.name,
        "negative_evidence_manifest": residual_evidence["artifact_manifest"],
        "acceptance_run": acceptance_dir.name,
        "acceptance_manifest": acceptance_evidence["artifact_manifest"],
        "next_hypothesis": (
            "Open a new registered attempt; do not change A04 thresholds or overwrite its evidence."
        ),
    }
    _write_json(result_dir / "rollback_report.json", rollback)
    _seal_stage(
        result_dir,
        {
            "schema": "kron_reduce.p1_a04_rollback_lineage.v1",
            "run_id": run_id,
            "stage": "rollback_report",
            "acceptance": acceptance_evidence["artifact_manifest"],
            "negative_evidence": residual_evidence["artifact_manifest"],
        },
        {
            key: rollback[key]
            for key in (
                "process_status",
                "scientific_verdict",
                "promotion_decision",
                "rollback_target",
                "negative_evidence_run",
                "next_hypothesis",
            )
        },
    )


def _report_text(acceptance: Mapping[str, Any]) -> str:
    answers = acceptance["question_answers"]
    lines = [
        "# IEEE39 模块化能量等值 P1-A04 诊断报告",
        "",
        "## 结论",
        "",
        f"- 科学结论：`{acceptance['scientific_verdict']}`",
        "- 模型晋级：`KEEP_BASELINE`",
        f"- 边界：{acceptance['claim_boundary']}",
        "",
        "## 三个问题",
        "",
    ]
    for name, answer in answers.items():
        lines.extend(
            [
                f"### {name}",
                "",
                str(answer["answer"]),
                "",
                f"边界：{answer['claim_boundary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一步",
            "",
            "进入 P2 公平基线与评价合同冻结；当前候选不作为已接受部署模型。",
            "",
        ]
    )
    return "\n".join(lines)


def run_report(
    *, root: Path, workflow: str, result_dir: Path, run_id: str
) -> None:
    state = load_state(root, workflow)
    terminal = state.get("current_stage")
    if terminal not in {"merge_gate", "rollback_report"}:
        raise RuntimeError("report requires a recorded merge or rollback branch")
    acceptance_dir, acceptance_evidence = _bound_result(root, workflow, "acceptance")
    acceptance = _read_json(acceptance_dir / "acceptance.json")
    relative = f"docs/reports/modular_energy/{run_id}_p1_a04_zh.md"
    path = root / relative
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_report_text(acceptance), encoding="utf-8", newline="\n")
    evidence: dict[str, Any] = {
        "branch_terminal": terminal,
        "acceptance_run": acceptance_dir.name,
        "artifact_manifest": acceptance_evidence["artifact_manifest"],
        "report_path": relative,
        "report_artifact": {"path": relative, "sha256": sha256_file(path)},
    }
    if terminal == "rollback_report":
        _, rollback = _bound_result(root, workflow, "rollback_report")
        evidence.update(
            {
                "rollback_target": rollback["rollback_target"],
                "negative_evidence_run": rollback["negative_evidence_run"],
            }
        )
    else:
        evidence["merge_commit"] = _report_merge_commit(root, workflow)
    _write_json(
        result_dir / "report_lineage.json",
        {"schema": "kron_reduce.p1_a04_report_lineage.v1", "run_id": run_id, **evidence},
    )
    _seal_stage(
        result_dir,
        {
            "schema": "kron_reduce.p1_a04_report_result_lineage.v1",
            "run_id": run_id,
            "stage": "report",
            "acceptance": acceptance_evidence["artifact_manifest"],
        },
        evidence,
    )
