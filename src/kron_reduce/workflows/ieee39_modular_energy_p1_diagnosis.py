"""Early, leakage-locked stages of the IEEE39 modular-energy P1 diagnosis."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

WORKFLOW = "ieee39_modular_energy_p1_a01_v1"
EXPECTED_BRANCH = "codex/loop-p1-a01"
PLAN_EXPERIMENT = "paper_ieee39_modular_energy_p1_a01_plan_registered_f1_seed324"

CONTROLLED_ARMS = [
    "b2_zoh_original_loss",
    "current_zoh_original_loss",
    "current_zoh_balanced_loss",
    "current_observed_pm_balanced_loss_conditional_upper_bound",
    "current_causal_pm_closure_balanced_loss",
]

EXPECTED_CONFIG: dict[str, Any] = {
    "task_type": "ieee39_modular_energy_p1_current_implementation_diagnosis",
    "protocol_revision": WORKFLOW,
    "branch": EXPECTED_BRANCH,
    "dataset": "ieee39_postclear_canonical_v1",
    "split_name": "combined_line_5fold_v1",
    "fold": "F1",
    "seed": 324,
    "baseline_run_id": (
        "paper_combined_ten_machine_identification_f1_seed324-"
        "identification_run-20260825T015128Z-d4d50de4"
    ),
    "model_count": 1,
    "source_routing": False,
    "test_access": "locked",
    "other_folds_access": "locked",
    "locked_folds": ["F2", "F3", "F4", "F5"],
    "primary_pm_policy": "zoh_pm_clear",
    "observed_pm_policy": "conditional_only",
    "observed_pm_main_model_input": False,
    "bootstrap_draws": 2000,
    "bootstrap_seed": 324,
    "minimum_independent_lines_for_modal_verdict": 3,
    "fewer_than_three_independent_lines_verdict": "UNDETERMINED",
    "module_thresholds": {
        "tau_y_floor": 1e-3,
        "tau_y_multiplier": 10.0,
        "tau_phys_floor": 1e-4,
        "tau_phys_multiplier": 100.0,
        "deletion_noninferiority_degradation": 0.02,
        "identifiability_sigma_ratio_min": 1e-6,
        "identifiability_condition_max": 1e6,
        "identifiability_profile_tolerance": 0.02,
    },
    "trajectory_thresholds": {
        "speed_improvement_min": 0.05,
        "non_target_degradation_max": 0.02,
        "causal_closure_observed_gap_recovery_min": 0.50,
    },
    "modal_thresholds": {
        "finite_window_energy_min": 0.01,
        "frequency_relative_error_max": 0.05,
        "damping_absolute_error_max": 0.05,
        "mac_min": 0.80,
        "bootstrap_support_min": 0.80,
    },
    "physical_numerical_thresholds": {
        "equilibrium_residual_max": 1e-7,
        "observability_min": 1e-3,
        "coi_drift_max": 1e-10,
        "supply_identity_error_max": 1e-8,
        "rk4_order_ratio_lower": 8.0,
        "rk4_order_ratio_upper": 32.0,
    },
    "controlled_arms": {"names": CONTROLLED_ARMS},
    "claim_boundaries": {
        "one_shared_model": True,
        "source_routing": False,
        "test_performance": False,
        "f2_f5_selection": False,
        "observed_pm_conditional_only": True,
        "level_6_certificate": False,
    },
}

CRITICAL_SOURCE_FILES = (
    "src/kron_reduce/workflows/combined_ten_machine_identification.py",
    "src/kron_reduce/models/prunable_energy_model.py",
    "configs/experiments/paper_combined_ten_machine_identification_f1_seed324.toml",
    "configs/workflows/combined_ten_machine_identification_v1.json",
    "tests/test_combined_ten_machine_identification.py",
    "src/kron_reduce/workflows/ieee39_modular_energy_p1_diagnosis.py",
    "configs/experiments/paper_ieee39_modular_energy_p1_a01_f1_seed324.toml",
    "configs/workflows/ieee39_modular_energy_p1_a01_v1.json",
    "docs/protocols/ieee39_modular_energy_p1_a01_v1.md",
    "tests/test_ieee39_modular_energy_p1_diagnosis.py",
)


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_hashes(result_dir: Path) -> dict[str, str]:
    hashes = {
        path.relative_to(result_dir).as_posix(): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", hashes)
    return hashes


def _artifact_path(result_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise RuntimeError(f"unsafe artifact path in hash manifest: {relative!r}")
    candidate = result_dir.joinpath(*pure.parts)
    if not candidate.resolve().is_relative_to(result_dir.resolve()):
        raise RuntimeError(f"artifact path escapes result directory: {relative!r}")
    return candidate


def _verify_hashes(result_dir: Path) -> dict[str, Any]:
    manifest_path = result_dir / "artifact_hashes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing artifact hash manifest: {manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(recorded, Mapping) or not recorded:
        raise RuntimeError(f"invalid or empty artifact hash manifest: {manifest_path}")
    files = []
    changed = []
    for relative, expected in sorted(recorded.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError(f"invalid artifact hash entry in {manifest_path}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"invalid SHA256 for {relative!r} in {manifest_path}")
        path = _artifact_path(result_dir, relative)
        actual = _sha256(path) if path.is_file() else None
        status = "PASS" if actual == expected else "FAIL"
        files.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": status,
            }
        )
        if status != "PASS":
            changed.append(relative)
    if changed:
        raise RuntimeError(f"frozen artifact hash verification failed: {changed}")
    return {
        "status": "PASS",
        "artifact_hashes_sha256": _sha256(manifest_path),
        "files": files,
    }


def _strict_differences(actual: Any, expected: Any, prefix: str = "config") -> list[str]:
    if type(actual) is not type(expected):
        return [
            (
                f"{prefix}: expected {type(expected).__name__} {expected!r}, "
                f"got {type(actual).__name__} {actual!r}"
            )
        ]
    if isinstance(expected, dict):
        output = []
        actual_keys = set(actual)
        expected_keys = set(expected)
        for key in sorted(expected_keys - actual_keys):
            output.append(f"{prefix}.{key}: missing")
        for key in sorted(actual_keys - expected_keys):
            output.append(f"{prefix}.{key}: unexpected")
        for key in sorted(actual_keys & expected_keys):
            output.extend(_strict_differences(actual[key], expected[key], f"{prefix}.{key}"))
        return output
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return [f"{prefix}: expected length {len(expected)}, got {len(actual)}"]
        output = []
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            output.extend(
                _strict_differences(actual_item, expected_item, f"{prefix}[{index}]")
            )
        return output
    return [] if actual == expected else [f"{prefix}: expected {expected!r}, got {actual!r}"]


def _validate_config(config: Mapping[str, Any], current_branch: str) -> None:
    differences = _strict_differences(dict(config), EXPECTED_CONFIG)
    if differences:
        raise RuntimeError("P1 frozen configuration violation: " + "; ".join(differences))
    if current_branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"P1 branch lock violation: expected {EXPECTED_BRANCH!r}, got {current_branch!r}"
        )


def _git_run(root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=text,
    )


def _git_required(root: Path, *args: str) -> str:
    completed = _git_run(root, *args)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _arm_registry() -> dict[str, Any]:
    policies = (
        ("b2_zoh_original_loss", "pm_clear_zoh", "original", "causal_zoh"),
        ("current_zoh_original_loss", "pm_clear_zoh", "original", "causal_zoh"),
        ("current_zoh_balanced_loss", "pm_clear_zoh", "balanced", "causal_zoh"),
        (
            "current_observed_pm_balanced_loss_conditional_upper_bound",
            "observed_pm",
            "balanced",
            "noncausal_conditional_upper_bound",
        ),
        (
            "current_causal_pm_closure_balanced_loss",
            "causal_pm_closure",
            "balanced",
            "causal_learned_closure",
        ),
    )
    arms = []
    for index, (name, pm_policy, loss_policy, causal_status) in enumerate(policies, start=1):
        conditional_only = index == 4
        arms.append(
            {
                "index": index,
                "name": name,
                "pm_policy": pm_policy,
                "loss_policy": loss_policy,
                "causal_status": causal_status,
                "causal": not conditional_only,
                "conditional_only": conditional_only,
                "eligible_for_promotion": not conditional_only,
            }
        )
    return {"names": CONTROLLED_ARMS, "arms": arms}


def _registration(config: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "stage": "plan_registered",
        "registered_before_new_diagnostic_metrics": True,
        "p1_questions": [
            {
                "id": "Q1",
                "question": "Do the registered modules materially change predictions?",
            },
            {
                "id": "Q2",
                "question": "Why is long-horizon speed error large?",
            },
            {
                "id": "Q3",
                "question": "Is the modal audit consistent with local observability rules?",
            },
        ],
        "controlled_arms": _arm_registry(),
        "module_audit": {
            "required_together": True,
            "parameter_zeroing_counterfactual": {
                "refit": False,
                "purpose": "direct dynamical effect of a frozen module",
                "granularity": ["event", "horizon"],
            },
            "whole_block_deletion_and_refit": {
                "refit_all_remaining_parameters": True,
                "purpose": "replaceability of the deleted module",
                "granularity": ["event", "horizon"],
            },
        },
        "gates": {
            "module": dict(config["module_thresholds"]),
            "trajectory": dict(config["trajectory_thresholds"]),
            "physical_numerical": dict(config["physical_numerical_thresholds"]),
            "modal": {
                **dict(config["modal_thresholds"]),
                "minimum_independent_lines": config[
                    "minimum_independent_lines_for_modal_verdict"
                ],
                "insufficient_lines_verdict": config[
                    "fewer_than_three_independent_lines_verdict"
                ],
                "frequency_source": "shared_equation_poles",
                "excitation_view_role": "local_observability_audit_only",
            },
            "dynamic_threshold_formulas": {
                "tau_y": "max(1e-3, 10 * trajectory_numerical_error)",
                "tau_phys": "max(1e-4, 100 * supply_identity_error)",
            },
        },
        "allowed_scientific_verdicts": ["ACCEPT", "RETRY", "REJECT"],
        "decision_routing": {
            "ACCEPT": "manual_merge_gate_for_diagnosis_only",
            "RETRY": "rollback_report",
            "REJECT": "rollback_report",
        },
        "promotion_decision": "KEEP_BASELINE",
        "rollback_target": config["baseline_run_id"],
        "claim_boundary": dict(config["claim_boundaries"]),
        "test_ids_read": [],
        "other_folds_read": [],
    }


def _run_plan(
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "registration.json", _registration(config, run_id))
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": "plan_registered",
            "algorithm_unit": "one_corpus_one_model",
            "model_count": 1,
            "source_routing": False,
            "labels_used": False,
            "future_observed_pm_primary": False,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)


def _find_registered_plan(
    root: Path, config_path: Path, config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    results_dir = root / "results"
    candidates = sorted(
        path
        for path in results_dir.glob(f"{PLAN_EXPERIMENT}-plan_registered-*")
        if path.is_dir() and not (path / "RUN_FAILED.json").exists()
    )
    matches = []
    verifications: dict[Path, dict[str, Any]] = {}
    for path in candidates:
        verification = _verify_hashes(path)
        verifications[path] = verification
        registration_path = path / "registration.json"
        frozen_config_path = path / "frozen_config.toml"
        if not registration_path.is_file() or not frozen_config_path.is_file():
            raise RuntimeError(f"completed plan result lacks required artifacts: {path.name}")
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        same_registration = (
            registration.get("workflow") == WORKFLOW
            and registration.get("stage") == "plan_registered"
            and registration.get("rollback_target") == config["baseline_run_id"]
            and registration.get("promotion_decision") == "KEEP_BASELINE"
        )
        same_config = _sha256(frozen_config_path) == _sha256(config_path)
        if same_registration and same_config:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            "candidate acquisition requires exactly one matching completed plan_registered "
            f"result; found {len(matches)}"
        )
    plan = matches[0]
    return plan, verifications[plan]


def _baseline_suffix_commit(run_id: str) -> str | None:
    match = re.search(r"-([0-9a-f]{8,40})$", run_id)
    return match.group(1) if match else None


def _source_snapshot(
    root: Path,
    baseline_run_id: str,
    critical_files: Sequence[str] = CRITICAL_SOURCE_FILES,
) -> dict[str, Any]:
    head = _git_required(root, "rev-parse", "HEAD")
    branch = _git_required(root, "branch", "--show-current")
    status_text = _git_required(root, "status", "--porcelain=v1", "--untracked-files=all")
    status_rows = [
        {"status": line[:2], "path": line[3:], "raw": line}
        for line in status_text.splitlines()
        if line
    ]
    findings = []
    files = []
    for relative in critical_files:
        current_path = root / relative
        current_sha256 = _sha256(current_path) if current_path.is_file() else None
        tracked = _git_run(root, "cat-file", "-e", f"HEAD:{relative}").returncode == 0
        head_sha256 = None
        if tracked:
            head_content = _git_run(root, "show", f"HEAD:{relative}", text=False)
            if head_content.returncode == 0:
                head_sha256 = hashlib.sha256(head_content.stdout).hexdigest()
            else:
                tracked = False
        proven = bool(tracked and current_sha256 is not None and current_sha256 == head_sha256)
        file_findings = []
        if current_sha256 is None:
            file_findings.append("current_file_missing")
        if not tracked:
            file_findings.append("not_tracked_by_current_head")
        elif not proven:
            file_findings.append("current_content_differs_from_current_head")
        for finding in file_findings:
            findings.append({"path": relative, "finding": finding})
        files.append(
            {
                "path": relative,
                "current_sha256": current_sha256,
                "tracked_by_current_head": tracked,
                "head_sha256": head_sha256,
                "current_hash_proven_by_head": proven,
                "findings": file_findings,
            }
        )
    suffix = _baseline_suffix_commit(baseline_run_id)
    suffix_resolves = bool(
        suffix and _git_run(root, "cat-file", "-e", f"{suffix}^{{commit}}").returncode == 0
    )
    return {
        "current_head": head,
        "branch": branch,
        "working_tree_clean": not status_rows,
        "git_status_paths": status_rows,
        "baseline_run_id": baseline_run_id,
        "baseline_run_suffix_commit": suffix,
        "baseline_run_suffix_commit_resolves": suffix_resolves,
        "critical_files": files,
        "reproducible_from_clean_commit": all(
            row["current_hash_proven_by_head"] for row in files
        ),
        "findings": findings,
    }


def _instance_family_map(frozen_model: Mapping[str, Any]) -> dict[str, str]:
    instances = frozen_model.get("active_block_instances")
    if not isinstance(instances, list):
        diagnostics = frozen_model.get("diagnostics", {})
        instances = diagnostics.get("active_block_instances", [])
    output = {}
    for row in instances:
        if isinstance(row, dict) and "instance_id" in row and "family" in row:
            output[str(row["instance_id"])] = str(row["family"])
    return output


def _candidate_inventory(
    validation_summary: Mapping[str, Any],
    frozen_model: Mapping[str, Any],
    structure_selection: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = frozen_model.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise TypeError("frozen_model diagnostics must be a mapping")
    block_norms = dict(diagnostics.get("block_parameter_norms", {}))
    instance_norms = dict(diagnostics.get("instance_parameter_norms", {}))
    families = _instance_family_map(frozen_model)
    zero_instances = []
    by_family: dict[str, int] = {}
    for instance_id, norm in sorted(instance_norms.items()):
        if type(norm) in {int, float} and norm == 0:
            family = families.get(str(instance_id), "UNKNOWN")
            zero_instances.append(
                {"instance_id": str(instance_id), "family": family, "parameter_norm": norm}
            )
            by_family[family] = by_family.get(family, 0) + 1
    active_blocks = validation_summary.get(
        "active_blocks", structure_selection.get("active_blocks", diagnostics.get("active_blocks", []))
    )
    active_instances = validation_summary.get("active_block_instances")
    if active_instances is None:
        active_instances = sorted(families)
    return {
        "source": "hash_verified_frozen_baseline_artifacts_only",
        "terminal_state": validation_summary.get(
            "terminal_state", structure_selection.get("terminal_state")
        ),
        "active_blocks": list(active_blocks),
        "active_block_instances": list(active_instances),
        "block_parameter_norms": block_norms,
        "instance_parameter_norms": instance_norms,
        "model_count": validation_summary.get("model_count"),
        "source_routing": validation_summary.get("source_routing"),
        "test_ids_read": list(validation_summary.get("test_ids_read", [])),
        "test_trajectories_read": validation_summary.get("test_trajectories_read"),
        "future_observed_Pm_used": validation_summary.get("future_observed_Pm_used"),
        "zero_effect_by_parameter": {
            "criterion": "parameter_norm_strictly_equal_to_zero",
            "instances": zero_instances,
            "count": len(zero_instances),
            "count_by_family": by_family,
            "interpretation": "inventory_not_scientific_verdict",
        },
    }


def _data_access_audit(
    run_id: str,
    baseline_run_id: str,
    baseline_lineage: Mapping[str, Any],
    role_audit: Mapping[str, Any],
) -> dict[str, Any]:
    inherited_labels = baseline_lineage.get("labels_first_read_stage") is not None or any(
        str(key).startswith("postfreeze_safety") for key in role_audit.get("counts", {})
    )
    return {
        "run_id": run_id,
        "stage": "candidate_acquired",
        "accessed_sources": [
            {
                "type": "hash_verified_saved_baseline_artifacts",
                "run_id": baseline_run_id,
            },
            {
                "type": "canonical_role_manifest_copy",
                "origin": f"{baseline_run_id}/dataset_manifest.csv",
                "output": "data_role_manifest.csv",
            },
        ],
        "raw_trajectory_data_read": False,
        "test_ids_read": [],
        "test_trajectories_read": 0,
        "other_folds_read": [],
        "raw_labels_read": False,
        "labels_used_for_fitting_normalization_calibration_or_selection": False,
        "future_summaries_read": False,
        "future_observed_pm_read": False,
        "postfreeze_safety_labels": {
            "status": "inherited_artifact_only" if inherited_labels else "not_present",
            "inherited_from_baseline_saved_artifacts": inherited_labels,
            "used_for_new_fit_or_performance_conclusion": False,
        },
    }


def _run_candidate(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    plan_dir, plan_verification = _find_registered_plan(root, config_path, config)
    baseline_dir = root / "results" / str(config["baseline_run_id"])
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"configured baseline result does not exist: {baseline_dir}")
    baseline_verification = _verify_hashes(baseline_dir)

    required = (
        "dataset_manifest.csv",
        "validation_summary.json",
        "frozen_model.json",
        "structure_selection_result.json",
        "lineage.json",
        "role_audit.json",
    )
    missing = [name for name in required if not (baseline_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"configured baseline lacks required frozen artifacts: {missing}")

    validation = json.loads((baseline_dir / "validation_summary.json").read_text(encoding="utf-8"))
    frozen_model = json.loads((baseline_dir / "frozen_model.json").read_text(encoding="utf-8"))
    structure_selection = json.loads(
        (baseline_dir / "structure_selection_result.json").read_text(encoding="utf-8")
    )
    baseline_lineage = json.loads((baseline_dir / "lineage.json").read_text(encoding="utf-8"))
    role_audit = json.loads((baseline_dir / "role_audit.json").read_text(encoding="utf-8"))

    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        baseline_dir / "dataset_manifest.csv", result_dir / "data_role_manifest.csv"
    )
    _json(
        result_dir / "baseline_artifact_verification.json",
        {
            "status": "PASS",
            "baseline_run_id": baseline_dir.name,
            **baseline_verification,
        },
    )
    _json(
        result_dir / "source_snapshot.json",
        _source_snapshot(root, str(config["baseline_run_id"])),
    )
    _json(
        result_dir / "candidate_inventory.json",
        _candidate_inventory(validation, frozen_model, structure_selection),
    )
    _json(result_dir / "arm_registry.json", _arm_registry())
    _json(
        result_dir / "data_access_audit.json",
        _data_access_audit(run_id, baseline_dir.name, baseline_lineage, role_audit),
    )
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": "candidate_acquired",
            "algorithm_unit": "one_corpus_one_model",
            "model_count": 1,
            "source_routing": False,
            "labels_used": False,
            "future_observed_pm_primary": False,
            "plan_registered_run": plan_dir.name,
            "plan_artifact_hashes_sha256": plan_verification["artifact_hashes_sha256"],
            "baseline_run": baseline_dir.name,
            "baseline_artifact_hashes_sha256": baseline_verification[
                "artifact_hashes_sha256"
            ],
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if stage not in {"plan_registered", "candidate_acquired"}:
        raise ValueError(f"unsupported P1 diagnosis stage: {stage}")
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    current_branch = _git_required(root, "branch", "--show-current")
    _validate_config(config, current_branch)
    if stage == "plan_registered":
        _run_plan(config_path, config, result_dir, run_id)
    else:
        _run_candidate(root, config_path, config, result_dir, run_id)
