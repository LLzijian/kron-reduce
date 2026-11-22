"""Early, leakage-locked stages of the IEEE39 modular-energy P1 diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from project_harness.evidence import EvidenceValidationError, verify_artifact_directory
from project_harness.report_storage import locate_report

WORKFLOW_A01 = "ieee39_modular_energy_p1_a01_v1"
WORKFLOW_A02 = "ieee39_modular_energy_p1_a02_v1"
WORKFLOW_A03 = "ieee39_modular_energy_p1_a03_v1"
WORKFLOW = WORKFLOW_A01
EXPECTED_BRANCH = "codex/loop-p1-a01"
PLAN_EXPERIMENT = "paper_ieee39_modular_energy_p1_a01_plan_registered_f1_seed324"
EXPECTED_BRANCH_BY_WORKFLOW = {
    WORKFLOW_A01: EXPECTED_BRANCH,
    WORKFLOW_A02: "codex/loop-p1-a02",
    WORKFLOW_A03: "codex/loop-p1-a03",
}
PLAN_EXPERIMENT_BY_WORKFLOW = {
    WORKFLOW_A01: PLAN_EXPERIMENT,
    WORKFLOW_A02: "paper_ieee39_modular_energy_p1_a02_plan_registered_f1_seed324",
    WORKFLOW_A03: "paper_ieee39_modular_energy_p1_a03_plan_registered_f1_seed324",
}
CONFIG_FILE_BY_WORKFLOW = {
    workflow: name.replace("_plan_registered_", "_") + ".toml"
    for workflow, name in PLAN_EXPERIMENT_BY_WORKFLOW.items()
}

CONTROLLED_ARMS = [
    "b2_zoh_original_loss",
    "current_zoh_original_loss",
    "current_zoh_balanced_loss",
    "current_observed_pm_balanced_loss_conditional_upper_bound",
    "current_causal_pm_closure_balanced_loss",
]

EXPECTED_CONFIG: dict[str, Any] = {
    "task_type": "ieee39_modular_energy_p1_current_implementation_diagnosis",
    "protocol_revision": WORKFLOW_A01,
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

EXPECTED_CONFIG_A02: dict[str, Any] = {
    **EXPECTED_CONFIG,
    "protocol_revision": WORKFLOW_A02,
    "attempt": 2,
    "branch": EXPECTED_BRANCH_BY_WORKFLOW[WORKFLOW_A02],
    "previous_attempt_workflow": WORKFLOW_A01,
    "previous_retry_evidence": (
        "paper/planning/ieee39_modular_energy_equivalent/qa_logs/"
        "p1_a01_model_unit_gate_retry.json"
    ),
    "previous_retry_evidence_sha256": "53f2ebe96703ddc78245ce2edcd9c8ca9bc14e7ca3190477f8139203cdc3968d",
    "baseline_artifact_manifest_sha256": "98990e9e4480fa66a943277830d792f6869ef3dd405bbf89663f86e910d2127c",
}
EXPECTED_CONFIGS = {
    WORKFLOW_A01: EXPECTED_CONFIG,
    WORKFLOW_A02: EXPECTED_CONFIG_A02,
}
EXPECTED_CONFIG_A03: dict[str, Any] = {
    **EXPECTED_CONFIG_A02,
    "protocol_revision": WORKFLOW_A03,
    "attempt": 3,
    "branch": EXPECTED_BRANCH_BY_WORKFLOW[WORKFLOW_A03],
    "previous_attempt_workflow": WORKFLOW_A02,
    "previous_retry_evidence": (
        "paper/planning/ieee39_modular_energy_equivalent/qa_logs/"
        "p1_a02_candidate_acquisition_retry.json"
    ),
    "previous_retry_evidence_sha256": "c2951301ac15e8b93d6d29478817a731eafdad270bc5fb81424a3604dd51a1b1",
}
EXPECTED_CONFIGS[WORKFLOW_A03] = EXPECTED_CONFIG_A03

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
    "project_harness/evidence.py",
    "project_harness/loop.py",
    "project_harness/audit.py",
    "project_harness/runner.py",
    "src/kron_reduce/evaluation/module_contribution.py",
    "src/kron_reduce/evaluation/modal_observability.py",
    "src/kron_reduce/evaluation/pm_closure_diagnostics.py",
    "configs/experiments/paper_ieee39_modular_energy_p1_a02_f1_seed324.toml",
    "configs/workflows/ieee39_modular_energy_p1_a02_v1.json",
    "docs/protocols/ieee39_modular_energy_p1_a02_v1.md",
    "tests/test_p1_evaluation_primitives.py",
    "tests/test_prunable_modal_minimality.py",
    "src/kron_reduce/evaluation/stability_region.py",
    "tests/test_stability_region.py",
    "tests/test_harness.py",
    "configs/experiments/paper_ieee39_modular_energy_p1_a03_f1_seed324.toml",
    "configs/workflows/ieee39_modular_energy_p1_a03_v1.json",
    "docs/protocols/ieee39_modular_energy_p1_a03_design_zh.md",
    "tests/test_p1_a03_legacy_migration.py",
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


def _verify_hashes(
    result_dir: Path, *, expected_manifest_sha256: str | None = None
) -> dict[str, Any]:
    manifest_path = result_dir / "artifact_hashes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing artifact hash manifest: {manifest_path}")
    try:
        recorded = verify_artifact_directory(
            result_dir, expected_manifest_sha256=expected_manifest_sha256, strict=True
        )
    except EvidenceValidationError as error:
        raise RuntimeError(f"frozen artifact hash verification failed: {error}") from error
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
    workflow = config.get("protocol_revision")
    expected = EXPECTED_CONFIGS.get(str(workflow))
    if expected is None:
        raise RuntimeError(f"unsupported P1 protocol revision: {workflow!r}")
    differences = _strict_differences(dict(config), expected)
    if differences:
        raise RuntimeError("P1 frozen configuration violation: " + "; ".join(differences))
    expected_branch = EXPECTED_BRANCH_BY_WORKFLOW[str(workflow)]
    if current_branch != expected_branch:
        raise RuntimeError(
            f"P1 branch lock violation: expected {expected_branch!r}, got {current_branch!r}"
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


def _git_status_rows(root: Path) -> list[dict[str, str]]:
    """Read porcelain records without stripping a leading status-space."""
    completed = _git_run(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        text=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git status failed: {detail}")
    tokens = completed.stdout.split(b"\0")
    rows: list[dict[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        if len(token) < 4 or token[2:3] != b" ":
            raise RuntimeError("invalid git porcelain v1 -z record")
        status = token[:2].decode("ascii", errors="strict")
        path = token[3:].decode("utf-8", errors="surrogateescape")
        row = {"status": status, "path": path}
        if ("R" in status or "C" in status) and index < len(tokens):
            original = tokens[index]
            index += 1
            row["original_path"] = original.decode(
                "utf-8", errors="surrogateescape"
            )
        rows.append(row)
    return rows


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
    registration = {
        "run_id": run_id,
        "workflow": config["protocol_revision"],
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
    if config.get("attempt") is not None:
        registration["attempt"] = config["attempt"]
        registration["previous_attempt_workflow"] = config["previous_attempt_workflow"]
        registration["previous_retry_evidence"] = config["previous_retry_evidence"]
    return registration


def _run_plan(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    if config["protocol_revision"] in {WORKFLOW_A02, WORKFLOW_A03}:
        previous = _artifact_path(root, str(config["previous_retry_evidence"]))
        if not previous.is_file() or _sha256(previous) != config["previous_retry_evidence_sha256"]:
            raise RuntimeError("previous attempt RETRY evidence hash mismatch")
        previous_record = json.loads(previous.read_text(encoding="utf-8"))
        if previous_record.get("scientific_verdict") != "RETRY" or previous_record.get("state_advanced") is not False:
            raise RuntimeError("previous attempt must record RETRY without gate advancement")
        if config["protocol_revision"] == WORKFLOW_A03 and previous_record.get("workflow") != config["previous_attempt_workflow"]:
            raise RuntimeError("previous RETRY evidence belongs to another workflow")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    registration = _registration(config, run_id)
    if config["protocol_revision"] == WORKFLOW_A03:
        protocol = root / locate_report(root, "docs/protocols/ieee39_modular_energy_p1_a03_design_zh.md")
        frozen_protocol = result_dir / "frozen_protocol.md"
        shutil.copyfile(protocol, frozen_protocol)
        registration["protocol_artifact"] = {"path": frozen_protocol.name, "sha256": _sha256(frozen_protocol)}
    _json(result_dir / "registration.json", registration)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "workflow": config["protocol_revision"],
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
    workflow = str(config["protocol_revision"])
    if workflow in {WORKFLOW_A02, WORKFLOW_A03}:
        from project_harness.loop import load_state

        history = load_state(root, workflow)["history"]
        registrations = [
            event["evidence"] for event in history
            if event.get("action") == "advance" and event.get("to") == "plan_registered"
        ]
        if len(registrations) != 1:
            raise RuntimeError("P1 retry requires one history-bound plan registration")
        evidence = registrations[0]
        binding = evidence.get("artifact_manifest", {})
        run_name = evidence.get("result_dir")
        if (
            not isinstance(run_name, str) or not run_name
            or "/" in run_name or "\\" in run_name or ":" in run_name
            or run_name in {".", ".."}
        ):
            raise RuntimeError("invalid registered plan run identity")
        if (
            not isinstance(binding, dict)
            or binding.get("run_id") != run_name
            or binding.get("path") != "artifact_hashes.json"
            or not isinstance(binding.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])
        ):
            raise RuntimeError("registered plan lacks its independent manifest anchor")
        plan_dir = results_dir / run_name
        verification = _verify_hashes(plan_dir, expected_manifest_sha256=binding.get("sha256"))
        if _sha256(plan_dir / "frozen_config.toml") != _sha256(config_path):
            raise RuntimeError("current config differs from the registered plan")
        if workflow == WORKFLOW_A03:
            protocol = root / locate_report(root, "docs/protocols/ieee39_modular_energy_p1_a03_design_zh.md")
            registration = json.loads((plan_dir / "registration.json").read_text(encoding="utf-8"))
            if registration.get("protocol_artifact") != {"path": "frozen_protocol.md", "sha256": _sha256(protocol)} or _sha256(plan_dir / "frozen_protocol.md") != _sha256(protocol):
                raise RuntimeError("current protocol differs from the registered plan")
        return plan_dir, verification
    plan_experiment = PLAN_EXPERIMENT_BY_WORKFLOW[workflow]
    candidates = sorted(
        path
        for path in results_dir.glob(f"{plan_experiment}-plan_registered-*")
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
            registration.get("workflow") == workflow
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
    status_rows = _git_status_rows(root)
    findings = []
    files = []
    for relative in critical_files:
        physical_relative = locate_report(root, relative)
        current_path = root / physical_relative
        current_sha256 = _sha256(current_path) if current_path.is_file() else None
        tracked = _git_run(root, "cat-file", "-e", f"HEAD:{physical_relative}").returncode == 0
        head_sha256 = None
        if tracked:
            head_content = _git_run(root, "show", f"HEAD:{physical_relative}", text=False)
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


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key in legacy evidence: {key!r}")
        output[key] = value
    return output


def _load_legacy_object(path: Path) -> dict[str, Any]:
    """Read historical JSON, preserving nonfinite numbers internally for audit.

    This reader does not make those numbers usable scientific observations.
    Duplicate keys are refused because their interpretation is ambiguous.
    """
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _nonfinite_locations(value: Any, pointer: str = "") -> list[dict[str, str]]:
    """Return JSON pointers and string descriptions, never the nonfinite value."""
    if isinstance(value, float) and not math.isfinite(value):
        kind = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return [{
            "json_pointer": pointer, "value_kind": kind,
            "disposition": "preserved_in_legacy_payload_not_claimable",
        }]
    if isinstance(value, dict):
        return [
            row for key, child in value.items()
            for row in _nonfinite_locations(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
        ]
    if isinstance(value, list):
        return [row for index, child in enumerate(value) for row in _nonfinite_locations(child, f"{pointer}/{index}")]
    return []


def _legacy_migration_records(baseline_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a strict metadata envelope without reserializing legacy science."""
    legacy = baseline_dir / "structure_selection_result.json"
    structure = _load_legacy_object(legacy)
    declarations = {
        name: _load_legacy_object(baseline_dir / name)
        for name in ("validation_summary.json", "lineage.json")
    }
    for name, record in declarations.items():
        if type(record.get("model_count")) is not int or record["model_count"] != 1:
            raise RuntimeError(f"{name} requires explicit integer model_count=1")
        if record.get("source_routing") is not False:
            raise RuntimeError(f"{name} requires explicit source_routing=false")
        if "routing_keys" in record and record["routing_keys"] != []:
            raise RuntimeError(f"{name} contradicts empty routing_keys")
    for field, expected in (("model_count", 1), ("source_routing", False), ("routing_keys", [])):
        if field in structure and (type(structure[field]) is not type(expected) or structure[field] != expected):
            raise RuntimeError(f"legacy structure contradicts {field}")
    digest = _sha256(legacy)
    locations = _nonfinite_locations(structure)
    envelope = {
        "schema": "kron_reduce.legacy_structure_envelope.v1",
        "model_count": 1,
        "source_routing": False,
        "routing_keys": [],
        "legacy_structure_artifact": {"path": "legacy_structure_selection_result.json", "sha256": digest},
        "format_audit_path": "legacy_format_audit.json",
        "legacy_baseline_migration": {
            "baseline_run_id": baseline_dir.name,
            "original_structure_selection_sha256": digest,
            "added_fields": [field for field in ("model_count", "source_routing", "routing_keys") if field not in structure],
            "evidence_sources": ["validation_summary.json", "lineage.json"],
            "payload_strategy": "byte_for_byte_copy",
            "scientific_metrics_changed": False,
        },
        "claim_boundary": "legacy metadata only; no trajectory, modal, or stability performance claim",
    }
    audit = {
        "schema": "kron_reduce.legacy_json_format_audit.v1",
        "baseline_run_id": baseline_dir.name,
        "legacy_structure_artifact": dict(envelope["legacy_structure_artifact"]),
        "contains_nonfinite_values": bool(locations),
        "nonfinite_value_count": len(locations),
        "nonfinite_values": locations,
        "scientific_values_rewritten": False,
        "disposition": "preserved_in_legacy_payload_not_claimable",
        "scientific_acceptance": "NOT_EVALUATED",
        "raw_trajectory_data_read": False,
    }
    return envelope, audit


def _write_legacy_migration(baseline_dir: Path, result_dir: Path) -> None:
    envelope, audit = _legacy_migration_records(baseline_dir)
    paths = [result_dir / name for name in (
        "legacy_structure_selection_result.json", "candidate_structure_selection_explicit.json",
        "legacy_format_audit.json",
    )]
    if any(path.exists() for path in paths):
        raise FileExistsError("legacy migration cannot overwrite existing artifacts")
    # The copied file is an opaque historical artifact; never JSON-dump it.
    with (
        (baseline_dir / "structure_selection_result.json").open("rb") as source,
        paths[0].open("xb") as destination,
    ):
        shutil.copyfileobj(source, destination)
    if _sha256(paths[0]) != envelope["legacy_structure_artifact"]["sha256"]:
        raise RuntimeError("legacy structure changed during byte-for-byte migration")
    for path, record in ((paths[1], envelope), (paths[2], audit)):
        with path.open("x", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, indent=2, allow_nan=False)
            destination.write("\n")


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
    baseline_verification = _verify_hashes(
        baseline_dir,
        expected_manifest_sha256=config.get("baseline_artifact_manifest_sha256"),
    )

    required = (
        "dataset_manifest.csv",
        "validation_summary.json",
        "frozen_model.json",
        "frozen_model.npz",
        "b2_baseline_model.npz",
        "b2_baseline_model.json",
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
    if config["protocol_revision"] in {WORKFLOW_A02, WORKFLOW_A03}:
        model_count_values = {
            "validation_summary": validation.get("model_count"),
            "lineage": baseline_lineage.get("model_count"),
        }
        routing_values = {
            "validation_summary": validation.get("source_routing"),
            "lineage": baseline_lineage.get("source_routing"),
        }
        if any(type(value) is not int or value != 1 for value in model_count_values.values()):
            raise RuntimeError(f"legacy baseline model_count is inconsistent: {model_count_values}")
        if any(value is not False for value in routing_values.values()):
            raise RuntimeError(f"legacy baseline source_routing is inconsistent: {routing_values}")
        copied_model = result_dir / "frozen_model.npz"
        copied_b2 = result_dir / "b2_baseline_model.npz"
        shutil.copyfile(baseline_dir / copied_model.name, copied_model)
        shutil.copyfile(baseline_dir / copied_b2.name, copied_b2)
        shutil.copyfile(baseline_dir / "frozen_model.json", result_dir / "frozen_model.json")
        shutil.copyfile(baseline_dir / "b2_baseline_model.json", result_dir / "b2_baseline_model.json")
        for field, expected_value in (
            ("model_count", 1), ("source_routing", False), ("routing_keys", [])
        ):
            if field in structure_selection and (
                type(structure_selection[field]) is not type(expected_value)
                or structure_selection[field] != expected_value
            ):
                raise RuntimeError(f"legacy structure contradicts {field}")
        if config["protocol_revision"] == WORKFLOW_A03:
            _write_legacy_migration(baseline_dir, result_dir)
        else:
            explicit_structure = {
                **structure_selection,
                "model_count": 1,
                "source_routing": False,
                "routing_keys": [],
                "legacy_baseline_migration": {
                    "baseline_run_id": baseline_dir.name,
                    "original_structure_selection_sha256": _sha256(
                        baseline_dir / "structure_selection_result.json"
                    ),
                    "added_fields": [
                        field for field in ("model_count", "source_routing", "routing_keys")
                        if field not in structure_selection
                    ],
                    "evidence_sources": ["validation_summary.json", "lineage.json"],
                    "scientific_metrics_changed": False,
                },
            }
            _json(result_dir / "candidate_structure_selection_explicit.json", explicit_structure)
        _json(
            result_dir / "model_artifact_manifest.json",
            {
                "algorithm_unit": "one_corpus_one_model",
                "model_count": 1,
                "source_routing": False,
                "routing_keys": [],
                "final_models": [
                    {
                        "role": "final_model",
                        "path": copied_model.name,
                        "sha256": _sha256(copied_model),
                        "metadata_path": "frozen_model.json",
                        "metadata_sha256": _sha256(result_dir / "frozen_model.json"),
                    }
                ],
                "comparison_models": [
                    {
                        "role": "B2_reference",
                        "path": copied_b2.name,
                        "sha256": _sha256(copied_b2),
                        "metadata_path": "b2_baseline_model.json",
                        "metadata_sha256": _sha256(result_dir / "b2_baseline_model.json"),
                    }
                ],
                "legacy_baseline_not_reproducible": True,
                "level_6_claimed": False,
                "stability_artifact_role": "exploratory_stability_artifact",
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
            "workflow": config["protocol_revision"],
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
    if config["protocol_revision"] == WORKFLOW_A03:
        # A stage cannot advance merely because its files were writable.
        validate_p1_candidate_contract(
            root, result_dir, config,
            expected_manifest_sha256=_sha256(result_dir / "artifact_hashes.json"),
        )


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
        _run_plan(root, config_path, config, result_dir, run_id)
    else:
        _run_candidate(root, config_path, config, result_dir, run_id)


def validate_p1_candidate_contract(
    root: Path,
    candidate_dir: Path,
    config: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Check saved-candidate consistency, not scientific or full unit acceptance.

    This is one prerequisite of model_unit_gate. It neither evaluates raw
    trajectories nor advances the workflow. In particular it does not certify
    that the controlled-experiment implementation is complete.
    """
    _verify_hashes(candidate_dir, expected_manifest_sha256=expected_manifest_sha256)
    workflow = config.get("protocol_revision")
    if workflow not in {WORKFLOW_A02, WORKFLOW_A03}:
        raise RuntimeError("explicit candidate contract requires A02 or A03")
    if not candidate_dir.resolve().is_relative_to((root / "results").resolve()):
        raise RuntimeError("candidate directory must be within registered results")

    def read_object(directory: Path, name: str) -> dict[str, Any]:
        value = json.loads((directory / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"{name} must contain a JSON object")
        return value

    lineage = read_object(candidate_dir, "lineage.json")
    inventory = read_object(candidate_dir, "candidate_inventory.json")
    structure = read_object(candidate_dir, "candidate_structure_selection_explicit.json")
    models = read_object(candidate_dir, "model_artifact_manifest.json")
    access = read_object(candidate_dir, "data_access_audit.json")
    source = read_object(candidate_dir, "source_snapshot.json")
    baseline_id = str(config["baseline_run_id"])
    baseline_dir = root / "results" / baseline_id
    baseline_verification = _verify_hashes(
        baseline_dir, expected_manifest_sha256=config["baseline_artifact_manifest_sha256"]
    )
    baseline_validation = read_object(baseline_dir, "validation_summary.json")
    baseline_lineage = read_object(baseline_dir, "lineage.json")
    baseline_structure = _load_legacy_object(baseline_dir / "structure_selection_result.json")
    verification = read_object(candidate_dir, "baseline_artifact_verification.json")
    config_path = root / "configs" / "experiments" / CONFIG_FILE_BY_WORKFLOW[workflow]
    if _strict_differences(tomllib.loads(config_path.read_text(encoding="utf-8")), dict(config)):
        raise RuntimeError("candidate contract config differs from the saved protocol")
    plan_dir, plan_verification = _find_registered_plan(root, config_path, config)
    if (
        lineage.get("plan_registered_run") != plan_dir.name
        or lineage.get("plan_artifact_hashes_sha256") != plan_verification["artifact_hashes_sha256"]
    ):
        raise RuntimeError("candidate plan lineage mismatch")
    if (
        lineage.get("run_id") != candidate_dir.name
        or lineage.get("workflow") != config["protocol_revision"]
        or lineage.get("stage") != "candidate_acquired"
        or lineage.get("baseline_run") != baseline_id
        or verification.get("baseline_run_id") != baseline_id
        or verification.get("status") != "PASS"
        or verification.get("artifact_hashes_sha256") != baseline_verification["artifact_hashes_sha256"]
        or lineage.get("baseline_artifact_hashes_sha256") != baseline_verification["artifact_hashes_sha256"]
    ):
        raise RuntimeError("candidate/baseline lineage mismatch")
    records = {
        "inventory": inventory, "structure": structure, "model_manifest": models,
        "lineage": lineage, "baseline_validation": baseline_validation,
        "baseline_lineage": baseline_lineage,
    }
    for name, record in records.items():
        if type(record.get("model_count")) is not int or record["model_count"] != 1:
            raise RuntimeError(f"{name} requires explicit integer model_count=1")
        if record.get("source_routing") is not False:
            raise RuntimeError(f"{name} requires explicit source_routing=false")
    if structure.get("routing_keys") != [] or models.get("routing_keys") != []:
        raise RuntimeError("candidate must not have routing keys")
    migration = structure.get("legacy_baseline_migration", {})
    if (
        migration.get("original_structure_selection_sha256")
        != _sha256(baseline_dir / "structure_selection_result.json")
        or migration.get("scientific_metrics_changed") is not False
    ):
        raise RuntimeError("legacy structure migration lacks its unchanged-source anchor")
    if workflow == WORKFLOW_A03:
        expected_structure, expected_audit = _legacy_migration_records(baseline_dir)
        if _strict_differences(structure, expected_structure, "legacy_envelope"):
            raise RuntimeError("legacy metadata envelope differs from the frozen-source contract")
        if _strict_differences(read_object(candidate_dir, "legacy_format_audit.json"), expected_audit):
            raise RuntimeError("legacy format audit differs from the frozen-source scan")
        if _sha256(candidate_dir / "legacy_structure_selection_result.json") != _sha256(baseline_dir / "structure_selection_result.json"):
            raise RuntimeError("legacy payload is not a byte-for-byte source copy")
    else:
        for field, value in baseline_structure.items():
            if structure.get(field) != value:
                raise RuntimeError(f"legacy scientific field changed during migration: {field}")
    for field in ("final_models", "comparison_models"):
        entries = models.get(field)
        expected_role = "final_model" if field == "final_models" else "B2_reference"
        expected_name = "frozen_model" if field == "final_models" else "b2_baseline_model"
        if (
            not isinstance(entries, list) or len(entries) != 1
            or not isinstance(entries[0], dict) or entries[0].get("role") != expected_role
        ):
            raise RuntimeError(f"{field} must declare exactly one {expected_role}")
        entry = entries[0]
        for path_key, hash_key, suffix in (
            ("path", "sha256", ".npz"), ("metadata_path", "metadata_sha256", ".json")
        ):
            filename = expected_name + suffix
            if (
                entry.get(path_key) != filename
                or entry.get(hash_key) != _sha256(candidate_dir / filename)
                or _sha256(candidate_dir / filename) != _sha256(baseline_dir / filename)
            ):
                raise RuntimeError(f"candidate model binding mismatch: {filename}")
    if _sha256(candidate_dir / "data_role_manifest.csv") != _sha256(baseline_dir / "dataset_manifest.csv"):
        raise RuntimeError("candidate data roles differ from frozen baseline")
    for name, record in {
        "lineage": lineage, "inventory": inventory, "access": access,
        "baseline_validation": baseline_validation, "baseline_lineage": baseline_lineage,
    }.items():
        if record.get("test_ids_read") != []:
            raise RuntimeError(f"{name} has missing or nonempty test access evidence")
    for record in (lineage, access):
        if record.get("other_folds_read") != []:
            raise RuntimeError("candidate accessed locked folds")
    for record in (inventory, access, baseline_validation, baseline_lineage):
        if type(record.get("test_trajectories_read")) is not int or record["test_trajectories_read"] != 0:
            raise RuntimeError("candidate/baseline test trajectory access is not explicitly zero")
    for field in (
        "raw_trajectory_data_read", "raw_labels_read", "future_summaries_read",
        "future_observed_pm_read", "labels_used_for_fitting_normalization_calibration_or_selection",
    ):
        if access.get(field) is not False:
            raise RuntimeError(f"candidate access audit violates {field}")
    inherited = access.get("postfreeze_safety_labels", {})
    if inherited.get("status") not in {"inherited_artifact_only", "not_present"} or inherited.get("used_for_new_fit_or_performance_conclusion") is not False:
        raise RuntimeError("postfreeze safety evidence cannot be reused for selection")
    if any(record.get("future_observed_Pm_used") is not False for record in (inventory, baseline_validation)):
        raise RuntimeError("baseline/candidate future Pm status is not explicitly false")
    if baseline_validation.get("level_6_claimed") is not False or models.get("level_6_claimed") is not False:
        raise RuntimeError("P1 cannot claim Level 6")
    if _strict_differences(read_object(candidate_dir, "arm_registry.json"), _arm_registry(), "arms"):
        raise RuntimeError("candidate arm registry differs from frozen five-arm contract")
    if source.get("branch") != config["branch"]:
        raise RuntimeError("candidate source snapshot branch mismatch")
    if (
        access.get("run_id") != candidate_dir.name
        or access.get("stage") != "candidate_acquired"
        or lineage.get("algorithm_unit") != "one_corpus_one_model"
        or models.get("algorithm_unit") != "one_corpus_one_model"
        or lineage.get("labels_used") is not False
        or lineage.get("future_observed_pm_primary") is not False
    ):
        raise RuntimeError("candidate identity/input declaration mismatch")
    return {
        "status": "PASS",
        "scope": "saved_candidate_metadata_and_lineage_only",
        "candidate_run": candidate_dir.name,
        "model_count": 1,
        "source_routing": False,
        "level_6_claimed": False,
        "legacy_baseline_reproducibility_proven": False,
        "full_model_unit_gate_passed": False,
        "claim_boundary": "no trajectory, modal, or stability performance claim",
    }
