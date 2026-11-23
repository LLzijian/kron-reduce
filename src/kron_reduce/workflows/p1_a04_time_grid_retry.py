"""P1-A04 retry registration and harness entry points.

A04 changes only the stored-time to registered-time representation.  The
five-arm, counterfactual, support, and modal algorithms are reused from A03.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py

from kron_reduce.data.p1_causal import (
    P1_SOURCE_POINTS,
    P1RoleManifest,
    prepare_p1_time_grid,
)
from kron_reduce.workflows import p1_a03_real_f1_controlled_chain as parent
from kron_reduce.workflows.p1_counterfactual_execution import _seal
from kron_reduce.workflows.p1_five_arm_execution import _write_json, file_sha256
from project_harness.evidence import merge_evidence_commit, verify_artifact_directory
from project_harness.loop import load_state
from project_harness.report_storage import locate_report

PROTOCOL_REVISION = "ieee39_modular_energy_p1_a04_time_grid_retry_v1"
WORKFLOW = "ieee39_modular_energy_p1_a04_v1"
CONFIG_NAME = "paper_ieee39_modular_energy_p1_a04_time_grid_retry_f1_seed324.toml"
CONFIG_SHA256 = "ad88b40a5bbf17b8ee28affc41574e97e87159652963126101f13721c20cbcf9"
PROTOCOL_PATH = "docs/protocols/ieee39_modular_energy_p1_a04_time_grid_retry_v1_zh.md"
PROTOCOL_SHA256 = "c52a28e708c296e7dbad8ffdfffb6765f7e2857d7a6be396c12f8a00b5276d8a"
WORKFLOW_PATH = "configs/workflows/ieee39_modular_energy_p1_a04_v1.json"
WORKFLOW_SHA256 = "f6e823ac18b6b2e0ecc865f25164925b52ecf3c1a988a2f0840aae3f86ebd48e"
FAILURE_REPORT_SHA256 = "8c9ae63238b9929577f718641540fc6d15dd7c848a590c8f722ed419363ae4ee"
EXPECTED_EVENT_COUNT = 178
EXPECTED_SHIFTED_COUNT = 69


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"frozen A04 configuration mismatch: {label}")


def _load_config(root: Path, experiment: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    if experiment.get("config") != CONFIG_NAME:
        raise ValueError("A04 requires its frozen time-grid retry config")
    path = root / "configs" / "experiments" / CONFIG_NAME
    parent._checked_file(path, CONFIG_SHA256)
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    frozen = {
        "task_type": "ieee39_modular_energy_p1_a04_time_grid_retry",
        "protocol_revision": PROTOCOL_REVISION,
        "workflow": WORKFLOW,
        "attempt": 4,
        "branch": "codex/loop-p1-a04",
        "protocol_document": PROTOCOL_PATH,
        "protocol_document_sha256": PROTOCOL_SHA256,
        "workflow_config": WORKFLOW_PATH,
        "workflow_config_sha256": WORKFLOW_SHA256,
        "previous_failed_run": "paper_ieee39_modular_energy_p1_a03_controlled_experiment_f1_seed324-controlled_experiment-20260901T173705Z-78225d2c",
        "previous_failure_report": "paper/planning/ieee39_modular_energy_equivalent/qa_logs/p1_a03_controlled_experiment_time_grid_failure.json",
        "previous_failure_report_sha256": FAILURE_REPORT_SHA256,
        "previous_source_commit": "78225d2c4330574609580701d3fd24d895f58f3a",
        "previous_model_gate_run": "paper_ieee39_modular_energy_p1_a03_model_unit_gate_f1_seed324-model_unit_gate-20260901T172607Z-78225d2c",
        "previous_model_gate_manifest_sha256": "7cef96cb9da030cfb30d03935bfe20419ea7c70399924e5da8abdedb55ce1f76",
        "candidate_run_id": "paper_ieee39_modular_energy_p1_a03_candidate_acquired_f1_seed324-candidate_acquired-20260831T130327Z-93d29d92",
        "candidate_artifact_manifest_sha256": "6864dfaf8805782b194d9d30fe3eda7c877cdae04a4465b3b1f85d691946daf3",
        "candidate_structure_file": "frozen_model.json",
        "candidate_structure_sha256": "c61872fac88b483e788cd2eafe76e33a318d05218f0fe82d7f8eeaf2b499fedc",
        "role_manifest_file": "data_role_manifest.csv",
        "role_manifest_sha256": "1f4d677ab53d3eeee692bc8dee694c02ddfd3718202bd85e795d8d4818e2e3cb",
        "candidate_config": "configs/experiments/paper_ieee39_modular_energy_p1_a03_f1_seed324.toml",
        "candidate_config_sha256": "78d9010e4a62c2fa7e947697e2077c9d3e03431399287fb9933f71f159b23526",
        "dataset": "ieee39_postclear_canonical_v1",
        "dataset_h5_sha256": "3f6546411f3cdbea4de7977968a779b62228fa843e62fbc4f606972eff0e9636",
        "dataset_index_sha256": "a117bafd2067b5846e15b910f194cdccdedac248f7ede2ee0bae54699111817c",
        "dataset_schema_sha256": "5e692018b7a3e9cfda68fe5846b036cf70eefddef680adc7187f9b2ebe3d3cc3",
        "fold": "F1",
        "seed": 324,
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
    }
    for key, expected in frozen.items():
        _require(config.get(key), expected, key)
    _require(
        config.get("time_grid"),
        {
            "target_start_seconds": 0.0,
            "target_stop_seconds": 10.0,
            "target_step_seconds": 0.01,
            "target_points": 1001,
            "maximum_source_points_read": 1002,
            "maximum_source_gap_seconds": 0.0100000001,
            "post_first_gap_absolute_tolerance": 1e-12,
            "exact_grid_absolute_tolerance": 1e-12,
            "method": "channelwise_linear_interpolation",
            "exact_grid_action": "copy",
            "extrapolation": False,
            "pm_clear_from_raw_t0": True,
        },
        "time_grid",
    )
    parent._checked_file(root / locate_report(root, PROTOCOL_PATH, PROTOCOL_SHA256), PROTOCOL_SHA256)
    parent._checked_file(root / WORKFLOW_PATH, WORKFLOW_SHA256)
    parent._checked_file(root / str(config["previous_failure_report"]), FAILURE_REPORT_SHA256)
    parent._checked_file(root / str(config["candidate_config"]), str(config["candidate_config_sha256"]))
    previous = root / "results" / str(config["previous_failed_run"])
    if not (previous / "RUN_FAILED.json").is_file():
        raise ValueError("A03 failed controlled run is missing")
    verify_artifact_directory(
        root / "results" / str(config["previous_model_gate_run"]),
        expected_manifest_sha256=str(config["previous_model_gate_manifest_sha256"]),
        strict=True,
    )
    return path, config


def _time_grid_rows(
    h5_path: Path, manifest: P1RoleManifest
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    accesses: list[dict[str, Any]] = []
    lookup = {record.event_id: record for record in manifest.records}
    with h5py.File(h5_path, "r") as handle:
        for role in ("fit", "selection", "internal_audit"):
            for event_id in manifest.ids(role):
                selection = slice(0, P1_SOURCE_POINTS)
                accesses.append(
                    {
                        "event_id": event_id,
                        "role": role,
                        "dataset": "postclear/t",
                        "selection": str(selection),
                    }
                )
                time = handle["events"][event_id]["postclear/t"][selection]
                plan = prepare_p1_time_grid(time, event_id=event_id, role=role)
                record = lookup[event_id]
                rows.append(
                    {
                        **plan.audit,
                        "source_dataset": record.source_dataset,
                        "fault_line": record.fault_line,
                    }
                )
    return rows, accesses


def _summarize_time_grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_EVENT_COUNT or len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("A04 time-grid audit must cover exactly 178 unique events")
    shifted = sum(row["interpolation_applied"] is True for row in rows)
    if shifted != EXPECTED_SHIFTED_COUNT:
        raise ValueError("A04 shifted-event count differs from preregistration")
    if any(row["extrapolation_applied"] is not False for row in rows):
        raise ValueError("A04 time-grid audit detected extrapolation")
    return {
        "schema": "kron_reduce.p1_a04_time_grid_audit.v1",
        "event_count": len(rows),
        "exact_grid_event_count": len(rows) - shifted,
        "shifted_event_count": shifted,
        "role_counts": dict(sorted(Counter(row["role"] for row in rows).items())),
        "source_counts": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "first_positive_time_counts": dict(
            sorted(Counter(str(round(row["first_positive_time_seconds"], 6)) for row in rows).items())
        ),
        "maximum_bracketing_index": max(row["bracketing_index"] for row in rows),
        "maximum_source_gap_seconds": max(row["maximum_source_gap_seconds"] for row in rows),
        "no_extrapolation_verified": True,
        "test_ids_read": [],
        "other_folds_read": [],
        "state_or_pm_channels_read": False,
        "labels_read": False,
    }


def _run_retry_registered(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    candidate, validation = parent._candidate_context(root, config)
    dataset = parent._verify_dataset_inputs(root, config)
    manifest = P1RoleManifest.from_csv(
        candidate / str(config["role_manifest_file"]),
        fold=str(config["fold"]),
        expected_sha256=str(config["role_manifest_sha256"]),
    )
    rows, accesses = _time_grid_rows(dataset / "ieee39_postclear_canonical_v1.h5", manifest)
    summary = _summarize_time_grid(rows)
    _write_json(result_dir / "time_grid_audit.json", {**summary, "rows": rows})
    _write_json(
        result_dir / "data_access_audit.json",
        {
            "schema": "kron_reduce.p1_a04_retry_access.v1",
            "accesses": accesses,
            "candidate_contract": validation,
            "role_manifest_sha256": manifest.source_sha256,
            "only_postclear_time_read": True,
            "raw_state_or_pm_read": False,
            "test_ids_read": [],
            "other_folds_read": [],
            "labels_read": False,
        },
    )
    _write_json(
        result_dir / "lineage.json",
        {
            "schema": "kron_reduce.p1_a04_retry_lineage.v1",
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": "retry_registered",
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_sha256": PROTOCOL_SHA256,
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": CONFIG_SHA256,
            "previous_failed_run": config["previous_failed_run"],
            "previous_failure_report_sha256": config["previous_failure_report_sha256"],
            "candidate_run": candidate.name,
            "candidate_manifest_sha256": config["candidate_artifact_manifest_sha256"],
            "time_grid_audit_sha256": file_sha256(result_dir / "time_grid_audit.json"),
        },
    )
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "previous_failed_run": config["previous_failed_run"],
            "time_grid_event_count": summary["event_count"],
            "shifted_event_count": summary["shifted_event_count"],
            "no_extrapolation_verified": True,
            "promotion_decision": "KEEP_BASELINE",
        },
    )
    digest = _seal(result_dir)
    verify_artifact_directory(result_dir, expected_manifest_sha256=digest, strict=True)


def _require_retry_stage(root: Path) -> dict[str, Any]:
    state = load_state(root, WORKFLOW)
    if state.get("current_stage") != "retry_registered":
        raise RuntimeError("A04 model gate requires retry_registered evidence")
    entries = [
        event.get("evidence", {})
        for event in state.get("history", [])
        if event.get("action") == "advance" and event.get("to") == "retry_registered"
    ]
    if len(entries) != 1 or entries[0].get("no_extrapolation_verified") is not True:
        raise RuntimeError("A04 retry evidence is missing or ambiguous")
    return entries[0]


def _git_command(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )


def _require_report_branch(root: Path, config: Mapping[str, Any]) -> None:
    """Allow report only on its recorded terminal branch and commit lineage."""
    state = load_state(root, WORKFLOW)
    terminal = state.get("current_stage")
    branch_result = _git_command(root, "branch", "--show-current")
    if branch_result.returncode != 0:
        raise RuntimeError("A04 could not determine the execution branch")
    branch = branch_result.stdout.strip()
    if terminal == "rollback_report":
        if branch != config["branch"]:
            raise RuntimeError("A04 rollback report must run on the preregistered branch")
        return
    if terminal != "merge_gate":
        raise RuntimeError("A04 report requires a recorded merge or rollback branch")
    merge_entries = [
        event.get("evidence")
        for event in state.get("history", [])
        if event.get("action") == "advance" and event.get("to") == "merge_gate"
    ]
    if len(merge_entries) != 1 or not isinstance(merge_entries[0], dict):
        raise RuntimeError("A04 merge report requires exactly one merge evidence object")
    merge = merge_entries[0]
    target_branch = merge.get("target_branch")
    if not isinstance(target_branch, str) or not target_branch.strip():
        raise RuntimeError("A04 merge evidence lacks target_branch")
    if branch != target_branch:
        raise RuntimeError("A04 merge report must run on the merge evidence target_branch")
    commit = merge_evidence_commit(merge)
    if commit is None:
        raise RuntimeError("A04 merge evidence lacks merge_commit")
    if not re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", commit):
        raise RuntimeError("A04 merge evidence has an invalid merge_commit")
    resolved = _git_command(root, "cat-file", "-e", f"{commit}^{{commit}}")
    if resolved.returncode != 0:
        raise RuntimeError("A04 merge evidence commit does not resolve to a Git commit")
    ancestor = _git_command(root, "merge-base", "--is-ancestor", commit, "HEAD")
    if ancestor.returncode != 0:
        raise RuntimeError("A04 merge evidence commit must be an ancestor of HEAD")


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if stage not in {
        "retry_registered",
        "model_unit_gate",
        "controlled_experiment",
        "residual_diagnosis",
        "adversarial_review",
        "acceptance",
        "rollback_report",
        "report",
    }:
        raise ValueError(f"unsupported P1-A04 stage: {stage}")
    config_path, config = _load_config(root, experiment)
    if stage == "report":
        _require_report_branch(root, config)
    else:
        branch = _git_command(root, "branch", "--show-current")
        if branch.returncode != 0 or branch.stdout.strip() != config["branch"]:
            raise RuntimeError("A04 execution branch differs from preregistration")
    if stage == "retry_registered":
        _run_retry_registered(root, config_path, config, result_dir, run_id)
    elif stage == "model_unit_gate":
        retry = _require_retry_stage(root)
        parent._run_model_unit_gate(
            root,
            config_path,
            config,
            result_dir,
            run_id,
            workflow=WORKFLOW,
            protocol_revision=PROTOCOL_REVISION,
            protocol_sha256=PROTOCOL_SHA256,
            config_sha256=CONFIG_SHA256,
            summary_schema="kron_reduce.p1_a04_model_unit_gate.v1",
            source_snapshot_limitations=(
                "The gate verifies the preregistered no-extrapolation time harmonization and software contracts only; "
                f"retry evidence is {retry['result_dir']}. It makes no IEEE39 performance, modal, stability, or promotion claim."
            ),
        )
    elif stage == "controlled_experiment":
        parent._run_controlled_experiment(
            root,
            config_path,
            config,
            result_dir,
            run_id,
            workflow=WORKFLOW,
            protocol_revision=PROTOCOL_REVISION,
            protocol_sha256=PROTOCOL_SHA256,
            config_sha256=CONFIG_SHA256,
            summary_schema="kron_reduce.p1_a04_real_f1_controlled_chain.v1",
        )
    else:
        # Imported lazily so the post-processor can reuse this module's frozen
        # constants without a module-import cycle.  It only reads sealed parent
        # runs and writes the new stage result.
        from kron_reduce.workflows import p1_a04_postprocess as postprocess

        if stage == "residual_diagnosis":
            postprocess.run_residual_diagnosis(
                root=root,
                workflow=WORKFLOW,
                config=config,
                result_dir=result_dir,
                run_id=run_id,
            )
        elif stage == "adversarial_review":
            postprocess.run_adversarial_review(
                root=root,
                workflow=WORKFLOW,
                config=config,
                result_dir=result_dir,
                run_id=run_id,
            )
        elif stage == "acceptance":
            postprocess.run_acceptance(
                root=root,
                workflow=WORKFLOW,
                config=config,
                result_dir=result_dir,
                run_id=run_id,
            )
        elif stage == "rollback_report":
            postprocess.run_rollback_report(
                root=root,
                workflow=WORKFLOW,
                result_dir=result_dir,
                run_id=run_id,
            )
        else:
            postprocess.run_report(
                root=root,
                workflow=WORKFLOW,
                result_dir=result_dir,
                run_id=run_id,
            )
