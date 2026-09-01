"""Harness-owned P1-A03 model gate and real-F1 controlled parent chain.

The leaf computations remain deliberately non-promotional.  This module only
binds them to verified inputs, a clean source commit, immutable child manifests,
and one strict root manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from project_harness.audit import audit_project
from project_harness.data_check import verify_data
from project_harness.evidence import verify_artifact_directory
from project_harness.git_state import git_head, status_lines
from project_harness.loop import load_state

from kron_reduce.data.p1_causal import P1CausalReader, P1RoleManifest
from kron_reduce.data.p1_pm_targets import PmReadPurpose, read_pm_batch
from kron_reduce.data.store import CanonicalStore
from kron_reduce.evaluation.module_contribution import (
    DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS,
)
from kron_reduce.models.p1_calibration import P1CalibrationSettings
from kron_reduce.workflows.ieee39_modular_energy_p1_diagnosis import (
    validate_p1_candidate_contract,
)
from kron_reduce.workflows.p1_counterfactual_execution import (
    _seal,
    evaluate_saved_counterfactuals,
    fit_and_save_counterfactuals,
    verify_counterfactual_artifacts,
)
from kron_reduce.workflows.p1_five_arm_execution import (
    _read_json,
    _write_json,
    evaluate_saved_five_arms,
    file_sha256,
    fit_and_save_five_arms,
    verify_five_arm_bundle,
)
from kron_reduce.workflows.p1_modal_observability_execution import (
    evaluate_saved_modal_observability,
    verify_modal_observability_artifacts,
)
from kron_reduce.workflows.p1_module_support_execution import (
    evaluate_saved_module_support,
    verify_module_support_artifacts,
)


PROTOCOL_REVISION = "ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1"
WORKFLOW = "ieee39_modular_energy_p1_a03_v1"
CONFIG_NAME = "paper_ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1.toml"
CONFIG_SHA256 = "1272bbe7a3c8ee66bf17b508840dde27509ed72023bc6d7a30c50981636b959a"
PROTOCOL_PATH = "docs/protocols/ieee39_modular_energy_p1_a03_real_f1_controlled_chain_v1_zh.md"
PROTOCOL_SHA256 = "3234d2286b96f361346577f60784f0ba6266648bb807eee279d60d152f78600f"
ORIGINAL_CONFIG_PATH = "configs/experiments/paper_ieee39_modular_energy_p1_a03_f1_seed324.toml"
WORKFLOW_PATH = "configs/workflows/ieee39_modular_energy_p1_a03_v1.json"
EXPECTED_CHILDREN = (
    "five_arm_fit",
    "five_arm_internal_evaluation",
    "counterfactual_models",
    "counterfactual_evaluation",
    "module_support",
    "modal_observability",
)
EXPECTED_FAMILIES = {
    "NODE_HIGH_ORDER": 1,
    "EDGE_MEMORY": 3,
    "RESONANT_MEMORY": 3,
}
_STRUCTURAL_INSTANCE_FIELDS = (
    "instance_id",
    "family",
    "time_constant",
    "frequency_hz",
    "damping_ratio",
    "factor",
    "port_rank",
    "energy_constraint",
    "removable",
)


def _checked_file(path: Path, expected_sha256: str) -> None:
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or len(expected_sha256) != 64
        or file_sha256(path) != expected_sha256
    ):
        raise ValueError(f"missing, unsafe, or hash-mismatched file: {path}")


def _write_text(path: Path, value: str) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"frozen configuration mismatch: {label}")


def _load_execution_config(root: Path, experiment: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    if experiment.get("config") != CONFIG_NAME:
        raise ValueError("the P1-A03 parent chain requires its frozen execution config")
    config_path = root / "configs" / "experiments" / CONFIG_NAME
    _checked_file(config_path, CONFIG_SHA256)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    frozen = {
        "task_type": "ieee39_modular_energy_p1_a03_real_f1_controlled_chain",
        "protocol_revision": PROTOCOL_REVISION,
        "workflow": WORKFLOW,
        "attempt": 3,
        "branch": "codex/loop-p1-a03",
        "protocol_document": PROTOCOL_PATH,
        "candidate_config": ORIGINAL_CONFIG_PATH,
        "candidate_config_sha256": "78d9010e4a62c2fa7e947697e2077c9d3e03431399287fb9933f71f159b23526",
        "workflow_config": WORKFLOW_PATH,
        "workflow_config_sha256": "a846f44eb80f6ec542150d992fb2273c98c7357c74c27652a55b627eadef58c5",
        "candidate_run_id": "paper_ieee39_modular_energy_p1_a03_candidate_acquired_f1_seed324-candidate_acquired-20260831T130327Z-93d29d92",
        "candidate_artifact_manifest_sha256": "6864dfaf8805782b194d9d30fe3eda7c877cdae04a4465b3b1f85d691946daf3",
        "candidate_structure_file": "frozen_model.json",
        "candidate_structure_sha256": "c61872fac88b483e788cd2eafe76e33a318d05218f0fe82d7f8eeaf2b499fedc",
        "role_manifest_file": "data_role_manifest.csv",
        "role_manifest_sha256": "1f4d677ab53d3eeee692bc8dee694c02ddfd3718202bd85e795d8d4818e2e3cb",
        "previous_retry_evidence": "paper/planning/ieee39_modular_energy_equivalent/qa_logs/p1_a02_candidate_acquisition_retry.json",
        "previous_retry_evidence_sha256": "c2951301ac15e8b93d6d29478817a731eafdad270bc5fb81424a3604dd51a1b1",
        "dataset": "ieee39_postclear_canonical_v1",
        "dataset_h5_sha256": "3f6546411f3cdbea4de7977968a779b62228fa843e62fbc4f606972eff0e9636",
        "dataset_index_sha256": "a117bafd2067b5846e15b910f194cdccdedac248f7ede2ee0bae54699111817c",
        "dataset_schema_sha256": "5e692018b7a3e9cfda68fe5846b036cf70eefddef680adc7187f9b2ebe3d3cc3",
        "split_name": "combined_line_5fold_v1",
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
    for key, value in frozen.items():
        _require_exact(config.get(key), value, key)
    _require_exact(
        config.get("structure_projection"),
        {
            "source_format_version": 2,
            "target_format_version": 3,
            "node_tau": 0.2,
            "potential_scale": 0.25,
            "max_internal_step": 0.01,
            "expected_node_high_order_instances": 1,
            "expected_edge_memory_instances": 3,
            "expected_resonant_memory_instances": 3,
            "numerical_coefficients_inherited": False,
        },
        "structure_projection",
    )
    _require_exact(config.get("calibration"), {"max_nfev": 16}, "calibration")
    _require_exact(
        config.get("execution"),
        {
            "child_directories": list(EXPECTED_CHILDREN),
            "required_roles": ["fit", "selection", "internal_audit"],
            "registered_horizons_seconds": [1, 5, 10],
        },
        "execution",
    )
    _require_exact(
        config.get("module_thresholds"),
        {
            "tau_y_floor": 1e-3,
            "tau_y_multiplier": 10.0,
            "tau_phys_floor": 1e-4,
            "tau_phys_multiplier": 100.0,
            "deletion_noninferiority_degradation": 0.02,
            "identifiability_sigma_ratio_min": 1e-6,
            "identifiability_condition_max": 1e6,
            "identifiability_profile_tolerance": 0.02,
        },
        "module_thresholds",
    )
    thresholds = config["module_thresholds"]
    _require_exact(
        DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS.as_dict(),
        {
            "prediction_floor": thresholds["tau_y_floor"],
            "prediction_numerical_multiplier": thresholds["tau_y_multiplier"],
            "physical_floor": thresholds["tau_phys_floor"],
            "physical_numerical_multiplier": thresholds["tau_phys_multiplier"],
            "deletion_degradation": thresholds["deletion_noninferiority_degradation"],
            "sigma_ratio": thresholds["identifiability_sigma_ratio_min"],
            "condition_number": thresholds["identifiability_condition_max"],
            "profile_loss_increase": thresholds["identifiability_profile_tolerance"],
        },
        "module_threshold_runtime_mapping",
    )
    expected_nested = {
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
            "minimum_independent_lines_for_verdict": 3,
        },
        "physical_numerical_thresholds": {
            "equilibrium_residual_max": 1e-7,
            "observability_min": 1e-3,
            "coi_drift_max": 1e-10,
            "supply_identity_error_max": 1e-8,
            "rk4_order_ratio_lower": 8.0,
            "rk4_order_ratio_upper": 32.0,
        },
        "claim_boundaries": {
            "one_shared_model": True,
            "source_routing": False,
            "test_performance": False,
            "f2_f5_selection": False,
            "observed_pm_conditional_only": True,
            "level_6_certificate": False,
            "scientific_verdict_before_acceptance": False,
        },
    }
    for key, value in expected_nested.items():
        _require_exact(config.get(key), value, key)
    _checked_file(root / PROTOCOL_PATH, PROTOCOL_SHA256)
    _checked_file(root / ORIGINAL_CONFIG_PATH, config["candidate_config_sha256"])
    _checked_file(root / WORKFLOW_PATH, config["workflow_config_sha256"])
    _checked_file(root / config["previous_retry_evidence"], config["previous_retry_evidence_sha256"])
    return config_path, config


def _candidate_context(root: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    candidate = root / "results" / str(config["candidate_run_id"])
    verify_artifact_directory(
        candidate,
        expected_manifest_sha256=str(config["candidate_artifact_manifest_sha256"]),
        strict=True,
    )
    original_path = root / str(config["candidate_config"])
    original = tomllib.loads(original_path.read_text(encoding="utf-8"))
    validation = validate_p1_candidate_contract(
        root,
        candidate,
        original,
        expected_manifest_sha256=str(config["candidate_artifact_manifest_sha256"]),
    )
    if validation.get("status") != "PASS":
        raise RuntimeError("candidate contract did not pass")
    return candidate, validation


def project_structure_prior(
    source_path: Path,
    expected_source_sha256: str,
    output_path: Path,
    projection_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only registered structure geometry; never inherit fitted arrays."""
    _checked_file(source_path, expected_source_sha256)
    source = _read_json(source_path)
    if source.get("format_version") != projection_contract["source_format_version"]:
        raise ValueError("candidate structure source format changed")
    for key in ("node_tau", "potential_scale", "max_internal_step"):
        if source.get(key) != projection_contract[key]:
            raise ValueError(f"candidate structure prior changed: {key}")
    instances = source.get("active_block_instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("candidate has no registered module instances")
    families = Counter(row.get("family") for row in instances if isinstance(row, dict))
    if dict(families) != EXPECTED_FAMILIES:
        raise ValueError("candidate module-union instance counts changed")
    identifiers = [row.get("instance_id") for row in instances]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate module instance IDs are invalid")
    projected_instances = []
    for row in instances:
        if not all(field in row for field in _STRUCTURAL_INSTANCE_FIELDS):
            raise ValueError("candidate module instance lacks structural fields")
        projected = {field: row[field] for field in _STRUCTURAL_INSTANCE_FIELDS}
        projected["source_support"] = {}
        factor = np.asarray(projected["factor"], dtype=float)
        if row["family"] == "NODE_HIGH_ORDER":
            valid_factor = factor.size == 0
        else:
            valid_factor = factor.ndim == 2 and factor.shape[0] > 0 and factor.shape[1] > 0
        if not valid_factor or not np.isfinite(factor).all():
            raise ValueError("candidate port geometry is malformed")
        projected_instances.append(projected)
    result = {
        "format_version": projection_contract["target_format_version"],
        "method": "P1-unified-module-union-structure-prior",
        "active_block_instances": projected_instances,
        "memory_taus": source.get("memory_taus"),
        "resonances": source.get("resonances"),
        "node_tau": source["node_tau"],
        "potential_scale": source["potential_scale"],
        "max_internal_step": source["max_internal_step"],
        "model_count": 1,
        "source_routing": False,
        "source_support_removed": True,
        "port_geometry_inherited_as_structure": True,
        "numerical_coefficients_inherited": False,
        "source_structure_path": str(source_path.resolve()),
        "source_structure_sha256": expected_source_sha256,
    }
    # JSON serialization is also the finite-value and type boundary.
    json.dumps(result, allow_nan=False)
    _write_json(output_path, result)
    return result


def _verify_dataset_inputs(root: Path, config: Mapping[str, Any]) -> Path:
    dataset = root / "data" / str(config["dataset"])
    _checked_file(dataset / "ieee39_postclear_canonical_v1.h5", str(config["dataset_h5_sha256"]))
    _checked_file(dataset / "dataset_index.csv", str(config["dataset_index_sha256"]))
    _checked_file(dataset / "schema.json", str(config["dataset_schema_sha256"]))
    return dataset


def _parse_junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    fields = ("tests", "failures", "errors", "skipped")
    values = {field: int(root.attrib.get(field, 0)) for field in fields}
    if values["tests"] < 1 or values["failures"] or values["errors"]:
        raise RuntimeError(f"full pytest gate did not pass: {values}")
    return values


def _scratch_run_directory(root: Path, run_id: str) -> Path:
    """Use a run-bound compact path that remains below Windows MAX_PATH."""
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("unsafe run ID for the model-gate scratch directory")
    token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    directory = root / "scratch" / "one_off" / f"p1-model-gate-{token}"
    if not directory.resolve().is_relative_to((root / "scratch" / "one_off").resolve()):
        raise ValueError("model-gate scratch directory escaped its registered root")
    return directory


def _run_full_pytest(root: Path, result_dir: Path, run_id: str) -> dict[str, Any]:
    scratch_parent = _scratch_run_directory(root, run_id)
    scratch_parent.mkdir(parents=True, exist_ok=False)
    junit = result_dir / "full_pytest.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--junitxml={junit}",
        f"--basetemp={scratch_parent / 'pytest_tmp'}",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _write_text(result_dir / "pytest_stdout.txt", completed.stdout)
    _write_text(result_dir / "pytest_stderr.txt", completed.stderr)
    _write_json(
        result_dir / "pytest_execution.json",
        {
            "logical_command": ["python", "-m", "pytest", "-q", "--junitxml=<result>/full_pytest.xml", "--basetemp=<scratch>/pytest_tmp"],
            "returncode": completed.returncode,
            "scratch_directory": str(scratch_parent.resolve()),
        },
    )
    if completed.returncode != 0 or not junit.is_file():
        raise RuntimeError(f"full pytest gate failed with return code {completed.returncode}")
    return _parse_junit(junit)


def _run_model_unit_gate(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    candidate, candidate_validation = _candidate_context(root, config)
    dataset = _verify_dataset_inputs(root, config)
    structure = project_structure_prior(
        candidate / str(config["candidate_structure_file"]),
        str(config["candidate_structure_sha256"]),
        result_dir / "structure_spec.json",
        config["structure_projection"],
    )
    tests = _run_full_pytest(root, result_dir, run_id)
    project_audit = audit_project(root)
    data_check = verify_data(root)
    dirty = status_lines(root)
    if project_audit.get("status") != "PASS":
        raise RuntimeError("project audit failed")
    if data_check.get("status") != "PASS":
        raise RuntimeError("data check failed")
    if dirty:
        raise RuntimeError(f"model unit gate requires a clean tree: {dirty}")
    _write_json(result_dir / "candidate_contract.json", candidate_validation)
    _write_json(result_dir / "project_audit.json", project_audit)
    _write_json(result_dir / "data_check.json", data_check)
    _write_json(result_dir / "git_check.json", {"status": "PASS", "dirty": dirty, "git_commit": git_head(root)})
    _write_json(
        result_dir / "model_gate_summary.json",
        {
            "schema": "kron_reduce.p1_a03_model_unit_gate.v1",
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": "model_unit_gate",
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_sha256": PROTOCOL_SHA256,
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": CONFIG_SHA256,
            "candidate_run": candidate.name,
            "candidate_artifact_manifest_sha256": config["candidate_artifact_manifest_sha256"],
            "dataset_path": dataset.relative_to(root).as_posix(),
            "structure_spec_sha256": file_sha256(result_dir / "structure_spec.json"),
            "projected_module_instances": len(structure["active_block_instances"]),
            "unit_tests": tests,
            "model_count": 1,
            "source_routing": False,
            "raw_ieee39_trajectory_data_read": False,
            "test_ids_read": [],
            "other_folds_read": [],
            "scientific_verdict": None,
            "promotion_decision": "KEEP_BASELINE",
            "level_6_claimed": False,
        },
    )
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "candidate_run": candidate.name,
            "unit_test_count": tests["tests"],
            "project_audit_status": "PASS",
            "data_check_status": "PASS",
            "model_contract_status": "PASS",
            "leakage_contract_status": "PASS",
            "level_6_claimed": False,
            "source_snapshot_limitations": "The gate proves software, contracts, and synthetic tests only; it makes no new IEEE39 trajectory, modal, stability, or promotion claim.",
            "scientific_verdict": None,
            "promotion_decision": "KEEP_BASELINE",
        },
    )
    digest = _seal(result_dir)
    verify_artifact_directory(result_dir, expected_manifest_sha256=digest, strict=True)


def _source_commit_evidence(root: Path) -> dict[str, Any]:
    state = load_state(root, WORKFLOW)
    if state.get("current_stage") != "source_commit":
        raise RuntimeError("controlled experiment requires the recorded source_commit stage")
    matches = [
        row.get("evidence", {})
        for row in state.get("history", [])
        if row.get("action") == "advance" and row.get("to") == "source_commit"
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("source_commit evidence is missing or ambiguous")
    evidence = matches[0]
    head = git_head(root)
    if evidence.get("git_commit") != head or head == "UNCOMMITTED":
        raise RuntimeError("controlled experiment source differs from the frozen commit")
    return evidence


def _child_binding(result: Mapping[str, Any], expected_directory: Path) -> dict[str, str]:
    digest = result.get("artifact_manifest_sha256")
    if result.get("directory") != str(expected_directory.resolve()) or not isinstance(digest, str):
        raise RuntimeError("child result lacks an exact directory/manifest binding")
    return {"path": expected_directory.name + "/artifact_hashes.json", "sha256": digest}


def _run_controlled_experiment(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    source_commit = _source_commit_evidence(root)
    candidate, candidate_validation = _candidate_context(root, config)
    dataset_dir = _verify_dataset_inputs(root, config)
    structure_path = result_dir / "structure_spec.json"
    project_structure_prior(
        candidate / str(config["candidate_structure_file"]),
        str(config["candidate_structure_sha256"]),
        structure_path,
        config["structure_projection"],
    )
    structure_sha256 = file_sha256(structure_path)

    manifest = P1RoleManifest.from_csv(
        candidate / str(config["role_manifest_file"]),
        fold=str(config["fold"]),
        expected_sha256=str(config["role_manifest_sha256"]),
    )
    store = CanonicalStore.open(dataset_dir)
    reader = P1CausalReader(store.h5_path, store.dataset_metadata.M_prior, manifest)
    fit_samples = reader.read_role("fit")
    selection_samples = reader.read_role("selection")
    internal_samples = reader.read_role("internal_audit")
    fit_pm = read_pm_batch(
        store.h5_path,
        manifest,
        fit_samples,
        role="fit",
        purpose=PmReadPurpose.FIT_CLOSURE_TARGET,
    )

    paths = {name: result_dir / name for name in EXPECTED_CHILDREN}
    base = fit_and_save_five_arms(
        paths["five_arm_fit"],
        structure_path,
        structure_sha256,
        fit_samples,
        selection_samples,
        fit_pm,
        manifest,
        settings=P1CalibrationSettings(max_nfev=int(config["calibration"]["max_nfev"])),
    )
    base_sha = str(base["artifact_manifest_sha256"])
    base_bundle = verify_five_arm_bundle(paths["five_arm_fit"], base_sha, manifest)

    observed_pm = read_pm_batch(
        store.h5_path,
        manifest,
        internal_samples,
        role="internal_audit",
        purpose=PmReadPurpose.CONDITIONAL_DIAGNOSTIC,
        core_checkpoint=paths["five_arm_fit"] / "core3.npz",
        expected_core_sha256=base_bundle["arms"][2]["checkpoint_hash"],
    )
    five_eval = evaluate_saved_five_arms(
        paths["five_arm_fit"],
        base_sha,
        manifest,
        internal_samples,
        observed_pm,
        paths["five_arm_internal_evaluation"],
    )
    five_eval_sha = file_sha256(paths["five_arm_internal_evaluation"] / "artifact_hashes.json")
    verify_counterfactual_artifacts(paths["five_arm_internal_evaluation"], five_eval_sha)

    counter = fit_and_save_counterfactuals(
        paths["five_arm_fit"],
        base_sha,
        manifest,
        fit_samples,
        selection_samples,
        paths["counterfactual_models"],
    )
    counter_sha = str(counter["artifact_manifest_sha256"])
    verify_counterfactual_artifacts(paths["counterfactual_models"], counter_sha)
    counter_eval = evaluate_saved_counterfactuals(
        paths["five_arm_fit"],
        base_sha,
        paths["counterfactual_models"],
        counter_sha,
        manifest,
        internal_samples,
        paths["counterfactual_evaluation"],
    )
    counter_eval_sha = str(counter_eval["artifact_manifest_sha256"])
    verify_counterfactual_artifacts(paths["counterfactual_evaluation"], counter_eval_sha)

    support = evaluate_saved_module_support(
        paths["five_arm_fit"],
        base_sha,
        paths["counterfactual_models"],
        counter_sha,
        paths["counterfactual_evaluation"],
        counter_eval_sha,
        manifest,
        fit_samples,
        paths["module_support"],
    )
    support_sha = str(support["artifact_manifest_sha256"])
    verify_module_support_artifacts(paths["module_support"], support_sha)

    modal = evaluate_saved_modal_observability(
        paths["five_arm_fit"],
        base_sha,
        manifest,
        internal_samples,
        paths["modal_observability"],
    )
    modal_sha = str(modal["artifact_manifest_sha256"])
    verify_modal_observability_artifacts(paths["modal_observability"], modal_sha)
    verify_five_arm_bundle(paths["five_arm_fit"], base_sha, manifest)

    child_bindings = {
        "five_arm_fit": _child_binding(base, paths["five_arm_fit"]),
        "five_arm_internal_evaluation": {
            "path": "five_arm_internal_evaluation/artifact_hashes.json",
            "sha256": five_eval_sha,
        },
        "counterfactual_models": _child_binding(counter, paths["counterfactual_models"]),
        "counterfactual_evaluation": _child_binding(counter_eval, paths["counterfactual_evaluation"]),
        "module_support": _child_binding(support, paths["module_support"]),
        "modal_observability": _child_binding(modal, paths["modal_observability"]),
    }
    role_counts = {role: len(manifest.ids(role)) for role in ("fit", "selection", "internal_audit")}
    _write_json(
        result_dir / "data_access_audit.json",
        {
            "schema": "kron_reduce.p1_a03_controlled_access.v1",
            "candidate_validation": candidate_validation,
            "role_manifest_sha256": manifest.source_sha256,
            "role_id_hashes": manifest.role_id_hashes,
            "role_counts": role_counts,
            "causal_reader_accesses": reader.accesses,
            "fit_pm_target_accesses": list(fit_pm.accesses),
            "conditional_internal_pm_accesses": list(observed_pm.accesses),
            "conditional_core_checkpoint_sha256": observed_pm.core_checkpoint_sha256,
            "test_ids_read": [],
            "other_folds_read": [],
            "labels_read": False,
            "pe_or_voltage_read": False,
            "future_pm_used_as_main_model_input": False,
            "source_or_fault_metadata_used_for_routing": False,
        },
    )
    _write_json(
        result_dir / "lineage.json",
        {
            "schema": "kron_reduce.p1_a03_controlled_lineage.v1",
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": "controlled_experiment",
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_sha256": PROTOCOL_SHA256,
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": CONFIG_SHA256,
            "source_commit": source_commit["git_commit"],
            "candidate_run": candidate.name,
            "candidate_artifact_manifest_sha256": config["candidate_artifact_manifest_sha256"],
            "dataset_h5_sha256": config["dataset_h5_sha256"],
            "structure_spec_sha256": structure_sha256,
            "child_artifacts": child_bindings,
            "model_count": 1,
            "source_routing": False,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_json(
        result_dir / "controlled_experiment_summary.json",
        {
            "schema": "kron_reduce.p1_a03_real_f1_controlled_chain.v1",
            "run_id": run_id,
            "process_status": "PASS",
            "role_counts": role_counts,
            "role_id_hashes": manifest.role_id_hashes,
            "five_arm_count": len(five_eval["arms"]),
            "counterfactual_variant_count": counter_eval["variant_count"],
            "module_count": support["module_count"],
            "module_statuses": support["module_statuses"],
            "modal_view_status_counts": modal["view_status_counts"],
            "scientific_verdict": None,
            "model_unit_gate_passed": False,
            "promotion_decision": "KEEP_BASELINE",
            "level_6_claimed": False,
            "claim_boundary": "controlled F1 diagnostic artifacts only; residual diagnosis, adversarial review, and acceptance remain pending",
        },
    )
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "candidate_run": candidate.name,
            "source_commit": source_commit["git_commit"],
            "scientific_verdict": None,
            "promotion_decision": "KEEP_BASELINE",
            "level_6_claimed": False,
            "claim_boundary": "diagnostic evidence only; no promotion or paper performance claim",
        },
    )
    digest = _seal(result_dir)
    verify_artifact_directory(result_dir, expected_manifest_sha256=digest, strict=True)


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if stage not in {"model_unit_gate", "controlled_experiment"}:
        raise ValueError(f"unsupported P1-A03 parent-chain stage: {stage}")
    config_path, config = _load_execution_config(root, experiment)
    completed = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.stdout.strip() != config["branch"]:
        raise RuntimeError("execution branch differs from the frozen protocol")
    if stage == "model_unit_gate":
        _run_model_unit_gate(root, config_path, config, result_dir, run_id)
    else:
        _run_controlled_experiment(root, config_path, config, result_dir, run_id)
