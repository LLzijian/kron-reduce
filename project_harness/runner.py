from __future__ import annotations

import importlib
import json
import sys
import tomllib
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .data_check import verify_data
from .evidence import sha256_file, verify_artifact_directory
from .git_state import assert_clean, git_head, verify_bound_file
from .loop import (
    _load_state,
    _record_stage,
    _workflow,
    approval_recorded,
    check_stage,
    workflow_lock,
)
from .manifest import load_manifest
from .policy import load_policy, validate_run_policy


def run_experiment(
    root: Path, experiment_name: str, stage: str, *, historical_reproduction: bool = False,
) -> Path:
    manifest = load_manifest(root)
    experiment = manifest["experiments"].get(experiment_name)
    if experiment is None:
        raise KeyError(f"unregistered experiment: {experiment_name}")
    if experiment.get("execution_mode") == "bounded_input_audit_cli_only":
        raise RuntimeError(
            "BOUNDED_INPUT_AUDIT_CLI_ONLY: use the registered dedicated audit CLI; "
            "generic dataset verification exceeds its registered read scope"
        )
    if experiment.get("execution_mode") == "bounded_history_cache_audit_cli_only":
        raise RuntimeError(
            "BOUNDED_HISTORY_CACHE_AUDIT_CLI_ONLY: use the registered dedicated audit CLI; "
            "generic dataset verification exceeds its registered read scope"
        )
    validate_run_policy(root, experiment_name, experiment,
                        historical_reproduction=historical_reproduction)
    _verify_development_text_bindings(
        root, experiment, historical_reproduction=historical_reproduction,
    )
    policy = load_policy(root)
    verify_source = bool(
        policy and not historical_reproduction and experiment.get("dataset") == policy["primary_real_dataset"]
    )
    workflow_name = experiment.get("workflow", "paper_v1")
    with workflow_lock(root, workflow_name):
        return _run_experiment(
            root, experiment_name, stage, experiment=experiment, verify_source=verify_source,
        )


def _verify_development_text_bindings(
    root: Path, experiment: dict, *, historical_reproduction: bool = False,
) -> None:
    """Check registered approval/design bytes before data checks or run work."""
    authorization_id = experiment.get("development_authorization_id")
    if historical_reproduction or not authorization_id:
        return

    config_name = experiment.get("config")
    if (not isinstance(config_name, str) or not config_name.endswith(".toml")
            or Path(config_name).name != config_name or "\\" in config_name):
        raise RuntimeError("DEVELOPMENT_BINDING_CONFIG_UNSAFE_OR_MISSING")
    config_path = root / "configs" / "experiments" / config_name
    try:
        config_resolved = config_path.resolve(strict=True)
        config_resolved.relative_to(root.resolve(strict=True))
        if config_path.is_symlink() or not config_resolved.is_file():
            raise RuntimeError("DEVELOPMENT_BINDING_CONFIG_UNSAFE_OR_MISSING")
        config = tomllib.loads(config_resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"DEVELOPMENT_BINDING_CONFIG_UNREADABLE: {error}") from error

    if config.get("development_authorization_id") != authorization_id:
        raise RuntimeError("DEVELOPMENT_BINDING_AUTHORIZATION_MISMATCH")
    approvals = load_policy(root).get("development_evaluations", {})
    approval = approvals.get(authorization_id) if isinstance(approvals, dict) else None
    if not isinstance(approval, dict) or approval.get("status") != "user_approved":
        raise RuntimeError("DEVELOPMENT_AUTHORIZATION_NOT_ACTIVE_USER_APPROVED")

    bindings = (
        ("development_protocol_path", "approval_protocol_sha256", "approval_protocol_path"),
        ("design_path", "design_sha256", "design_path"),
    )
    for config_path_key, config_sha_key, approval_path_key in bindings:
        relative = config.get(config_path_key)
        digest = config.get(config_sha_key)
        if (not isinstance(relative, str) or not isinstance(digest, str)
                or relative != approval.get(approval_path_key)
                or digest != approval.get(config_sha_key)):
            raise RuntimeError(f"DEVELOPMENT_BINDING_POLICY_MISMATCH: {config_path_key}")
        verify_bound_file(root, relative, digest)


def _run_experiment(
    root: Path, experiment_name: str, stage: str, *, experiment: dict | None = None,
    verify_source: bool = False,
) -> Path:
    """Dispatch one attempt with the workflow lock held by its entry point."""
    # Permit direct execution from a source checkout without requiring an
    # editable installation. The path is derived from the repository root,
    # so no machine- or task-specific absolute path enters the project.
    src_dir = str((root / "src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    if experiment is None:
        experiment = load_manifest(root)["experiments"].get(experiment_name)
        if experiment is None:
            raise KeyError(f"unregistered experiment: {experiment_name}")
    workflow_name = experiment.get("workflow", "paper_v1")
    workflow = _workflow(root, workflow_name)
    state = _load_state(root, workflow)
    registered_stage = experiment.get("stage")
    if registered_stage is not None and stage != registered_stage:
        raise RuntimeError(f"experiment {experiment_name} is registered for stage {registered_stage!r}")
    required_gate = experiment.get("required_gate")
    if required_gate and not approval_recorded(state, required_gate):
        raise RuntimeError(f"experiment is locked by gate: {required_gate}")
    check_stage(workflow, state, stage)
    if experiment.get("status") == "gated" or stage == "full":
        assert_clean(root)
    if verify_source and verify_data(root, dataset_name=experiment["dataset"])["status"] != "PASS":
        raise RuntimeError("DATA_INTEGRITY: registered kiki source hashes do not match")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{experiment_name}-{stage}-{timestamp}-{git_head(root)[:8]}-{uuid4().hex[:8]}"
    result_dir = root / "results" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    try:
        module = importlib.import_module(experiment["module"])
        module.run(root=root, experiment=experiment, stage=stage, result_dir=result_dir, run_id=run_id)
        evidence = {"status": "PASS", "result_dir": result_dir.name}
        stage_evidence_path = result_dir / "stage_evidence.json"
        if stage_evidence_path.is_file():
            supplied = json.loads(stage_evidence_path.read_text(encoding="utf-8"))
            if not isinstance(supplied, dict):
                raise RuntimeError("stage_evidence.json must be a JSON object")
            for protected, value in evidence.items():
                if protected in supplied and supplied[protected] != value:
                    raise RuntimeError(f"stage_evidence.json may not override runner {protected}")
            evidence.update(supplied)
        if workflow.get("evidence_contract_version") == 2:
            manifest_path = result_dir / "artifact_hashes.json"
            verify_artifact_directory(result_dir, strict=True)
            result_binding = {"run_id": result_dir.name, "path": "artifact_hashes.json", "sha256": sha256_file(manifest_path)}
            if evidence.get("result_artifact_manifest") is not None and evidence["result_artifact_manifest"] != result_binding:
                raise RuntimeError("stage_evidence.json may not override runner result_artifact_manifest")
            evidence["result_artifact_manifest"] = result_binding
            if stage != "report":
                if evidence.get("artifact_manifest") is not None and evidence["artifact_manifest"] != result_binding:
                    raise RuntimeError("stage_evidence.json may not override runner artifact_manifest")
                evidence["artifact_manifest"] = result_binding
        acceptance_path = result_dir / "acceptance.json"
        if stage == "acceptance" and acceptance_path.is_file():
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            for key in ("process_status", "scientific_verdict", "promotion_decision", "rollback_target", "claim_boundary", "paper_usable_claims"):
                if key in acceptance:
                    if key in evidence and evidence[key] != acceptance[key]:
                        raise RuntimeError(f"acceptance.json conflicts with stage_evidence.json for {key}")
                    evidence[key] = acceptance[key]
        _record_stage(root, workflow, state, stage, evidence)
    except Exception as error:
        (result_dir / "RUN_FAILED.json").write_text(json.dumps({
            "run_id": run_id, "status": "failed", "error_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "automatic_retry": False,
        }, indent=2), encoding="utf-8")
        raise
    return result_dir
