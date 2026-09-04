from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .evidence import sha256_file, verify_artifact_directory
from .git_state import assert_clean, git_head
from .loop import is_approved, next_stage, record_stage
from .manifest import load_manifest


def run_experiment(root: Path, experiment_name: str, stage: str) -> Path:
    # Permit direct execution from a source checkout without requiring an
    # editable installation. The path is derived from the repository root,
    # so no machine- or task-specific absolute path enters the project.
    src_dir = str((root / "src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    manifest = load_manifest(root)
    experiment = manifest["experiments"].get(experiment_name)
    if experiment is None:
        raise KeyError(f"unregistered experiment: {experiment_name}")
    workflow_name = experiment.get("workflow", "paper_v1")
    workflow_path = root / "configs" / "workflows" / f"{workflow_name}.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    required_gate = experiment.get("required_gate")
    if required_gate and not is_approved(root, required_gate, workflow_name):
        raise RuntimeError(f"experiment is locked by gate: {required_gate}")
    if experiment.get("status") == "gated" or stage == "full":
        assert_clean(root)
    expected_stage = next_stage(root, workflow_name)
    if stage != expected_stage:
        raise RuntimeError(
            f"workflow permits only the next registered stage: expected "
            f"{expected_stage!r}, got {stage!r}"
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{experiment_name}-{stage}-{timestamp}-{git_head(root)[:8]}"
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
        record_stage(root, stage, evidence, workflow_name)
    except Exception:
        (result_dir / "RUN_FAILED.json").write_text(json.dumps({"run_id": run_id, "status": "failed"}, indent=2), encoding="utf-8")
        raise
    return result_dir
