from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kron_reduce.workflows import kiki_integral_history_audit as workflow
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.loop import load_state, record_stage


@pytest.mark.parametrize("verdict", ["ACCEPT", "REJECT"])
def test_stage_evidence_contract_records_accept_and_reject_with_strict_artifacts(tmp_path, verdict):
    project_root = Path(__file__).resolve().parents[1]
    root = tmp_path / verdict.lower()
    workflow_dir = root / "configs" / "workflows"
    workflow_dir.mkdir(parents=True)
    shutil.copyfile(
        project_root / "configs" / "workflows" / "kiki_a14_integral_history_v1.json",
        workflow_dir / "kiki_a14_integral_history_v1.json",
    )
    run_id = f"synthetic-{verdict.lower()}"
    result = root / "results" / run_id
    result.mkdir(parents=True)
    (result / "synthetic_receipt.json").write_text(
        json.dumps({"synthetic": True, "result": verdict}), encoding="utf-8"
    )
    manifest = {"synthetic_receipt.json": sha256_file(result / "synthetic_receipt.json")}
    (result / "artifact_hashes.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    verify_artifact_directory(result, strict=True)
    evidence = workflow.stage_evidence(
        {"source_commit": "synthetic-source-commit"}, verdict, "synthetic contract fixture"
    )
    evidence["result_dir"] = run_id
    evidence["artifact_manifest"] = {
        "run_id": run_id,
        "path": "artifact_hashes.json",
        "sha256": sha256_file(result / "artifact_hashes.json"),
    }
    state = record_stage(root, "history_mechanism", evidence, "kiki_a14_integral_history_v1")
    assert state["current_stage"] == "history_mechanism"
    assert load_state(root, "kiki_a14_integral_history_v1")["history"][-1]["evidence"]["scientific_verdict"] == verdict
