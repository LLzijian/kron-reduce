from __future__ import annotations

import json
import shutil

import pytest

from project_harness.audit import audit_artifact_layout, audit_project, audit_workflow_registry
from project_harness.data_check import verify_data
from project_harness.loop import load_state, next_stage, record_stage


def test_project_audit(root):
    assert audit_project(root)["status"] == "PASS"


def test_artifact_layout_rejects_report_at_reports_root(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "manual_result.md").write_text("misplaced", encoding="utf-8")
    findings = audit_artifact_layout(tmp_path)
    assert findings[0]["check"] == "artifact_layout"


def test_registered_workflow_states_are_contiguous(root):
    assert audit_workflow_registry(root) == []


def test_data_checksums(root):
    result=verify_data(root); assert result["status"]=="PASS"; assert result["event_count"]==315


def test_loop_starts_at_scaffold_and_cannot_skip(root, tmp_path):
    workflow_dir = tmp_path / "configs" / "workflows"
    workflow_dir.mkdir(parents=True)
    shutil.copy2(root / "configs" / "workflows" / "paper_v1.json", workflow_dir)
    state = load_state(tmp_path)
    assert state["current_stage"] == "scaffold"
    assert next_stage(tmp_path) == "data_verified"
    with pytest.raises(RuntimeError, match="expected 'data_verified'"):
        record_stage(tmp_path, "smoke", {"status": "PASS"})


def test_named_workflow_has_independent_state(root, tmp_path):
    workflow_dir = tmp_path / "configs" / "workflows"
    workflow_dir.mkdir(parents=True)
    shutil.copy2(root / "configs" / "workflows" / "paper_v1.json", workflow_dir)
    shutil.copy2(root / "configs" / "workflows" / "paper_v2_integral.json", workflow_dir)
    v1 = load_state(tmp_path, "paper_v1")
    v2 = load_state(tmp_path, "paper_v2_integral")
    assert v1["workflow"] == "paper_v1"
    assert v2["workflow"] == "paper_v2_integral"
    record_stage(tmp_path, "data_verified", {"status": "PASS"}, "paper_v2_integral")
    assert load_state(tmp_path, "paper_v1")["current_stage"] == "scaffold"
    assert load_state(tmp_path, "paper_v2_integral")["current_stage"] == "data_verified"


def test_corrective_workflow_has_source_recommit_before_diagnostic(root):
    workflow = json.loads(
        (
            root
            / "configs"
            / "workflows"
            / "ten_machine_physical_discovery_v1.json"
        ).read_text(encoding="utf-8")
    )
    stages = workflow["stages"]
    assert stages.index("source_recommit") == stages.index("source_commit") + 1
    assert stages.index("source_finalize_commit") == stages.index("source_recommit") + 1
    assert stages.index("diagnostic") == stages.index("source_finalize_commit") + 1
