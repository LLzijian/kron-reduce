from __future__ import annotations

import shutil

import pytest

from project_harness.audit import audit_project
from project_harness.data_check import verify_data
from project_harness.loop import load_state, next_stage, record_stage


def test_project_audit(root):
    assert audit_project(root)["status"] == "PASS"


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
