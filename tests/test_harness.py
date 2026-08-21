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


def test_esc_workflow_registered_with_merge_gate_and_chain(root):
    workflow = json.loads(
        (
            root
            / "configs"
            / "workflows"
            / "m0_esc_v1_excitation_conditioned_selection.json"
        ).read_text(encoding="utf-8")
    )
    assert workflow["stages"][-1] == "merge_gate"
    assert workflow["approval_required"] == ["merge_gate"]
    chain = [
        "hypothesis_registration",
        "controlled_experiment",
        "residual_diagnosis",
        "adversarial_review",
        "acceptance",
    ]
    indices = [workflow["stages"].index(stage) for stage in chain]
    assert indices == sorted(indices)
    assert "stability_margin_without_level6_gates" in workflow["locked_claims"]


def test_esc_experiments_registered_in_manifest(root):
    manifest = json.loads((root / "configs" / "project_manifest.json").read_text(encoding="utf-8"))
    names = {
        "paper_m0_esc_v1_hypothesis_f1_seed324",
        "paper_m0_esc_v1_controlled_f1_seed324",
        "paper_m0_esc_v1_residual_f1_seed324",
        "paper_m0_esc_v1_adversarial_f1_seed324",
        "paper_m0_esc_v1_acceptance_f1_seed324",
    }
    assert names.issubset(manifest["experiments"])
    for name in names:
        entry = manifest["experiments"][name]
        assert entry["workflow"] == "m0_esc_v1_excitation_conditioned_selection"
        assert entry["reads_test_performance"] is False
        assert entry["synthetic_registry"] == "synthetic_benchmark_registry_v1"
    assert manifest["datasets"]["synthetic_benchmark_registry_v1"]["status"] == "planned"


def test_stability_certificate_naming_audit(tmp_path):
    from project_harness.audit import audit_stability_certificate_names

    run = tmp_path / "results" / "fake-acceptance-20260820T000000Z-deadbeef"
    run.mkdir(parents=True)
    (run / "acceptance.json").write_text(
        json.dumps({"model_reporting_name": "stability-certified equivalent"}),
        encoding="utf-8",
    )
    findings = audit_stability_certificate_names(tmp_path)
    assert findings[0]["check"] == "stability_certificate"
    (run / "stability_certificate.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    assert audit_stability_certificate_names(tmp_path) == []
