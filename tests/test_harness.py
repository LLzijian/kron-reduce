from __future__ import annotations

import json
import shutil
import sys
import tomllib
import types

import pytest

from project_harness.audit import audit_artifact_layout, audit_project, audit_workflow_registry
from project_harness.data_check import verify_data
from project_harness.evidence import (
    EvidenceContext,
    EvidenceValidationError,
    sha256_file,
    validate_stage_evidence,
    verify_artifact_directory,
)
from project_harness.loop import approve, load_state, next_stage, record_stage
from project_harness.runner import run_experiment


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
    result = verify_data(root)
    assert result["status"] == "PASS"
    assert result["event_count"] == 591
    assert result["holdout_payload_arrays_read"] == 0
    assert result["test_status"] == "prior_exposure_detected_not_fresh"
    assert result["auxiliary_datasets"]["synthetic_benchmark_registry_v1"]["status"] == "PASS"


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
        (root / "configs" / "workflows" / "ten_machine_physical_discovery_v1.json").read_text(
            encoding="utf-8"
        )
    )
    stages = workflow["stages"]
    assert stages.index("source_recommit") == stages.index("source_commit") + 1
    assert stages.index("source_finalize_commit") == stages.index("source_recommit") + 1
    assert stages.index("diagnostic") == stages.index("source_finalize_commit") + 1


def test_esc_workflow_registered_with_merge_gate_and_chain(root):
    workflow = json.loads(
        (
            root / "configs" / "workflows" / "m0_esc_v1_excitation_conditioned_selection.json"
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
    registry = manifest["datasets"]["synthetic_benchmark_registry_v1"]
    assert registry["status"] == "ready"
    assert registry["checksum_manifest"] == "checksums.sha256"


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
    model = run / "model.json"; model.write_text("{}", encoding="utf-8")
    synthetic = run / "synthetic.json"; synthetic.write_text("{}", encoding="utf-8")
    ieee39 = run / "ieee39.json"; ieee39.write_text("{}", encoding="utf-8")
    groups = run / "groups.json"
    groups.write_text(json.dumps({"unit": "fault_line_x_operating_point", "status": "PASS", "independent_unstable_groups": 80, "duplicate_groups": 0}), encoding="utf-8")
    binding = lambda path: {"path": path.name, "sha256": sha256_file(path)}
    certificate = run / "stability_certificate.json"
    certificate.write_text(json.dumps({
        "schema": "kron_reduce.level6_certificate.v1",
        "run_id": run.name, "workflow": "fixture", "sample_based": True,
        "integration_free": True, "claim_boundary": "registered envelope",
        "metrics": {"region_conservative": True, "stable_envelope_coverage": .5,
                    "region_symmetric_difference_ratio": .2, "normalized_margin_mae": .1,
                    "margin_rank_correlation": .8, "independent_unstable_groups": 80,
                    "fsr_95_upper": .05, "false_stable_certificates": 0,
                    "physics_gates_passed": True, "numerical_gates_passed": True,
                    "claim_boundary": "registered envelope"},
        "bindings": {"model_artifact": binding(model), "dataset_evidence": {
            "synthetic": binding(synthetic), "ieee39": binding(ieee39)},
            "independent_group_audit": binding(groups)},
    }), encoding="utf-8")
    (run / "acceptance.json").write_text(json.dumps({
        "model_reporting_name": "stability-certified equivalent",
        "level6_certificate": binding(certificate),
    }), encoding="utf-8")
    assert audit_stability_certificate_names(tmp_path) == []
    frozen_payload = json.loads(certificate.read_text(encoding="utf-8"))
    for bad_datasets in (None, {}, [], [{"role": "synthetic_ground_truth"}]):
        payload = json.loads(json.dumps(frozen_payload))
        payload["bindings"]["dataset_evidence"] = bad_datasets
        certificate.write_text(json.dumps(payload), encoding="utf-8")
        (run / "acceptance.json").write_text(json.dumps({
            "model_reporting_name": " STABILITY-CERTIFIED equivalent ",
            "level6_certificate": binding(certificate),
        }), encoding="utf-8")
        assert audit_stability_certificate_names(tmp_path), bad_datasets
    certificate.write_text(json.dumps(frozen_payload), encoding="utf-8")
    groups.write_text(json.dumps({"unit": "fault_line_x_operating_point", "independent_unstable_groups": 79, "duplicate_groups": 0}), encoding="utf-8")
    payload = json.loads(certificate.read_text(encoding="utf-8"))
    payload["bindings"]["independent_group_audit"] = binding(groups)
    certificate.write_text(json.dumps(payload), encoding="utf-8")
    (run / "acceptance.json").write_text(json.dumps({
        "model_reporting_name": "stability-certified equivalent",
        "level6_certificate": binding(certificate),
    }), encoding="utf-8")
    assert audit_stability_certificate_names(tmp_path)[0]["check"] == "stability_certificate"


def test_m2a_lyap_is_not_automatically_a_level6_claim(tmp_path):
    from project_harness.audit import audit_stability_certificate_names

    run = tmp_path / "results" / "historical-numerical-certificate"
    run.mkdir(parents=True)
    (run / "acceptance.json").write_text(
        json.dumps({"model_reporting_name": "M2A-LYAP"}), encoding="utf-8"
    )
    # Its separate numerical gates remain necessary; this is only the Level 6 audit.
    assert audit_stability_certificate_names(tmp_path) == []


def test_v2_report_evidence_uses_real_predecessor_and_artifact_run(tmp_path):
    run = tmp_path / "results" / "run-a"
    run.mkdir(parents=True)
    artifact = run / "evidence.json"; artifact.write_text("{}", encoding="utf-8")
    acceptance = run / "acceptance.json"; acceptance.write_text("{}", encoding="utf-8")
    manifest = run / "artifact_hashes.json"
    manifest.write_text(json.dumps({"evidence.json": sha256_file(artifact), "acceptance.json": sha256_file(acceptance)}), encoding="utf-8")
    report = tmp_path / "docs" / "reports" / "report.md"
    report.parent.mkdir(parents=True); report.write_text("report", encoding="utf-8")
    negative = tmp_path / "results" / "negative-a"; negative.mkdir()
    negative_file = negative / "negative.json"; negative_file.write_text("{}", encoding="utf-8")
    (negative / "artifact_hashes.json").write_text(json.dumps({"negative.json": sha256_file(negative_file)}), encoding="utf-8")
    workflow = {
        "evidence_contract_version": 2,
        "stage_evidence_contracts": {"report": {"required_fields": [
            "status", "branch_terminal", "acceptance_run", "artifact_manifest", "report_path"
        ]}},
    }
    evidence = {"status": "PASS", "branch_terminal": "rollback_report",
                "acceptance_run": "run-a", "report_path": "docs/reports/report.md",
                "artifact_manifest": {"run_id": "run-a", "path": "artifact_hashes.json",
                                      "sha256": sha256_file(manifest)},
                "report_artifact": {"path": "docs/reports/report.md", "sha256": sha256_file(report)},
                    "negative_evidence_run": "negative-a", "rollback_target": "baseline"}
    context = EvidenceContext(tmp_path, workflow, "report", state={"history": [
        {"action": "advance", "to": "acceptance", "evidence": {"result_dir": "run-a", "artifact_manifest": evidence["artifact_manifest"], "rollback_target": "baseline"}},
        {"action": "advance", "to": "rollback_report", "evidence": {"negative_evidence_run": "negative-a", "rollback_target": "baseline"}},
    ]}, predecessor="rollback_report")
    validate_stage_evidence(context, evidence)
    wrong_branch = dict(evidence, branch_terminal="merge_gate")
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_BRANCH_TERMINAL"):
        validate_stage_evidence(context, wrong_branch)
    absent = dict(evidence, artifact_manifest=dict(evidence["artifact_manifest"], run_id="abc"), acceptance_run="abc")
    with pytest.raises(EvidenceValidationError, match="ARTIFACT_DIRECTORY"):
        validate_stage_evidence(context, absent)
    escaped = dict(evidence, artifact_manifest=dict(evidence["artifact_manifest"], path="../artifact_hashes.json"))
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_ARTIFACT_PATH"):
        validate_stage_evidence(context, escaped)
    wrong_rollback = dict(evidence, rollback_target="invented-baseline")
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_ROLLBACK_TARGET"):
        validate_stage_evidence(context, wrong_rollback)
    own_run_without_binding = dict(evidence, result_dir="invented-report-run")
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_ARTIFACT_RUN"):
        validate_stage_evidence(context, own_run_without_binding)
    report.write_text("changed report", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_REPORT_HASH"):
        validate_stage_evidence(context, evidence)


def test_evidence_rejects_explicit_process_failure_in_runtime_and_audit(tmp_path):
    _write_branching_workflow(tmp_path)
    evidence = _acceptance_evidence("ACCEPT")
    evidence["process_status"] = "FAIL"
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_PROCESS_STATUS"):
        record_stage(tmp_path, "acceptance", evidence, "branching_test")
    state_dir = tmp_path / ".project_state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "branching_test_state.json").write_text(json.dumps({
        "workflow": "branching_test", "current_stage": "acceptance", "approvals": {},
        "history": [{"action": "advance", "from": "scaffold", "to": "acceptance", "evidence": evidence}],
    }), encoding="utf-8")
    finding = audit_workflow_registry(tmp_path)[0]
    assert finding["check"] == "workflow_state_history"
    assert "EVIDENCE_PROCESS_STATUS" in finding["message"]


def test_strict_artifact_verifier_rejects_extra_file_and_bad_hash(tmp_path):
    artifact = tmp_path / "answer.json"; artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "artifact_hashes.json"
    manifest.write_text(json.dumps({"answer.json": sha256_file(artifact)}), encoding="utf-8")
    assert verify_artifact_directory(tmp_path, strict=True) == {"answer.json": sha256_file(artifact)}
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="ARTIFACT_INVENTORY"):
        verify_artifact_directory(tmp_path, strict=True)
    (tmp_path / "extra.txt").unlink()
    artifact.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="ARTIFACT_HASH"):
        verify_artifact_directory(tmp_path, strict=True)


def test_runner_merges_stage_evidence_but_rejects_runner_field_conflicts(tmp_path, monkeypatch):
    def configure(root, module_name):
        (root / "configs" / "workflows").mkdir(parents=True)
        (root / "configs" / "workflows" / "runner_test.json").write_text(json.dumps({
            "workflow": "runner_test", "initial_stage": "scaffold",
            "stages": ["scaffold", "gate"], "transitions": {"scaffold": "gate"},
            "terminal_stages": ["gate"], "approval_required": [],
        }), encoding="utf-8")
        (root / "configs" / "project_manifest.json").write_text(json.dumps({
            "project": "kron-reduce", "experiments": {"fixture": {
                "module": module_name, "workflow": "runner_test"}},
        }), encoding="utf-8")

    module = types.ModuleType("stage_evidence_fixture")
    module.run = lambda **kwargs: (kwargs["result_dir"] / "stage_evidence.json").write_text(
        json.dumps({"status": "PASS", "marker": "merged"}), encoding="utf-8"
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    good = tmp_path / "good"; configure(good, module.__name__)
    run_experiment(good, "fixture", "gate")
    assert load_state(good, "runner_test")["history"][-1]["evidence"]["marker"] == "merged"

    conflict = types.ModuleType("stage_evidence_conflict")
    conflict.run = lambda **kwargs: (kwargs["result_dir"] / "stage_evidence.json").write_text(
        json.dumps({"status": "PASS", "result_dir": "forged"}), encoding="utf-8"
    )
    monkeypatch.setitem(sys.modules, conflict.__name__, conflict)
    bad = tmp_path / "bad"; configure(bad, conflict.__name__)
    with pytest.raises(RuntimeError, match="may not override runner result_dir"):
        run_experiment(bad, "fixture", "gate")
    assert len(list((bad / "results").glob("*/RUN_FAILED.json"))) == 1
    assert load_state(bad, "runner_test")["current_stage"] == "scaffold"


def test_v2_runner_injects_authoritative_frozen_manifest_binding(tmp_path, monkeypatch):
    root = tmp_path / "v2"
    (root / "configs" / "workflows").mkdir(parents=True)
    (root / "configs" / "workflows" / "runner_v2.json").write_text(json.dumps({
        "workflow": "runner_v2", "evidence_contract_version": 2,
        "initial_stage": "scaffold", "stages": ["scaffold", "plan_registered"],
        "transitions": {"scaffold": "plan_registered"}, "terminal_stages": ["plan_registered"],
        "approval_required": [], "stage_evidence_contracts": {"plan_registered": {
            "required_fields": ["status", "result_dir"]}},
    }), encoding="utf-8")
    (root / "configs" / "project_manifest.json").write_text(json.dumps({
        "project": "kron-reduce", "experiments": {"fixture": {
            "module": "runner_v2_fixture", "workflow": "runner_v2"}},
    }), encoding="utf-8")
    module = types.ModuleType("runner_v2_fixture")
    def run(**kwargs):
        evidence = kwargs["result_dir"] / "stage_evidence.json"
        evidence.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        (kwargs["result_dir"] / "artifact_hashes.json").write_text(
            json.dumps({"stage_evidence.json": sha256_file(evidence)}), encoding="utf-8"
        )
    module.run = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = run_experiment(root, "fixture", "plan_registered")
    evidence = load_state(root, "runner_v2")["history"][-1]["evidence"]
    assert evidence["artifact_manifest"]["run_id"] == result.name
    assert evidence["result_artifact_manifest"] == evidence["artifact_manifest"]
    saved = result / "stage_evidence.json"
    saved.write_text(json.dumps({"status": "PASS", "forged": True}), encoding="utf-8")
    (result / "artifact_hashes.json").write_text(
        json.dumps({"stage_evidence.json": sha256_file(saved)}), encoding="utf-8"
    )
    assert any("ARTIFACT_MANIFEST_SHA" in item["message"] for item in audit_workflow_registry(root))


@pytest.mark.parametrize("relative", ["../x", "./x", "folder//x", "x:stream", "C:/x"])
def test_artifact_manifest_rejects_noncanonical_paths(tmp_path, relative):
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps({relative: "a" * 64}), encoding="utf-8"
    )
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_PATH"):
        verify_artifact_directory(tmp_path, strict=True)


def test_artifact_manifest_rejects_duplicate_json_keys(tmp_path):
    (tmp_path / "artifact_hashes.json").write_text(
        '{"x":"' + "a" * 64 + '","x":"' + "b" * 64 + '"}', encoding="utf-8"
    )
    with pytest.raises(EvidenceValidationError):
        verify_artifact_directory(tmp_path, strict=True)


def _write_branching_workflow(root):
    workflow_dir = root / "configs" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "branching_test.json").write_text(
        json.dumps(
            {
                "workflow": "branching_test",
                "initial_stage": "scaffold",
                "stages": ["scaffold", "acceptance", "merge_gate", "rollback_report"],
                "transitions": {
                    "scaffold": "acceptance",
                    "acceptance": {
                        "ACCEPT": "merge_gate",
                        "REJECT": "rollback_report",
                        "RETRY": "rollback_report",
                    },
                },
                "terminal_stages": ["merge_gate", "rollback_report"],
                "approval_required": ["merge_gate"],
                "stage_evidence_contracts": {
                    "acceptance": {
                        "required_fields": [
                            "status",
                            "process_status",
                            "scientific_verdict",
                            "promotion_decision",
                            "rollback_target",
                            "claim_boundary",
                        ]
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _acceptance_evidence(verdict):
    return {
        "status": "PASS",
        "process_status": "PASS",
        "scientific_verdict": verdict,
        "promotion_decision": "PROMOTE" if verdict == "ACCEPT" else "KEEP_BASELINE",
        "rollback_target": "baseline-commit",
        "claim_boundary": "registered diagnostic scope",
    }


def test_branching_workflow_routes_rejection_to_rollback(tmp_path):
    _write_branching_workflow(tmp_path)
    record_stage(tmp_path, "acceptance", _acceptance_evidence("REJECT"), "branching_test")
    assert next_stage(tmp_path, "branching_test") == "rollback_report"
    with pytest.raises(RuntimeError, match="expected 'rollback_report'"):
        record_stage(tmp_path, "merge_gate", {"status": "PASS"}, "branching_test")
    record_stage(
        tmp_path,
        "rollback_report",
        {"status": "PASS", "scientific_verdict": "REJECT"},
        "branching_test",
    )
    assert next_stage(tmp_path, "branching_test") is None
    assert audit_workflow_registry(tmp_path) == []


def test_branching_workflow_requires_acceptance_contract_and_merge_approval(tmp_path):
    _write_branching_workflow(tmp_path)
    with pytest.raises(RuntimeError, match="missing required fields"):
        record_stage(tmp_path, "acceptance", {"status": "PASS"}, "branching_test")
    record_stage(tmp_path, "acceptance", _acceptance_evidence("ACCEPT"), "branching_test")
    assert next_stage(tmp_path, "branching_test") == "merge_gate"
    with pytest.raises(RuntimeError, match="requires explicit approval"):
        record_stage(tmp_path, "merge_gate", {"status": "PASS"}, "branching_test")
    approve(tmp_path, "merge_gate", "branching_test")
    record_stage(tmp_path, "merge_gate", {"status": "PASS"}, "branching_test")
    assert next_stage(tmp_path, "branching_test") is None
    assert audit_workflow_registry(tmp_path) == []


def test_master_workflow_registers_single_model_and_level6_boundaries(root):
    workflow = json.loads(
        (
            root
            / "configs"
            / "workflows"
            / "ieee39_modular_energy_paper_master_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert workflow["subworkflow_template"] == "ieee39_modular_energy_loop_template_v1"
    assert "multiple_models_for_one_input_corpus" in workflow["locked_claims"]
    assert "stability_certified_equivalent_without_level6_gates" in workflow["locked_claims"]
    assert workflow["stages"][-1] == "publication_gate"


def _p1_a01_acceptance_evidence(verdict):
    return {
        "status": "PASS",
        "process_status": "PASS",
        "scientific_verdict": verdict,
        "promotion_decision": "KEEP_BASELINE",
        "rollback_target": "accepted-causal-baseline",
        "claim_boundary": "P1 diagnostic scope only",
        "paper_usable_claims": ["registered diagnostic finding"],
    }


def _advance_p1_a01_to_acceptance(root):
    workflow_name = "ieee39_modular_energy_p1_a01_v1"
    for stage in [
        "plan_registered", "candidate_acquired", "model_unit_gate", "source_commit",
        "controlled_experiment", "residual_diagnosis", "adversarial_review",
    ]:
        record_stage(root, stage, {"status": "PASS"}, workflow_name)


def test_p1_a01_workflow_both_branches_end_at_report_and_merge_requires_approval(root, tmp_path):
    workflow_name = "ieee39_modular_energy_p1_a01_v1"
    accepted_root = tmp_path / "accepted"
    rejected_root = tmp_path / "rejected"
    for execution_root in [accepted_root, rejected_root]:
        workflow_dir = execution_root / "configs" / "workflows"
        workflow_dir.mkdir(parents=True)
        shutil.copy2(root / "configs" / "workflows" / f"{workflow_name}.json", workflow_dir)

    _advance_p1_a01_to_acceptance(accepted_root)
    invalid_promotion = _p1_a01_acceptance_evidence("ACCEPT")
    invalid_promotion["promotion_decision"] = "PROMOTE"
    with pytest.raises(RuntimeError, match="promotion_decision.*KEEP_BASELINE"):
        record_stage(accepted_root, "acceptance", invalid_promotion, workflow_name)
    record_stage(accepted_root, "acceptance", _p1_a01_acceptance_evidence("ACCEPT"), workflow_name)
    assert next_stage(accepted_root, workflow_name) == "merge_gate"
    with pytest.raises(RuntimeError, match="requires explicit approval"):
        record_stage(accepted_root, "merge_gate", {"status": "PASS"}, workflow_name)
    approve(accepted_root, "merge_gate", workflow_name)
    record_stage(accepted_root, "merge_gate", {"status": "PASS"}, workflow_name)
    with pytest.raises(RuntimeError, match="empty required fields"):
        record_stage(accepted_root, "report", {"status": "PASS", "branch_terminal": "merge_gate", "acceptance_run": "run-a", "artifact_hashes": {}, "report_path": "docs/reports/p1-a01.md"}, workflow_name)
    with pytest.raises(RuntimeError, match="missing conditional fields"):
        record_stage(accepted_root, "report", {"status": "PASS", "branch_terminal": "merge_gate", "acceptance_run": "run-a", "artifact_hashes": {"acceptance.json": "abc"}, "report_path": "docs/reports/p1-a01.md"}, workflow_name)
    with pytest.raises(RuntimeError, match="missing conditional fields"):
        record_stage(accepted_root, "report", {"status": "PASS", "branch_terminal": "merge_gate", "acceptance_run": "run-a", "artifact_hashes": {"acceptance.json": "abc"}, "report_path": "docs/reports/p1-a01.md", "merge_commit": ""}, workflow_name)
    record_stage(accepted_root, "report", {"status": "PASS", "branch_terminal": "merge_gate", "acceptance_run": "run-a", "artifact_hashes": {"acceptance.json": "abc"}, "report_path": "docs/reports/p1-a01.md", "merge_commit": "deadbeef"}, workflow_name)
    assert next_stage(accepted_root, workflow_name) is None

    _advance_p1_a01_to_acceptance(rejected_root)
    record_stage(rejected_root, "acceptance", _p1_a01_acceptance_evidence("RETRY"), workflow_name)
    assert next_stage(rejected_root, workflow_name) == "rollback_report"
    with pytest.raises(RuntimeError, match="missing required fields"):
        record_stage(
            rejected_root,
            "rollback_report",
            {
                "status": "PASS",
                "process_status": "PASS",
                "scientific_verdict": "RETRY",
                "promotion_decision": "KEEP_BASELINE",
                "next_hypothesis": "registered next diagnosis",
            },
            workflow_name,
        )
    record_stage(rejected_root, "rollback_report", {"status": "PASS", "process_status": "PASS", "scientific_verdict": "RETRY", "promotion_decision": "KEEP_BASELINE", "rollback_target": "accepted-causal-baseline", "negative_evidence_run": "run-r", "next_hypothesis": "registered next diagnosis"}, workflow_name)
    with pytest.raises(RuntimeError, match="missing conditional fields"):
        record_stage(rejected_root, "report", {"status": "PASS", "branch_terminal": "rollback_report", "acceptance_run": "run-r", "artifact_hashes": {"rollback.json": "abc"}, "report_path": "docs/reports/p1-a01.md"}, workflow_name)
    with pytest.raises(RuntimeError, match="missing conditional fields"):
        record_stage(rejected_root, "report", {"status": "PASS", "branch_terminal": "rollback_report", "acceptance_run": "run-r", "artifact_hashes": {"rollback.json": "abc"}, "report_path": "docs/reports/p1-a01.md", "rollback_target": "", "negative_evidence_run": "run-r"}, workflow_name)
    record_stage(rejected_root, "report", {"status": "PASS", "branch_terminal": "rollback_report", "acceptance_run": "run-r", "artifact_hashes": {"rollback.json": "abc"}, "report_path": "docs/reports/p1-a01.md", "rollback_target": "accepted-causal-baseline", "negative_evidence_run": "run-r"}, workflow_name)
    assert next_stage(rejected_root, workflow_name) is None


def test_p1_a01_protocol_config_and_manifest_are_locked(root):
    workflow_name = "ieee39_modular_energy_p1_a01_v1"
    config_name = "paper_ieee39_modular_energy_p1_a01_f1_seed324.toml"
    config = tomllib.loads((root / "configs" / "experiments" / config_name).read_text(encoding="utf-8"))
    assert config["branch"] == "codex/loop-p1-a01"
    assert config["fold"] == "F1" and config["seed"] == 324
    assert config["model_count"] == 1 and config["source_routing"] is False
    assert config["test_access"] == "locked" and config["locked_folds"] == ["F2", "F3", "F4", "F5"]
    assert config["primary_pm_policy"] == "zoh_pm_clear"
    assert config["observed_pm_policy"] == "conditional_only"
    assert config["bootstrap_draws"] == 2000 and config["bootstrap_seed"] == 324
    assert config["minimum_independent_lines_for_modal_verdict"] == 3
    assert config["fewer_than_three_independent_lines_verdict"] == "UNDETERMINED"
    assert config["module_thresholds"] == {"tau_y_floor": 1e-3, "tau_y_multiplier": 10.0, "tau_phys_floor": 1e-4, "tau_phys_multiplier": 100.0, "deletion_noninferiority_degradation": 0.02, "identifiability_sigma_ratio_min": 1e-6, "identifiability_condition_max": 1e6, "identifiability_profile_tolerance": 0.02}
    assert config["trajectory_thresholds"] == {"speed_improvement_min": 0.05, "non_target_degradation_max": 0.02, "causal_closure_observed_gap_recovery_min": 0.50}
    assert config["modal_thresholds"] == {"finite_window_energy_min": 0.01, "frequency_relative_error_max": 0.05, "damping_absolute_error_max": 0.05, "mac_min": 0.80, "bootstrap_support_min": 0.80}
    assert config["physical_numerical_thresholds"] == {"equilibrium_residual_max": 1e-7, "observability_min": 1e-3, "coi_drift_max": 1e-10, "supply_identity_error_max": 1e-8, "rk4_order_ratio_lower": 8.0, "rk4_order_ratio_upper": 32.0}
    assert config["controlled_arms"]["names"] == ["b2_zoh_original_loss", "current_zoh_original_loss", "current_zoh_balanced_loss", "current_observed_pm_balanced_loss_conditional_upper_bound", "current_causal_pm_closure_balanced_loss"]

    manifest = json.loads((root / "configs" / "project_manifest.json").read_text(encoding="utf-8"))
    names = {f"paper_ieee39_modular_energy_p1_a01_{stage}_f1_seed324" for stage in ["plan_registered", "candidate_acquired", "controlled_experiment", "residual_diagnosis", "adversarial_review", "acceptance", "rollback_report", "report"]}
    assert names.issubset(manifest["experiments"])
    for name in names:
        entry = manifest["experiments"][name]
        assert entry["module"] == "kron_reduce.workflows.ieee39_modular_energy_p1_diagnosis"
        assert entry["workflow"] == workflow_name
        assert entry["config"] == config_name
        assert entry["dataset"] == "ieee39_postclear_canonical_v1"
        assert entry["reads_test_performance"] is False
        assert entry["observed_pm_declaration"] == "conditional_only_not_main_model_input"
