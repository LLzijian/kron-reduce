from __future__ import annotations

import json
import shutil
import tomllib

import pytest

from project_harness.audit import audit_artifact_layout, audit_project, audit_workflow_registry
from project_harness.data_check import verify_data
from project_harness.loop import approve, load_state, next_stage, record_stage


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
    assert result["event_count"] == 315
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
    (run / "stability_certificate.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    assert audit_stability_certificate_names(tmp_path) == []


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
