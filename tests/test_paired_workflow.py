import json
import tomllib

import pytest

from kron_reduce.workflows import kiki_paired_training as workflow
from project_harness.evidence import EvidenceContext, sha256_file, validate_stage_evidence
from project_harness.loop import next_stage
from project_harness.report_storage import locate_report


def registration(root):
    manifest = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiments"]["paper_kiki_a05_paired"]
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    return experiment, config


def test_authorization_binds_both_plan_and_source(root, tmp_path):
    experiment, config = registration(root)
    # Completed campaigns remain auditable after policy is consumed.
    policy = json.loads((root / "configs/research_policy.json").read_text(encoding="utf-8"))
    approval = policy["development_evaluations"][workflow.AUTH]
    assert approval["status"] in ("user_approved", "consumed")
    digest = approval["approval_protocol_sha256"]
    protocol = locate_report(root, approval["approval_protocol_path"], digest)
    assert sha256_file(root / protocol) == digest
    if approval["status"] == "consumed":
        with pytest.raises(PermissionError):
            workflow.authorization(root, experiment, config)
        return
    assert workflow.authorization(root, experiment, config)[0] == approval
    for key, replacement in (("epochs", 81), ("gradient_clip", 5.),
                             ("history_weight", 1.), ("source_run", "wrong"),
                             ("source_test_payload_allowed", True)):
        with pytest.raises((PermissionError, ValueError)):
            workflow.authorization(root, experiment, {**config, key: replacement})


def test_a05_initial_stage_is_executable(root, tmp_path):
    directory = tmp_path / "configs/workflows"
    directory.mkdir(parents=True)
    (directory / "kiki_a05_paired_v1.json").write_bytes(
        (root / "configs/workflows/kiki_a05_paired_v1.json").read_bytes())
    assert next_stage(tmp_path, "kiki_a05_paired_v1") == "paired_training"


def test_budget_prevents_repeat_campaign_and_duplicate_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "shared_budget_directory", lambda root: root / ".git/a04_fit_budgets")
    result = tmp_path / "results/unit"
    result.mkdir(parents=True)
    ledger = workflow.PairedBudget(tmp_path, {"plan": 1}, "unit", result)
    ledger.claim("F", 324)
    ledger.progress("F", 324, 1)
    with pytest.raises(FileExistsError):
        ledger.progress("F", 324, 1)
    with pytest.raises(ValueError):
        ledger.progress("F", 324, 3)
    with pytest.raises(ValueError):
        ledger.claim("A", 324)
    with pytest.raises(FileExistsError):
        ledger.claim("F", 324)
    with pytest.raises(FileExistsError):
        workflow.PairedBudget(tmp_path, {"plan": 1}, "another", result)
    assert ledger.snapshot()["optimizer_steps_reserved"] == 80
    assert ledger.snapshot()["confirmed_optimizer_steps"] == 1


def test_incomplete_or_unresolved_comparison_cannot_pass():
    verdict = workflow.judge([], {}, {}, {})
    assert not verdict["next_stage_research_supported"]
    assert verdict["promotion_decision"] == "KEEP_BASELINE"
    records, decompositions, numerics, stability = [], {}, {}, {}
    for seed in workflow.SEEDS:
        for group, metric in (("F", .8), ("J", 1.)):
            key = (group, seed)
            records.append({"candidate": group, "seed": seed, "status": "completed",
                            "actual_optimizer_steps": 80, "val_main_nrmse": metric,
                            "val_complete3_nrmse": metric})
            decompositions[key] = {"groups": {"label": {
                "stable": {"same_denominator_mse_contribution": metric},
                "unstable": {"same_denominator_mse_contribution": .1}}}}
            numerics[key] = {"checkpoint_replay_exact": True,
                             "batched_direct_max_normalized_difference": 0.,
                             "rows": [{"qualified": True}] * 88}
            stability[key] = {"rows": [{"qualified": True}] * 88,
                              "summary": {"fn": 1, "tp": 13, "fp": 0, "tn": 74,
                                          "eligible": 88, "history_eligible": 88,
                                          "development_evaluation_refusals": 0,
                                          "numerical_unresolved": 0}}
    assert workflow.judge(records, decompositions, numerics, stability)["next_stage_research_supported"]
    stability[("F", 324)]["summary"]["numerical_unresolved"] = 1
    assert not workflow.judge(records, decompositions, numerics, stability)["next_stage_research_supported"]


@pytest.mark.parametrize("supported", [True, False])
def test_workflow_verdict_uses_registered_schema(tmp_path, supported):
    validate_stage_evidence(EvidenceContext(tmp_path, {}, "paired_training"), {
        "status": "PASS", "process_status": "PASS", "scientific_verdict": workflow.diagnostic_verdict(supported),
        "promotion_decision": "KEEP_BASELINE"})


def test_closeout_preserves_failure_and_rejects_other_failures(tmp_path):
    from kron_reduce.workflows.kiki_paired_closeout import validate_completed_source
    source = tmp_path / "source"
    source.mkdir()
    summary = {"process_status": "PASS", "scientific_verdict": "REJECT_DIAGNOSTIC",
               "checks": {"all_six_fits_completed": True},
               "budget": {"fit_attempts_reserved": 6, "confirmed_optimizer_steps": 480},
               "model_promoted": False, "next_stage_research_supported": False}
    path = source / "learning_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    manifest = source / "artifact_hashes.json"
    manifest.write_text(json.dumps({"learning_summary.json": sha256_file(path)}), encoding="utf-8")
    failure = source / "RUN_FAILED.json"
    failure.write_text(json.dumps({"error_type": "EvidenceValidationError", "message":
        "EVIDENCE_VERDICT: unregistered scientific_verdict: 'REJECT_DIAGNOSTIC'"}), encoding="utf-8")
    assert validate_completed_source(source, sha256_file(manifest), sha256_file(failure))[1] == summary
    failure.write_text(json.dumps({"error_type": "ValueError", "message": "other failure"}), encoding="utf-8")
    with pytest.raises(ValueError, match="other failure"):
        validate_completed_source(source, sha256_file(manifest), sha256_file(failure))
