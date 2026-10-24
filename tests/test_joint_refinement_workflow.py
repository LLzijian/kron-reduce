import json
import tomllib

import pytest

from kron_reduce.workflows import kiki_joint_refinement as workflow
from project_harness.evidence import EvidenceContext, validate_stage_evidence


def test_refinement_auth_binds_budget_checkpoint_and_source(root):
    config = tomllib.loads((root/"configs/experiments/paper_kiki_a06_refinement.toml").read_text(encoding="utf-8"))
    manifest = json.loads((root/"configs/project_manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiments"]["paper_kiki_a06_refinement"]
    policy = json.loads((root/"configs/research_policy.json").read_text(encoding="utf-8"))
    if policy["development_evaluations"][workflow.AUTH]["status"] == "consumed":
        with pytest.raises(PermissionError):
            workflow.authorization(root, experiment, config)
        return
    workflow.authorization(root, experiment, config)
    for key, value in (("updates_per_fit", 241), ("primary_checkpoint_update", 80),
                       ("history_weight", 1.), ("source_run", "wrong")):
        with pytest.raises((ValueError, PermissionError)):
            workflow.authorization(root, experiment, {**config, key: value})


def comparison():
    records, numerics, los, baselines = [], {}, {}, {}
    for seed in workflow.SEEDS:
        baselines[seed] = {"main": 1., "complete3": 1., "fn": 5, "fp": 0}
        for step in workflow.STEPS:
            records.append({"seed": seed, "checkpoint_step": step, "val_main_nrmse": .9,
                            "val_complete3_nrmse": .95})
            numerics[(step, seed)] = {"checkpoint_replay_exact": True,
                "batched_direct_max_normalized_difference": 0., "rows": [{"qualified": True}]*88}
            los[(step, seed)] = {"rows": [{"qualified": True}]*88, "summary": {
                "eligible": 88, "eligible_unstable": 14, "eligible_stable": 74,
                "numerical_unresolved": 0, "development_evaluation_refusals": 0, "fn": 0, "fp": 0}}
    return records, numerics, los, baselines


def test_primary_endpoint_zero_is_guarded_by_all_completed_fits(tmp_path):
    inputs = comparison()
    assert not workflow.outcome(*inputs)["zero_missed_instability_on_eligible_val"]
    result = workflow.outcome(*inputs, fit_complete=True)
    assert result["zero_missed_instability_on_eligible_val"]
    assert result["zero_missed_instability_on_all_allocated_events"] is False
    validate_stage_evidence(EvidenceContext(tmp_path, {}, "joint_refinement"), {
        "status": "PASS", "process_status": "PASS", "scientific_verdict": result["scientific_verdict"],
        "promotion_decision": result["promotion_decision"]})
    inputs[2][(240, 324)]["summary"]["fn"] = 6
    result = workflow.outcome(*inputs, fit_complete=True)
    assert not result["zero_missed_instability_on_eligible_val"]
    assert result["scientific_verdict"] == "REJECT"


def test_budget_is_once_only_and_progress_is_monotone(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "shared_budget_directory", lambda root: root/".git/a04_fit_budgets")
    result = tmp_path/"results"
    result.mkdir()
    ledger = workflow.RefinementBudget(tmp_path, {"test": 1}, "unit", result)
    ledger.claim(324)
    ledger.progress(324, 1)
    with pytest.raises(FileExistsError):
        ledger.claim(324)
    with pytest.raises(ValueError):
        ledger.progress(324, 3)
    with pytest.raises(FileExistsError):
        workflow.RefinementBudget(tmp_path, {}, "other", result)
    assert ledger.snapshot()["optimizer_steps_reserved"] == 240
