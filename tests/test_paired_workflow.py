import json
import tomllib

import pytest

from kron_reduce.workflows import kiki_paired_training as workflow
from project_harness.evidence import sha256_file
from project_harness.loop import next_stage


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
    assert sha256_file(root / approval["approval_protocol_path"]) == approval["approval_protocol_sha256"]
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
