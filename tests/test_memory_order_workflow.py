import json
import shutil
import tomllib
from pathlib import Path

import pytest

import kron_reduce.workflows.kiki_memory_order as memory
from project_harness.report_storage import locate_report

SEEDS = (324, 325, 326)


def _inputs(*, r2_fn=0, r4_fn=0, r2_train=0.5, r4_train=0.4,
            r2_val=0.8, r4_val=0.7, r4_complete=0.9):
    records, numerics, stability = [], {}, {}
    baselines = {seed: {"main": 1.0, "complete3": 1.0, "fn": 1, "fp": 0}
                 for seed in SEEDS}
    for group in memory.GROUPS:
        for seed in SEEDS:
            r2 = group == "r2"
            records.append({"group": group, "seed": seed,
                "train_main_mse": r2_train if r2 else r4_train,
                "train_worst10_weight_cvar_mse": 4.0 if r2 else 1.0,
                "val_main_nrmse": r2_val if r2 else r4_val,
                "val_complete3_nrmse": 0.9 if r2 else r4_complete})
            key = (group, seed)
            numerics[key] = {"rows": [{"qualified": True} for _ in range(88)],
                "checkpoint_replay_exact": True,
                "batched_direct_max_normalized_difference": 0.0}
            fn = r2_fn if r2 else r4_fn
            stability[key] = {"summary": {"fn": fn, "fp": 0, "eligible": 88,
                "eligible_unstable": 14, "eligible_stable": 74,
                "numerical_unresolved": 0, "development_evaluation_refusals": 0},
                "rows": [{"qualified": True} for _ in range(88)]}
    return records, numerics, stability, baselines


def test_zero_group_is_reported_independently_without_selecting_it():
    args = _inputs(r2_fn=0, r4_fn=1)
    result = memory.outcome(*args[:3], args[3], fit_complete=True)
    assert result["group_results"]["r2"]["zero_missed_instability_on_eligible_val"]
    assert not result["group_results"]["r4"]["zero_missed_instability_on_eligible_val"]
    assert result["scientific_verdict"] == "ACCEPT"
    assert result["selected_candidate"] is None


def test_both_zero_groups_are_reported_without_selection():
    args = _inputs(r2_fn=0, r4_fn=0)
    result = memory.outcome(*args[:3], args[3], fit_complete=True)
    assert all(result["group_results"][g]["zero_missed_instability_on_eligible_val"]
               for g in ("r2", "r4"))
    assert result["scientific_verdict"] == "ACCEPT"
    assert result["selected_candidate"] is None


def test_mse_gain_must_strictly_exceed_one_e_minus_twelve():
    args = _inputs(r2_train=0.5, r4_train=0.5 - 1e-12)
    result = memory.outcome(*args[:3], args[3], fit_complete=True)
    assert not result["memory_order_trajectory_gain_supported"]


def test_tail_risk_improvement_cannot_replace_main_train_mse_gain():
    args = list(_inputs(r2_train=0.5, r4_train=0.5, r2_val=0.8, r4_val=0.7))
    # Tail CVaR is strongly better for r4, but the registered main train MSE
    # does not improve, so the mechanism claim must fail.
    result = memory.outcome(*args[:3], args[3], fit_complete=True)
    assert all(row["train_worst10_weight_cvar_mse"] == 1.0
               for row in result["group_results"]["r4"]["paired_against_A06"])
    assert not result["memory_order_trajectory_gain_supported"]


def _authorization_fixture(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs/protocols").mkdir(parents=True)
    (tmp_path / "docs/research").mkdir(parents=True)
    for relative in ("configs/research_policy.json", "configs/project_manifest.json"):
        shutil.copyfile(repository / relative, tmp_path / relative)
    config = tomllib.loads((repository / "configs/experiments/paper_kiki_a08_memory.toml")
                           .read_text(encoding="utf-8"))
    for key in ("development_protocol_path", "design_path"):
        relative = config[key]
        shutil.copyfile(repository / locate_report(repository, relative), tmp_path / relative)
    policy_path = tmp_path / "configs/research_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["development_evaluations"][memory.AUTH]["status"] = "user_approved"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    manifest = json.loads((tmp_path / "configs/project_manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiments"]["paper_kiki_a08_memory"]
    return tmp_path, experiment, config


def test_authorization_accepts_isolated_approved_policy_and_rejects_tampering(tmp_path):
    root, experiment, config = _authorization_fixture(tmp_path)
    assert memory.authorization(root, experiment, config)["status"] == "user_approved"
    for changed in (
        dict(config, source_manifest_sha256="0" * 64),
        dict(config, source_checkpoint_sha256=["0" * 64] * 3),
        dict(config, new_encoder_fraction=0.06),
        dict(config, encoder_penalties=[0.0001, 0.0003]),
        dict(config, svd_relative_tolerance=1e-8),
    ):
        with pytest.raises((PermissionError, ValueError)):
            memory.authorization(root, experiment, changed)


def test_authorization_rejects_consumed_status_even_with_isolated_policy(tmp_path):
    root, experiment, config = _authorization_fixture(tmp_path)
    policy_path = root / "configs/research_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["development_evaluations"][memory.AUTH]["status"] = "consumed"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not bind"):
        memory.authorization(root, experiment, config)


def test_memory_budget_enforces_group_seed_uniqueness_order_and_total(tmp_path, monkeypatch):
    shared = tmp_path / ".git-common"
    shared.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    result_dir = results / "run-1"
    result_dir.mkdir()
    monkeypatch.setattr(memory, "shared_budget_directory", lambda _root: shared)
    ledger = memory.MemoryBudget(tmp_path, {"frozen": True}, "run-1", result_dir)
    for group in memory.GROUPS:
        for seed in SEEDS:
            ledger.claim(group, seed)
    with pytest.raises(FileExistsError):
        ledger.claim("r2", 324)
    with pytest.raises(ValueError, match="nonsequential"):
        ledger.progress("r2", 324, 2)
    ledger.progress("r2", 324, 1)
    with pytest.raises(FileExistsError):
        memory.MemoryBudget(tmp_path, {"frozen": True}, "run-2", result_dir)
    snapshot = ledger.snapshot()
    assert snapshot["fit_attempts_reserved"] == 6
    assert snapshot["optimizer_steps_reserved"] == 1440
    assert [(row["group"], row["seed"]) for row in snapshot["slots"]] == [
        (group, seed) for group in memory.GROUPS for seed in SEEDS]
