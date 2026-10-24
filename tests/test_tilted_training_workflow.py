import json
import math
import shutil
import tomllib
from pathlib import Path

import pytest

import kron_reduce.workflows.kiki_tilted_training as tilted

SEEDS = (324, 325, 326)


def _score(values):
    rows = []
    for index, value in enumerate(values):
        rows.append({"event_id": f"e{index}", "source_dataset": "source",
                     "fault_line": "line", "role": "train", "status": "ok",
                     "normalized_mse": value})
    mean_mse = sum(values) / len(values)
    horizons = {"rows": rows}
    return {"available": {"horizons": {"1.0": horizons, "3.0": horizons},
                           "multi_horizon_normalized_rmse": math.sqrt(mean_mse)}}


def _inputs(*, m_fn=0, t_fn=0, m_tail=3.0, t_tail=2.0,
            m_val=0.8, t_val=0.7, t_complete=0.9,
            m_numeric_ok=True, t_numeric_ok=True):
    records, numerics, stability = [], {}, {}
    baselines = {seed: {"main": 1.0, "complete3": 1.0, "fn": 1, "fp": 0}
                 for seed in SEEDS}
    for group in ("M", "T"):
        for seed in SEEDS:
            records.append({"group": group, "seed": seed,
                "val_main_nrmse": m_val if group == "M" else t_val,
                "val_complete3_nrmse": 0.9 if group == "M" else t_complete,
                "train_worst10_weight_cvar_mse": m_tail if group == "M" else t_tail})
            key = group, seed
            numeric_ok = m_numeric_ok if group == "M" else t_numeric_ok
            numerics[key] = {"rows": [{"qualified": numeric_ok} for _ in range(88)],
                "checkpoint_replay_exact": True,
                "batched_direct_max_normalized_difference": 0.0}
            fn = m_fn if group == "M" else t_fn
            stability[key] = {"summary": {"fn": fn, "fp": 0, "eligible": 88,
                "eligible_unstable": 14, "eligible_stable": 74,
                "numerical_unresolved": 0, "development_evaluation_refusals": 0},
                "rows": [{"qualified": numeric_ok} for _ in range(88)]}
    return records, numerics, stability, baselines


def test_train_tail_risk_uses_fractional_boundary_weight():
    # Four equal-weight events, with alpha=0.375: all of the worst event and
    # half of the second-worst event form the requested tail mass.
    value = tilted.train_tail_risk(_score([4.0, 3.0, 2.0, 1.0]), alpha=0.375)
    assert value == pytest.approx((0.25 * 4.0 + 0.125 * 3.0) / 0.375)


def test_tail_risk_rejects_failed_event_instead_of_dropping_it():
    score = _score([4.0, 3.0, 2.0, 1.0])
    score["available"]["horizons"]["3.0"]["rows"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="failed or mismatched"):
        tilted.train_tail_risk(score, alpha=0.1)


def test_zero_is_reported_per_group_without_selecting_a_group():
    args = _inputs(m_fn=0, t_fn=1)
    result = tilted.outcome(*args[:3], args[3], fit_complete=True)
    assert result["group_results"]["M"]["zero_missed_instability_on_eligible_val"]
    assert not result["group_results"]["T"]["zero_missed_instability_on_eligible_val"]
    assert result["tilted_zero_hypothesis_rejected_when_control_zero"]
    assert result["scientific_verdict"] == "ACCEPT"
    assert result["selected_candidate"] is None


def test_both_groups_can_be_zero_without_group_selection():
    args = _inputs(m_fn=0, t_fn=0)
    result = tilted.outcome(*args[:3], args[3], fit_complete=True)
    assert result["group_results"]["M"]["zero_missed_instability_on_eligible_val"]
    assert result["group_results"]["T"]["zero_missed_instability_on_eligible_val"]
    assert result["scientific_verdict"] == "ACCEPT"
    assert result["selected_candidate"] is None


def test_tilted_tail_improvement_does_not_pass_when_val_guard_degrades():
    args = _inputs(m_fn=2, t_fn=1, m_tail=4.0, t_tail=3.0,
                   m_val=0.7, t_val=0.8, t_complete=0.9)
    result = tilted.outcome(*args[:3], args[3], fit_complete=True)
    assert not result["tilted_mechanism_supported"]
    assert all(not row["tail_and_val_guards"] for row in result["mechanism_results"])


@pytest.mark.parametrize("mutation,fit_complete,m_numeric_ok,t_numeric_ok", [
    ("missing", True, True, True), ("nan", True, True, True),
    (None, False, True, True), (None, True, False, True),
])
def test_incomplete_nonfinite_or_numerically_failed_group_cannot_pass(
        mutation, fit_complete, m_numeric_ok, t_numeric_ok):
    args = list(_inputs(m_fn=2, t_fn=0, m_numeric_ok=m_numeric_ok,
                        t_numeric_ok=t_numeric_ok))
    if mutation == "missing":
        args[0][0].pop("val_main_nrmse")
    elif mutation == "nan":
        args[0][0]["val_main_nrmse"] = float("nan")
    result = tilted.outcome(*args[:3], args[3], fit_complete=fit_complete)
    assert not result["group_results"]["M"]["all_terminal_safeguards"]
    if fit_complete and t_numeric_ok:
        assert result["group_results"]["T"]["all_terminal_safeguards"]
        assert result["group_results"]["T"]["zero_missed_instability_on_eligible_val"]
        assert result["scientific_verdict"] == "ACCEPT"
    else:
        assert not result["group_results"]["T"]["all_terminal_safeguards"]
        assert result["scientific_verdict"] == "REJECT"


def _authorization_fixture(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs/protocols").mkdir(parents=True)
    (tmp_path / "docs/research").mkdir(parents=True)
    for relative in ("configs/research_policy.json", "configs/project_manifest.json"):
        shutil.copyfile(repository / relative, tmp_path / relative)
    config = tomllib.loads((repository / "configs/experiments/paper_kiki_a07_tilted.toml")
                           .read_text(encoding="utf-8"))
    for key in ("development_protocol_path", "design_path"):
        shutil.copyfile(repository / config[key], tmp_path / config[key])
    manifest = json.loads((tmp_path / "configs/project_manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiments"]["paper_kiki_a07_tilted"]
    return tmp_path, experiment, config


def test_authorization_accepts_registered_config_and_rejects_tampering(tmp_path):
    root, experiment, config = _authorization_fixture(tmp_path)
    assert tilted.authorization(root, experiment, config)["status"] == "user_approved"
    changed = dict(config, risk_temperatures=[1.0, 1.0, 1.0])
    with pytest.raises(PermissionError, match="frozen source changed"):
        tilted.authorization(root, experiment, changed)


def test_authorization_rejects_consumed_status(tmp_path):
    root, experiment, config = _authorization_fixture(tmp_path)
    policy_path = root / "configs/research_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["development_evaluations"]["kiki_a07_development_20261024"]["status"] = "consumed"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not bind"):
        tilted.authorization(root, experiment, config)


def test_objective_budget_claims_each_group_seed_once_and_counts_all_slots(tmp_path, monkeypatch):
    shared = tmp_path / ".git-common"
    shared.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    (results / "run-1").mkdir()
    monkeypatch.setattr(tilted, "shared_budget_directory", lambda _root: shared)
    ledger = tilted.ObjectiveBudget(tmp_path, {"fixed": True}, "run-1", results / "run-1")
    for group in ("M", "T"):
        for seed in SEEDS:
            ledger.claim(group, seed)
    with pytest.raises(FileExistsError):
        ledger.claim("M", 324)
    with pytest.raises(ValueError, match="nonsequential"):
        ledger.progress("M", 324, 2)
    ledger.progress("M", 324, 1)
    with pytest.raises(FileExistsError):
        tilted.ObjectiveBudget(tmp_path, {"fixed": True}, "run-2", results / "run-1")
    snapshot = ledger.snapshot()
    assert snapshot["fit_attempts_reserved"] == 6
    assert snapshot["optimizer_steps_reserved"] == 1440
    assert [(row["group"], row["seed"]) for row in snapshot["slots"]] == [
        (group, seed) for group in ("M", "T") for seed in SEEDS]
