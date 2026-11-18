import json
from pathlib import Path

import pytest

from project_harness.evidence import sha256_file
from project_harness.policy import validate_run_policy

POLICY = json.loads((Path(__file__).resolve().parents[1] / "configs/research_policy.json").read_text(encoding="utf-8"))


@pytest.fixture
def policy_root(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for name in ("research_policy.json", "input_contracts.json"):
        (tmp_path / "configs" / name).write_bytes((repository / "configs" / name).read_bytes())
    return tmp_path


def candidate(**updates):
    return {"dataset": "kiki_ieee39_v1", "spatial_node_count": 10,
            "spatial_clustering": False, "input_contract_id": "measured_history_100ms_zoh_v1",
            "data_policy": "kiki_official_train_test_v1", "training_split": "train",
            "test_population": "task3_campaign2_final_test", "internal_resplit": False,
            "execution_mode": "bounded_kiki_official_cli_only",
            "split_reference_sha256": POLICY["official_split_reference_sha256"],
            "reads_test_performance": False, **updates}


def test_new_kiki_experiment_is_admitted(policy_root):
    validate_run_policy(policy_root, "next", candidate())


def test_official_train_test_contract_admits_training_on_official_train(policy_root):
    validate_run_policy(policy_root, "next", candidate())


def test_official_policy_rejects_internal_resplit(policy_root):
    with pytest.raises(RuntimeError, match="OFFICIAL_SPLIT_POLICY"):
        validate_run_policy(policy_root, "next", candidate(internal_resplit=True))


@pytest.mark.parametrize("updates,reason", [
    ({"dataset": "ieee39_postclear_canonical_v1"}, "DATASET_POLICY"),
    ({"dataset": None}, "DATASET_POLICY"),
    ({"spatial_node_count": 7}, "SPATIAL_POLICY"),
    ({"spatial_clustering": True}, "SPATIAL_POLICY"),
    ({"input_contract_id": "unknown"}, "INPUT_CONTRACT"),
    ({"reads_test_performance": True}, "TEST_EXPOSED"),
])
def test_forbidden_prospective_runs_fail_closed(policy_root, updates, reason):
    with pytest.raises(RuntimeError, match=reason):
        validate_run_policy(policy_root, "next", candidate(**updates))


def _test_experiment(policy_root, *, authorization="test-once", population="task3_campaign2_final_test"):
    manifest = policy_root / "models/frozen-model-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"model":"frozen"}\n', encoding="utf-8")
    experiment = candidate(reads_test_performance=True, test_authorization=authorization,
                           test_population=population)
    policy_path = policy_root / "configs/research_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["official_test_evaluations"][authorization] = {
        "status": "active", "experiment": "next", "fit_attempts": 0,
        "test_population": experiment["test_population"],
        "split_reference_sha256": policy["official_split_reference_sha256"],
        "max_executions": 1, "retry_allowed": False,
        "frozen_model_manifest": {"path": "models/frozen-model-manifest.json",
                                  "sha256": sha256_file(manifest)},
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return experiment, policy


def test_official_test_allows_active_one_use_scope_with_frozen_model(policy_root):
    experiment, _ = _test_experiment(policy_root)
    validate_run_policy(policy_root, "next", experiment)


def test_task3_is_the_primary_population_and_official_105_is_rejected(policy_root):
    experiment, _ = _test_experiment(
        policy_root, authorization="task3-once", population="task3_campaign2_final_test",
    )
    validate_run_policy(policy_root, "next", experiment)

    wrong_population, _ = _test_experiment(
        policy_root, authorization="different-roster", population="official_kiki_test",
    )
    with pytest.raises(RuntimeError, match="OFFICIAL_SPLIT_POLICY"):
        validate_run_policy(policy_root, "next", wrong_population)


@pytest.mark.parametrize("field,value", [
    ("fit_attempts", False),
    ("max_executions", True),
])
def test_official_test_rejects_boolean_values_for_integer_scope_fields(policy_root, field, value):
    experiment, policy = _test_experiment(policy_root)
    policy["official_test_evaluations"][experiment["test_authorization"]][field] = value
    (policy_root / "configs/research_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="TEST_EXPOSED"):
        validate_run_policy(policy_root, "next", experiment)


@pytest.mark.parametrize("case,reason", [
    ("consumed", "TEST_EXPOSED"),
    ("changed_freeze", "TEST_MODEL_FREEZE"),
])
def test_official_test_rejects_consumed_scope_or_changed_model_freeze(policy_root, case, reason):
    experiment, policy = _test_experiment(policy_root)
    authorization = policy["official_test_evaluations"][experiment["test_authorization"]]
    if case == "consumed":
        authorization["status"] = "consumed"
    else:
        (policy_root / authorization["frozen_model_manifest"]["path"]).write_text(
            '{"model":"changed"}\n', encoding="utf-8",
        )
    (policy_root / "configs/research_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(RuntimeError, match=reason):
        validate_run_policy(policy_root, "next", experiment)


def test_legacy_real_experiment_hash_stays_frozen(policy_root):
    repository = Path(__file__).resolve().parents[1]
    name = "paper_kiki_p2_a03_seed324"
    experiment = json.loads((repository / "configs/project_manifest.json").read_text(encoding="utf-8"))[
        "experiments"
    ][name]
    validate_run_policy(policy_root, name, experiment)
    with pytest.raises(RuntimeError, match="LEGACY_EXPERIMENT_CHANGED"):
        validate_run_policy(policy_root, name, {**experiment, "input_contract_id": "measured_history_100ms_zoh_v1"})


def test_generic_runner_blocks_official_cli_before_policy_and_data(tmp_path, monkeypatch):
    from project_harness import runner

    touched = []
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        "next": {"execution_mode": "bounded_kiki_official_cli_only"},
    }})
    monkeypatch.setattr(runner, "validate_run_policy", lambda *_args, **_kwargs: touched.append("policy"))
    monkeypatch.setattr(runner, "verify_data", lambda *_args, **_kwargs: touched.append("data"))
    with pytest.raises(RuntimeError, match="BOUNDED_KIKI_OFFICIAL_CLI_ONLY"):
        runner.run_experiment(tmp_path, "next", "train")
    assert touched == []


def test_synthetic_correctness_tests_remain_available(policy_root):
    validate_run_policy(policy_root, "synthetic", {"dataset": "synthetic_benchmark_registry_v1"})


def test_historical_replay_requires_explicit_mode_and_frozen_contract(policy_root):
    config = policy_root / "configs/old.toml"
    config.write_text("seed = 324\n")
    registry = {"old": {"dataset": "ieee39_postclear_canonical_v1",
                        "bindings": {"configs/old.toml": sha256_file(config)}}}
    (policy_root / "configs/historical_registry_20261023.json").write_text(json.dumps(registry))
    experiment = {"dataset": "ieee39_postclear_canonical_v1",
                  "execution_mode": "historical_reproduction_only"}
    with pytest.raises(RuntimeError, match="HISTORICAL_ONLY"):
        validate_run_policy(policy_root, "old", experiment)
    validate_run_policy(policy_root, "old", experiment, historical_reproduction=True)
    config.write_text("seed = 325\n")
    with pytest.raises(RuntimeError, match="HISTORICAL_CONTRACT_CHANGED"):
        validate_run_policy(policy_root, "old", experiment, historical_reproduction=True)


def test_replay_flag_cannot_bypass_policy(policy_root):
    with pytest.raises(RuntimeError, match="HISTORICAL_IDENTITY"):
        validate_run_policy(policy_root, "next", candidate(), historical_reproduction=True)


def test_registered_historical_replay_contracts_remain_intact(root):
    manifest = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    for name, experiment in manifest["experiments"].items():
        if experiment.get("execution_mode") == "historical_reproduction_only":
            validate_run_policy(root, name, experiment, historical_reproduction=True)
