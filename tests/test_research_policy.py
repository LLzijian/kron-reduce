import json
from pathlib import Path

import pytest

from project_harness.evidence import sha256_file
from project_harness.policy import validate_run_policy


@pytest.fixture
def policy_root(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for name in ("research_policy.json", "input_contracts.json"):
        (tmp_path / "configs" / name).write_bytes((repository / "configs" / name).read_bytes())
    return tmp_path


def candidate(**updates):
    return {"dataset": "kiki_ieee39_v1", "spatial_node_count": 10,
            "spatial_clustering": False, "input_contract_id": "measured_history_200ms_zoh_v1",
            "reads_test_performance": False, **updates}


def test_new_kiki_experiment_is_admitted(policy_root):
    validate_run_policy(policy_root, "next", candidate())


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
