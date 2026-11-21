"""Consumed test registrations stay auditable without reopening execution."""

import json

import pytest

from project_harness.evidence import sha256_file
from project_harness.policy import audit_research_policy, validate_run_policy


@pytest.fixture
def registered(tmp_path, monkeypatch):
    from project_harness import data_check

    monkeypatch.setattr(data_check, "_verify_checksum_manifest", lambda *a, **kw: [{"pass": True}])
    configs = tmp_path / "configs"
    configs.mkdir()
    frozen = tmp_path / "frozen.json"
    frozen.write_text('{"checkpoint":"fixed"}', encoding="utf-8")
    contract = "measured_history_100ms_zoh_v1"
    (configs / "input_contracts.json").write_text(json.dumps({contract: {}}), encoding="utf-8")
    experiment = {
        "dataset": "kiki_ieee39_v1", "spatial_node_count": 10, "spatial_clustering": False,
        "input_contract_id": contract, "data_policy": "kiki_official_train_test_v1",
        "training_split": "train", "internal_resplit": False,
        "execution_mode": "bounded_kiki_official_cli_only",
        "split_reference_sha256": "a" * 64, "reads_test_performance": True,
        "test_population": "task3_campaign2_final_test", "test_authorization": "one-use",
    }
    authorization = {
        "status": "consumed", "experiment": "next", "fit_attempts": 0,
        "max_executions": 1, "retry_allowed": False,
        "test_population": experiment["test_population"], "split_reference_sha256": "a" * 64,
        "frozen_model_manifest": {"path": "frozen.json", "sha256": sha256_file(frozen)},
    }
    policy = {
        "synthetic_datasets": [], "primary_real_dataset": "kiki_ieee39_v1",
        "input_contracts": "configs/input_contracts.json",
        "default_real_data_policy": "kiki_official_train_test_v1",
        "official_split_reference_sha256": "a" * 64,
        "official_test_evaluations": {"one-use": authorization},
    }
    manifest = {
        "default_dataset": "kiki_ieee39_v1",
        "datasets": {"kiki_ieee39_v1": {"path": "data/kiki", "checksum_manifest": "hashes.json"}},
        "experiments": {"next": experiment},
    }
    path = configs / "research_policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return tmp_path, policy, manifest


def test_consumed_test_registration_audits_without_reopening_execution(registered):
    root, _, manifest = registered
    path = root / "configs/research_policy.json"
    original = path.read_bytes()
    assert audit_research_policy(root, manifest) == []
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match="TEST_EXPOSED"):
        validate_run_policy(root, "next", manifest["experiments"]["next"])


def test_independent_confirmation_requires_seal_and_cannot_run_twice(registered):
    root, policy, manifest = registered
    name = "kiki_ieee39_blind_v2"
    cohort = root / "data" / name
    cohort.mkdir(parents=True)
    for filename in ("manifest.json", "checksums.sha256"):
        (cohort / filename).write_text("sealed metadata", encoding="utf-8")
    experiment = manifest["experiments"]["next"] | {
        "dataset": name, "stage": "blind", "execution_mode": "bounded_kiki_independent_cli_only",
        "blind_authorization": "independent",
    }
    auth = {"status": "active", "dataset": name, "experiment": "next", "allowed_roles": ["blind"],
            "max_executions": 1, "fit_attempts": 0, "gradients": 0, "optimizer_updates": 0,
            "test_cases": 0, "retry_allowed": False,
            "dataset_manifest_sha256": sha256_file(cohort / "manifest.json"),
            "dataset_checksums_sha256": sha256_file(cohort / "checksums.sha256")}
    config = root / "configs/research_policy.json"
    policy["independent_confirmations"] = {"independent": auth}
    config.write_text(json.dumps(policy), encoding="utf-8")
    validate_run_policy(root, "next", experiment)
    auth["status"] = "consumed"
    config.write_text(json.dumps(policy), encoding="utf-8")
    validate_run_policy(root, "next", experiment, audit_only=True)
    with pytest.raises(RuntimeError, match="INDEPENDENT_CONFIRMATION"):
        validate_run_policy(root, "next", experiment)
    (cohort / "checksums.sha256").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed checksums.sha256"):
        validate_run_policy(root, "next", experiment, audit_only=True)


@pytest.mark.parametrize("field,value", [
    ("status", "paused"), ("status", "rejected"), ("experiment", "other"),
    ("fit_attempts", 1), ("max_executions", 2), ("retry_allowed", True),
    ("test_population", "other"), ("split_reference_sha256", "b" * 64),
])
def test_consumed_audit_still_checks_exact_scope(registered, field, value):
    root, policy, manifest = registered
    policy["official_test_evaluations"]["one-use"][field] = value
    (root / "configs/research_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    findings = audit_research_policy(root, manifest)
    assert len(findings) == 1 and "TEST_EXPOSED" in findings[0]["message"]


def test_consumed_audit_still_checks_frozen_model_hash(registered):
    root, _, manifest = registered
    (root / "frozen.json").write_text('{"checkpoint":"changed"}', encoding="utf-8")
    findings = audit_research_policy(root, manifest)
    assert len(findings) == 1 and "TEST_MODEL_FREEZE" in findings[0]["message"]


@pytest.mark.parametrize("damage", [None, "receipt", "content", "path", "active", "existing_changed"])
def test_consumed_missing_local_artifact_uses_exact_committed_metadata(registered, damage):
    root, policy, manifest = registered
    auth = policy["official_test_evaluations"]["one-use"]
    frozen = root / "frozen.json"
    snapshot = {**auth["frozen_model_manifest"], "content_utf8": frozen.read_bytes().decode("utf-8")}
    if damage == "content":
        snapshot["content_utf8"] = "changed"
    if damage == "path":
        snapshot["path"] = "other.json"
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({"frozen_model_manifest": snapshot}), encoding="utf-8")
    auth.update(result_receipt_path="receipt.json", result_receipt_sha256=sha256_file(receipt))
    frozen.unlink()
    if damage == "receipt":
        receipt.write_text("{}", encoding="utf-8")
    if damage == "active":
        auth["status"] = "active"
    if damage == "existing_changed":
        frozen.write_text("changed", encoding="utf-8")
    (root / "configs/research_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    findings = audit_research_policy(root, manifest)
    if damage is None:
        assert findings == []
    else:
        assert len(findings) == 1 and "TEST_MODEL_FREEZE" in findings[0]["message"]
    with pytest.raises(RuntimeError, match="TEST_EXPOSED|TEST_MODEL_FREEZE"):
        validate_run_policy(root, "next", manifest["experiments"]["next"])
