from __future__ import annotations

import copy
import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from kron_reduce.workflows.ieee39_modular_energy_p1_diagnosis import (
    CONTROLLED_ARMS,
    EXPECTED_BRANCH,
    EXPECTED_CONFIG,
    EXPECTED_CONFIG_A02,
    WORKFLOW_A02,
    _arm_registry,
    _candidate_inventory,
    _find_registered_plan,
    _json,
    _run_candidate,
    _run_plan,
    _sha256,
    _source_snapshot,
    _validate_config,
    _verify_hashes,
    _write_hashes,
    validate_p1_candidate_contract,
)


def test_frozen_config_rejects_wrong_branch_and_arm_list() -> None:
    with pytest.raises(RuntimeError, match="branch lock violation"):
        _validate_config(copy.deepcopy(EXPECTED_CONFIG), "codex/not-the-registered-branch")

    wrong_arms_config = copy.deepcopy(EXPECTED_CONFIG)
    wrong_arms_config["controlled_arms"]["names"] = list(reversed(CONTROLLED_ARMS))
    with pytest.raises(RuntimeError, match="controlled_arms.names"):
        _validate_config(wrong_arms_config, EXPECTED_BRANCH)

    _validate_config(copy.deepcopy(EXPECTED_CONFIG_A02), "codex/loop-p1-a02")
    with pytest.raises(RuntimeError, match="branch lock violation"):
        _validate_config(copy.deepcopy(EXPECTED_CONFIG_A02), EXPECTED_BRANCH)


def test_artifact_hash_round_trip_rejects_changed_file(tmp_path: Path) -> None:
    artifact = tmp_path / "registration.json"
    artifact.write_text('{"stage": "plan_registered"}\n', encoding="utf-8")
    _write_hashes(tmp_path)

    verified = _verify_hashes(tmp_path)
    assert verified["status"] == "PASS"
    assert verified["files"][0]["path"] == "registration.json"

    artifact.write_text('{"stage": "candidate_acquired"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen artifact hash verification failed"):
        _verify_hashes(tmp_path)


def test_source_snapshot_marks_untracked_critical_file_not_reproducible(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "p1-test@example.invalid"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "P1 test"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    tracked = repository / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial source"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    tracked.write_text("VALUE = 3\n", encoding="utf-8")
    (repository / "critical_untracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    snapshot = _source_snapshot(
        repository,
        "baseline-00000000",
        critical_files=("critical_untracked.py",),
    )

    assert snapshot["reproducible_from_clean_commit"] is False
    assert snapshot["git_status_paths"][0] == {"status": " M", "path": "tracked.py"}
    assert snapshot["critical_files"][0]["tracked_by_current_head"] is False
    assert snapshot["findings"] == [
        {"path": "critical_untracked.py", "finding": "not_tracked_by_current_head"}
    ]


@pytest.fixture
def candidate_inventory_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    instances = [
        {"instance_id": f"resonant-{index}", "family": "resonant_memory"}
        for index in range(3)
    ] + [
        {"instance_id": f"edge-{index}", "family": "edge_memory"}
        for index in range(2)
    ]
    validation_summary: dict[str, object] = {
        "active_blocks": ["resonant_memory", "edge_memory"],
        "active_block_instances": [row["instance_id"] for row in instances],
        "model_count": 1,
        "source_routing": False,
    }
    frozen_model: dict[str, object] = {
        "diagnostics": {
            "active_block_instances": instances,
            "block_parameter_norms": {"resonant_memory": 0.0, "edge_memory": 0.0},
            "instance_parameter_norms": {row["instance_id"]: 0.0 for row in instances},
        }
    }
    return validation_summary, frozen_model, {}


def test_candidate_inventory_counts_strictly_zero_memory_instances(
    candidate_inventory_inputs: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    inventory = _candidate_inventory(*candidate_inventory_inputs)

    zero_effect = inventory["zero_effect_by_parameter"]
    assert zero_effect["criterion"] == "parameter_norm_strictly_equal_to_zero"
    assert zero_effect["count"] == 5
    assert zero_effect["count_by_family"] == {"edge_memory": 2, "resonant_memory": 3}


def test_arm_four_is_conditional_only_and_not_promotion_eligible() -> None:
    arm_four = _arm_registry()["arms"][3]

    assert arm_four["conditional_only"] is True
    assert arm_four["eligible_for_promotion"] is False


def test_a02_saved_configuration_matches_strict_code_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "configs/experiments/paper_ieee39_modular_energy_p1_a02_f1_seed324.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, "codex/loop-p1-a02")


@pytest.mark.parametrize("attack", ["empty_manifest", "extra_file", "nested_manifest"])
def test_frozen_inventory_rejects_missing_or_unregistered_evidence(
    tmp_path: Path, attack: str
) -> None:
    _json(tmp_path / "evidence.json", {"fixture_only": True})
    _write_hashes(tmp_path)
    if attack == "empty_manifest":
        _json(tmp_path / "artifact_hashes.json", {})
    elif attack == "extra_file":
        _json(tmp_path / "unregistered.json", {})
    else:
        (tmp_path / "nested").mkdir()
        _json(tmp_path / "nested/artifact_hashes.json", {})
    with pytest.raises(RuntimeError, match="frozen artifact hash verification failed"):
        _verify_hashes(tmp_path)


@pytest.fixture
def a02_saved_candidate(tmp_path: Path, monkeypatch):
    """Metadata-only fake artifacts; never mistaken for a model/execution test."""
    import kron_reduce.workflows.ieee39_modular_energy_p1_diagnosis as diagnosis

    repo = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(EXPECTED_CONFIG_A02)
    config["baseline_run_id"] = "baseline-fixture"
    baseline = tmp_path / "results" / config["baseline_run_id"]
    baseline.mkdir(parents=True)
    validation = {
        "model_count": 1, "source_routing": False,
        "active_blocks": [], "active_block_instances": [],
        "test_ids_read": [], "test_trajectories_read": 0,
        "future_observed_Pm_used": False, "level_6_claimed": False,
        "terminal_state": "IDENTIFICATION_FAILED",
    }
    _json(baseline / "validation_summary.json", validation)
    _json(baseline / "lineage.json", validation)
    _json(baseline / "structure_selection_result.json", {"terminal_state": "IDENTIFICATION_FAILED"})
    _json(baseline / "role_audit.json", {"counts": {}})
    for name in ("frozen_model", "b2_baseline_model"):
        _json(baseline / f"{name}.json", {"diagnostics": {}})
        (baseline / f"{name}.npz").write_bytes(b"metadata-only test, not a loadable model")
    (baseline / "dataset_manifest.csv").write_text("event_id,role\nfixture,fit\n", encoding="utf-8")
    _write_hashes(baseline)
    config["baseline_artifact_manifest_sha256"] = _sha256(baseline / "artifact_hashes.json")
    retry_path = tmp_path / config["previous_retry_evidence"]
    retry_path.parent.mkdir(parents=True)
    _json(retry_path, {"scientific_verdict": "RETRY", "state_advanced": False})
    config["previous_retry_evidence_sha256"] = _sha256(retry_path)
    config_relative = "configs/experiments/paper_ieee39_modular_energy_p1_a02_f1_seed324.toml"
    config_path = tmp_path / config_relative
    config_path.parent.mkdir(parents=True)
    config_text = (repo / config_relative).read_text(encoding="utf-8")
    for key in (
        "baseline_run_id", "baseline_artifact_manifest_sha256", "previous_retry_evidence_sha256"
    ):
        config_text = config_text.replace(EXPECTED_CONFIG_A02[key], config[key])
    config_path.write_text(config_text, encoding="utf-8")
    plan = tmp_path / "results" / "plan-fixture"
    _run_plan(tmp_path, config_path, config, plan, plan.name)
    state_path = tmp_path / ".project_state" / f"{WORKFLOW_A02}_state.json"
    state_path.parent.mkdir()
    _json(state_path, {
        "workflow": WORKFLOW_A02, "current_stage": "plan_registered", "approvals": {},
        "history": [{"action": "advance", "from": "scaffold", "to": "plan_registered", "evidence": {
            "status": "PASS", "result_dir": plan.name,
            "artifact_manifest": {"run_id": plan.name, "path": "artifact_hashes.json", "sha256": _sha256(plan / "artifact_hashes.json")},
        }}],
    })
    monkeypatch.setattr(diagnosis, "_source_snapshot", lambda *_args: {
        "branch": config["branch"], "reproducible_from_clean_commit": False,
    })
    candidate = tmp_path / "results" / "candidate-fixture"
    _run_candidate(tmp_path, config_path, config, candidate, candidate.name)
    return tmp_path, config, config_path, plan, candidate


def test_candidate_metadata_gate_preserves_claim_boundary(a02_saved_candidate) -> None:
    root, config, _, _, candidate = a02_saved_candidate
    result = validate_p1_candidate_contract(
        root, candidate, config,
        expected_manifest_sha256=_sha256(candidate / "artifact_hashes.json"),
    )
    assert result["status"] == "PASS"
    assert result["scope"] == "saved_candidate_metadata_and_lineage_only"
    assert result["full_model_unit_gate_passed"] is False
    assert result["legacy_baseline_reproducibility_proven"] is False
    assert result["level_6_claimed"] is False


@pytest.mark.parametrize("file,field,value,reason", [
    ("candidate_structure_selection_explicit.json", "model_count", True, "integer model_count"),
    ("candidate_structure_selection_explicit.json", "source_routing", True, "source_routing=false"),
    ("candidate_structure_selection_explicit.json", "routing_keys", ["source"], "routing keys"),
    ("candidate_structure_selection_explicit.json", "terminal_state", "ACCEPT", "scientific field changed"),
    ("data_access_audit.json", "test_ids_read", ["test-event"], "test access evidence"),
    ("data_access_audit.json", "test_trajectories_read", False, "explicitly zero"),
    ("data_access_audit.json", "other_folds_read", ["F2"], "locked folds"),
    ("data_access_audit.json", "future_observed_pm_read", True, "future_observed_pm_read"),
    ("model_artifact_manifest.json", "level_6_claimed", True, "cannot claim Level 6"),
    ("lineage.json", "plan_registered_run", "unregistered", "plan lineage mismatch"),
    ("lineage.json", "labels_used", True, "input declaration mismatch"),
])
def test_candidate_contract_rejects_self_consistently_rehashed_metadata_attacks(
    a02_saved_candidate, file, field, value, reason
) -> None:
    root, config, _, _, candidate = a02_saved_candidate
    path = candidate / file
    record = json.loads(path.read_text(encoding="utf-8"))
    record[field] = value
    _json(path, record)
    _write_hashes(candidate)
    with pytest.raises(RuntimeError, match=reason):
        validate_p1_candidate_contract(
            root, candidate, config,
            expected_manifest_sha256=_sha256(candidate / "artifact_hashes.json"),
        )


def test_registered_plan_cannot_be_rewritten_with_new_local_hash(a02_saved_candidate) -> None:
    root, config, config_path, plan, _ = a02_saved_candidate
    registration = json.loads((plan / "registration.json").read_text(encoding="utf-8"))
    registration["promotion_decision"] = "PROMOTE"
    _json(plan / "registration.json", registration)
    _write_hashes(plan)
    with pytest.raises(RuntimeError, match="manifest SHA256 does not match"):
        _find_registered_plan(root, config_path, config)


def test_registered_plan_requires_nonempty_independent_hash(a02_saved_candidate) -> None:
    root, config, config_path, _, _ = a02_saved_candidate
    state_path = root / ".project_state" / f"{WORKFLOW_A02}_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["history"][0]["evidence"]["artifact_manifest"]["sha256"]
    _json(state_path, state)
    with pytest.raises(RuntimeError, match="independent manifest anchor"):
        _find_registered_plan(root, config_path, config)
