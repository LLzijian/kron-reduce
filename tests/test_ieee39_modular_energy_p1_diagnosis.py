from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from kron_reduce.workflows.ieee39_modular_energy_p1_diagnosis import (
    CONTROLLED_ARMS,
    EXPECTED_BRANCH,
    EXPECTED_CONFIG,
    _arm_registry,
    _candidate_inventory,
    _source_snapshot,
    _validate_config,
    _verify_hashes,
    _write_hashes,
)


def test_frozen_config_rejects_wrong_branch_and_arm_list() -> None:
    with pytest.raises(RuntimeError, match="branch lock violation"):
        _validate_config(copy.deepcopy(EXPECTED_CONFIG), "codex/not-the-registered-branch")

    wrong_arms_config = copy.deepcopy(EXPECTED_CONFIG)
    wrong_arms_config["controlled_arms"]["names"] = list(reversed(CONTROLLED_ARMS))
    with pytest.raises(RuntimeError, match="controlled_arms.names"):
        _validate_config(wrong_arms_config, EXPECTED_BRANCH)


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
    (repository / "critical_untracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    snapshot = _source_snapshot(
        repository,
        "baseline-00000000",
        critical_files=("critical_untracked.py",),
    )

    assert snapshot["reproducible_from_clean_commit"] is False
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
