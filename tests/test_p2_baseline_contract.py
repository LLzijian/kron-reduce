from __future__ import annotations

import json
from pathlib import Path

from kron_reduce.workflows import p2_baseline_contract as p2
from project_harness.report_storage import locate_report


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_p2_config_freezes_seven_ten_machine_causal_baselines() -> None:
    root = _root()
    _, config = p2._load_config(root, {"config": p2.CONFIG_NAME})
    assert tuple(config["baseline_methods"]) == p2.EXPECTED_BASELINES
    assert config["representation"] == "generator_10_mass_weighted_COI"
    assert config["model_count_per_method"] == 1
    assert config["source_routing"] is False
    assert config["input_contract"]["primary_pm_policy"] == "Pm_clear_ZOH"
    assert config["input_contract"]["conditional_sensitivity_is_upper_bound"] is False
    assert config["test_access"] == "forbidden"
    assert config["canonical_validation_access"] == "forbidden"
    assert config["other_folds_access"] == "forbidden"


def test_p2_roles_restore_minimum_source_level_line_uncertainty() -> None:
    root = _root()
    _, config = p2._load_config(root, {"config": p2.CONFIG_NAME})
    roles = config["roles"]
    assert roles["highorder_line_counts"] == [10, 3, 3]
    assert roles["near_critical_line_counts"] == [3, 1, 3]
    assert roles["minimum_internal_lines_per_source"] == 3
    assert roles["role_assignment_reads_trajectories"] is False
    assert roles["role_assignment_reads_labels"] is False


def test_p2_inventory_binds_real_historical_evidence_without_trajectory_read() -> None:
    root = _root()
    _, inventory = p2._inventory(root)
    checks = p2._verify_historical_evidence(root, inventory)
    assert len(checks) == 5
    assert all(row["status"] == "PASS" for row in checks)
    assert inventory["scientific_verdict"] == "REBUILD_ALL_MAIN_TABLE_NUMBERS"
    assert inventory["claim_boundary"].startswith("Historical artifacts establish")


def test_p2_workflow_branches_only_after_scientific_acceptance() -> None:
    root = _root()
    workflow = json.loads(
        (root / "configs/workflows/ieee39_modular_energy_p2_a01_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["approval_required"] == ["merge_gate"]
    assert workflow["transitions"]["acceptance"] == {
        "ACCEPT": "merge_gate",
        "REJECT": "rollback_report",
        "RETRY": "rollback_report",
    }
    assert workflow["terminal_stages"] == ["report"]
    locked = set(workflow["locked_claims"])
    assert "observed_pm_as_primary_or_deployment_input" in locked
    assert "best_model_claim_from_P2" in locked


def test_p2_main_table_excludes_native_solver_mismatch() -> None:
    root = _root()
    protocol_path = locate_report(root, p2.PROTOCOL_PATH)
    protocol = (root / protocol_path).read_text(encoding="utf-8")
    assert "原生离散 LSTM" in protocol
    assert "不能进入同积分器主表" in protocol
    assert "LIN-ODE-10" in protocol
    assert "SINDYc-ODE-10" in protocol
    assert "NODE-MLP-10" in protocol
