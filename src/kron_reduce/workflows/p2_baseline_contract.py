"""P2 baseline-contract registration and historical inventory audit."""

from __future__ import annotations

import json
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from project_harness.evidence import sha256_file, verify_artifact_directory

WORKFLOW = "ieee39_modular_energy_p2_a01_v1"
CONFIG_NAME = "paper_ieee39_modular_energy_p2_a01_f1_seed324.toml"
PROTOCOL_PATH = "docs/protocols/ieee39_modular_energy_p2_a01_baseline_contract_v1_zh.md"
INVENTORY_PATH = (
    "paper/planning/ieee39_modular_energy_equivalent/qa_logs/"
    "p2_a01_baseline_inventory.json"
)
EXPECTED_BASELINES = (
    "B2-10-ZOH",
    "EGDF-10-ZOH",
    "GMDPM-10-CAUSAL",
    "SSDM-10-ZOH",
    "LIN-ODE-10",
    "SINDYc-ODE-10",
    "NODE-MLP-10",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _seal(result_dir: Path) -> None:
    _write_json(
        result_dir / "artifact_hashes.json",
        {
            path.relative_to(result_dir).as_posix(): sha256_file(path)
            for path in sorted(result_dir.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )
    verify_artifact_directory(result_dir, strict=True)


def _load_config(root: Path, experiment: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    if experiment.get("config") != CONFIG_NAME:
        raise ValueError("P2-A01 requires its registered config")
    path = root / "configs" / "experiments" / CONFIG_NAME
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    expected = {
        "task_type": "baseline_contract_and_controlled_replay",
        "protocol_revision": "ieee39_modular_energy_p2_a01_baseline_contract_v1",
        "dataset": "ieee39_postclear_canonical_v1",
        "fold": "F1",
        "seed": 324,
        "branch": "codex/loop-p2-a01",
        "representation": "generator_10_mass_weighted_COI",
        "model_count_per_method": 1,
        "source_routing": False,
        "test_access": "forbidden",
        "canonical_validation_access": "forbidden",
        "other_folds_access": "forbidden",
        "stability_label_access": "forbidden",
        "historical_metric_reuse": "forbidden",
        "baseline_methods": list(EXPECTED_BASELINES),
    }
    for key, value in expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(f"P2-A01 frozen config mismatch: {key}")
    roles = config.get("roles", {})
    if roles.get("highorder_line_counts") != [10, 3, 3]:
        raise ValueError("P2-A01 high-order role counts changed")
    if roles.get("near_critical_line_counts") != [3, 1, 3]:
        raise ValueError("P2-A01 near-critical role counts changed")
    if roles.get("minimum_internal_lines_per_source") != 3:
        raise ValueError("P2-A01 minimum independent-line gate changed")
    if config.get("input_contract", {}).get("conditional_sensitivity_is_upper_bound") is not False:
        raise ValueError("P2-A01 observed-Pm sensitivity cannot be called an upper bound")
    if config.get("statistics", {}).get("pooled_r2_centering") != "within_event_within_node_over_time":
        raise ValueError("P2-A01 pooled R2 contract changed")
    return path, config


def _require_branch(root: Path, config: Mapping[str, Any]) -> None:
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    ).stdout.strip()
    if branch != config["branch"]:
        raise RuntimeError("P2-A01 execution branch differs from registration")


def _inventory(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / INVENTORY_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "kron_reduce.p2_a01_baseline_inventory.v1":
        raise ValueError("P2-A01 inventory schema mismatch")
    if value.get("status") != "COMPLETE_WITH_REBUILD_REQUIRED":
        raise ValueError("P2-A01 inventory status mismatch")
    findings = value.get("findings")
    if not isinstance(findings, list) or len(findings) != 5:
        raise ValueError("P2-A01 inventory must contain five historical families")
    return path, value


def _verify_historical_evidence(root: Path, inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for finding in inventory["findings"]:
        reference = str(finding["historical_evidence"])
        if "artifact_manifest_sha256" in finding:
            directory = root / "results" / reference
            digest = str(finding["artifact_manifest_sha256"])
            verify_artifact_directory(
                directory, expected_manifest_sha256=digest, strict=True
            )
            checks.append(
                {
                    "method_family": finding["method_family"],
                    "reference": reference,
                    "kind": "immutable_run",
                    "sha256": digest,
                    "status": "PASS",
                }
            )
        else:
            path = root / reference
            digest = sha256_file(path)
            if digest != finding.get("evidence_sha256"):
                raise RuntimeError(f"P2-A01 historical evidence changed: {reference}")
            checks.append(
                {
                    "method_family": finding["method_family"],
                    "reference": reference,
                    "kind": "versioned_file",
                    "sha256": digest,
                    "status": "PASS",
                }
            )
    return checks


def _lineage(
    *, root: Path, result_dir: Path, run_id: str, stage: str, config_path: Path
) -> None:
    _write_json(
        result_dir / "lineage.json",
        {
            "schema": "kron_reduce.p2_a01_lineage.v1",
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": stage,
            "protocol_path": PROTOCOL_PATH,
            "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": sha256_file(config_path),
            "inventory_path": INVENTORY_PATH,
            "inventory_sha256": sha256_file(root / INVENTORY_PATH),
        },
    )


def _run_plan(
    root: Path, config_path: Path, result_dir: Path, run_id: str
) -> None:
    inventory_path, inventory = _inventory(root)
    record = {
        "schema": "kron_reduce.p2_a01_plan_registration.v1",
        "run_id": run_id,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "workflow_path": f"configs/workflows/{WORKFLOW}.json",
        "workflow_sha256": sha256_file(root / "configs" / "workflows" / f"{WORKFLOW}.json"),
        "inventory_path": inventory_path.relative_to(root).as_posix(),
        "inventory_sha256": sha256_file(inventory_path),
        "baseline_methods": list(EXPECTED_BASELINES),
        "historical_numbers_reuse_forbidden": True,
        "test_locked": True,
        "other_folds_locked": True,
        "inventory_scientific_verdict": inventory["scientific_verdict"],
    }
    _write_json(result_dir / "plan_registration.json", record)
    _lineage(
        root=root,
        result_dir=result_dir,
        run_id=run_id,
        stage="plan_registered",
        config_path=config_path,
    )
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "protocol_path": PROTOCOL_PATH,
            "config_path": config_path.relative_to(root).as_posix(),
            "historical_numbers_reuse_forbidden": True,
            "test_locked": True,
            "other_folds_locked": True,
            "claim_boundary": "P2 comparison design only; no baseline performance or best-model claim.",
        },
    )
    _seal(result_dir)


def _run_inventory_audit(
    root: Path, config_path: Path, result_dir: Path, run_id: str
) -> None:
    _, inventory = _inventory(root)
    checks = _verify_historical_evidence(root, inventory)
    audit = {
        "schema": "kron_reduce.p2_a01_inventory_audit.v1",
        "run_id": run_id,
        "historical_evidence_checks": checks,
        "baseline_family_count": len(EXPECTED_BASELINES),
        "historical_family_count": len(checks),
        "rebuild_required_count": len(EXPECTED_BASELINES),
        "role_repartition_required": True,
        "historical_numbers_copied": False,
        "test_ids_read": [],
        "other_folds_read": [],
        "trajectory_channels_read": [],
        "labels_read": False,
        "status": "PASS",
    }
    _write_json(result_dir / "baseline_inventory_audit.json", audit)
    _lineage(
        root=root,
        result_dir=result_dir,
        run_id=run_id,
        stage="baseline_inventory_audited",
        config_path=config_path,
    )
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "inventory_path": INVENTORY_PATH,
            "baseline_family_count": len(EXPECTED_BASELINES),
            "rebuild_required_count": len(EXPECTED_BASELINES),
            "role_repartition_required": True,
            "claim_boundary": "Historical evidence establishes incompatibility and rebuild requirements only; no comparative performance claim.",
        },
    )
    _seal(result_dir)


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    config_path, config = _load_config(root, experiment)
    _require_branch(root, config)
    if stage == "plan_registered":
        _run_plan(root, config_path, result_dir, run_id)
    elif stage == "baseline_inventory_audited":
        _run_inventory_audit(root, config_path, result_dir, run_id)
    else:
        raise ValueError(f"P2-A01 stage not implemented yet: {stage}")
