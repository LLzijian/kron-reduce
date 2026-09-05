"""P2 baseline-contract registration and historical inventory audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kron_reduce.data.p2_roles import build_p2_role_rows
from kron_reduce.models.p2_baselines import BASELINE_CLASSES, P2_BASELINE_NAMES
from project_harness.audit import audit_project
from project_harness.data_check import verify_data
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines

WORKFLOW = "ieee39_modular_energy_p2_a01_v1"
CONFIG_NAME = "paper_ieee39_modular_energy_p2_a01_f1_seed324.toml"
PROTOCOL_PATH = "docs/protocols/ieee39_modular_energy_p2_a01_baseline_contract_v1_zh.md"
INVENTORY_PATH = (
    "paper/planning/ieee39_modular_energy_equivalent/qa_logs/p2_a01_baseline_inventory.json"
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
    if (
        config.get("statistics", {}).get("pooled_r2_centering")
        != "within_event_within_node_over_time"
    ):
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
            verify_artifact_directory(directory, expected_manifest_sha256=digest, strict=True)
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


def _lineage(*, root: Path, result_dir: Path, run_id: str, stage: str, config_path: Path) -> None:
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


def _run_plan(root: Path, config_path: Path, result_dir: Path, run_id: str) -> None:
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


def _run_inventory_audit(root: Path, config_path: Path, result_dir: Path, run_id: str) -> None:
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


def _parse_junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError("P2 JUnit report contains no test suites")
    fields = ("tests", "failures", "errors", "skipped")
    try:
        values = {field: sum(int(suite.attrib[field]) for suite in suites) for field in fields}
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("P2 JUnit counters are missing or invalid") from error
    if values["tests"] < 1 or values["failures"] or values["errors"]:
        raise RuntimeError(f"P2 contract tests did not pass: {values}")
    return values


def _scratch_directory(root: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("unsafe P2 run ID")
    import hashlib

    token = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    directory = root / "scratch" / "one_off" / f"p2-contract-gate-{token}"
    if not directory.resolve().is_relative_to((root / "scratch" / "one_off").resolve()):
        raise ValueError("P2 scratch directory escaped its registered root")
    return directory


def _run_contract_tests(root: Path, result_dir: Path, run_id: str) -> dict[str, int]:
    scratch = _scratch_directory(root, run_id)
    scratch.mkdir(parents=True, exist_ok=False)
    junit = result_dir / "p2_contract_pytest.xml"
    test_files = [
        "tests/test_p2_baseline_contract.py",
        "tests/test_p2_roles.py",
        "tests/test_p2_training.py",
        "tests/test_p2_metrics.py",
        "tests/test_p2_baseline_models.py",
    ]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *test_files,
        "-q",
        f"--junitxml={junit}",
        f"--basetemp={scratch / 'pytest_tmp'}",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (result_dir / "pytest_stdout.txt").write_text(completed.stdout, encoding="utf-8", newline="\n")
    (result_dir / "pytest_stderr.txt").write_text(completed.stderr, encoding="utf-8", newline="\n")
    _write_json(
        result_dir / "pytest_execution.json",
        {
            "logical_command": [
                "python",
                "-m",
                "pytest",
                *test_files,
                "-q",
                "--junitxml=<result>/p2_contract_pytest.xml",
                "--basetemp=<scratch>/pytest_tmp",
            ],
            "returncode": completed.returncode,
            "scratch_directory": str(scratch.resolve()),
        },
    )
    if completed.returncode != 0 or not junit.is_file():
        raise RuntimeError(f"P2 contract pytest failed with return code {completed.returncode}")
    return _parse_junit(junit)


def _run_contract_unit_gate(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    tests = _run_contract_tests(root, result_dir, run_id)
    project_audit = audit_project(root)
    data_check = verify_data(root)
    dirty = status_lines(root)
    if project_audit.get("status") != "PASS":
        raise RuntimeError("P2 project audit failed")
    if data_check.get("status") != "PASS":
        raise RuntimeError("P2 data check failed")
    if dirty:
        raise RuntimeError(f"P2 contract unit gate requires a clean tree: {dirty}")
    if tuple(P2_BASELINE_NAMES) != EXPECTED_BASELINES or set(BASELINE_CLASSES) != set(
        EXPECTED_BASELINES
    ):
        raise RuntimeError("P2 baseline registry differs from the frozen contract")

    dataset_dir = root / "data" / str(config["dataset"])
    rows, role_audit = build_p2_role_rows(
        dataset_dir,
        fold=str(config["fold"]),
        seed=int(config["seed"]),
        minimum_steps=int(config["roles"]["minimum_required_steps"]),
    )
    access_audits = {
        method: BASELINE_CLASSES[method]().access_audit() for method in EXPECTED_BASELINES
    }
    forbidden = [
        method
        for method, audit in access_audits.items()
        if audit["observed_pm_reads"]
        or audit["future_pm_reads"]
        or audit["forbidden_field_reads"]
        or audit["source_routing"]
    ]
    if forbidden:
        raise RuntimeError(f"P2 baseline access contract failed: {forbidden}")
    _write_json(result_dir / "project_audit.json", project_audit)
    _write_json(result_dir / "data_check.json", data_check)
    _write_json(
        result_dir / "git_check.json",
        {"status": "PASS", "dirty": dirty, "git_commit": git_head(root)},
    )
    _write_json(result_dir / "role_contract_audit.json", role_audit)
    _write_json(result_dir / "baseline_access_contracts.json", access_audits)
    _write_json(
        result_dir / "contract_unit_gate.json",
        {
            "schema": "kron_reduce.p2_a01_contract_unit_gate.v1",
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": "contract_unit_gate",
            "baseline_methods": list(EXPECTED_BASELINES),
            "baseline_family_count": len(EXPECTED_BASELINES),
            "unit_tests": tests,
            "adversarial_test_count": tests["tests"],
            "role_event_count": len(rows),
            "role_manifest_sha256": role_audit["role_manifest_sha256"],
            "raw_ieee39_trajectory_data_read": False,
            "test_ids_read": [],
            "other_folds_read": [],
            "observed_pm_primary_reads": [],
            "interface_status": "PASS",
            "role_contract_status": "PASS",
            "causality_status": "PASS",
            "evaluation_status": "PASS",
            "scientific_verdict": None,
            "promotion_decision": "KEEP_BASELINE",
            "claim_boundary": (
                "Software, causal interfaces, synthetic ten-machine attacks, role allocation, "
                "and evaluation primitives only; no IEEE39 baseline performance or best-model claim."
            ),
        },
    )
    _lineage(
        root=root,
        result_dir=result_dir,
        run_id=run_id,
        stage="contract_unit_gate",
        config_path=config_path,
    )
    _write_json(
        result_dir / "stage_evidence.json",
        {
            "process_status": "PASS",
            "interface_status": "PASS",
            "role_contract_status": "PASS",
            "causality_status": "PASS",
            "evaluation_status": "PASS",
            "adversarial_test_count": tests["tests"],
            "claim_boundary": (
                "P2 unit and synthetic contract evidence only; no IEEE39 performance, "
                "selection, promotion, modal, stability, or Level 6 claim."
            ),
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
    elif stage == "contract_unit_gate":
        _run_contract_unit_gate(root, config_path, config, result_dir, run_id)
    else:
        raise ValueError(f"P2-A01 stage not implemented yet: {stage}")
