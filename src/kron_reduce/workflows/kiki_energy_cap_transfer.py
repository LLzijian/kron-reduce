"""Dedicated one-use fixed-model train comparison; no generic source-data access."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from kron_reduce.workflows.constant_c_synthetic import (
    _read_json,
    _verify_loaded_runtime,
    _write_new_json,
)
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

AUTH = "kiki_energy_cap_transfer_v1_20261109"
EXPERIMENT = WORKFLOW = "kiki_energy_cap_transfer_v1"
STAGE = "train_transfer"
MODULE = "kron_reduce.workflows.kiki_energy_cap_transfer"
DATASET = "kiki_ieee39_v1"
DATA_POLICY = "frozen_saved_train_only"
MODE = "bounded_saved_train_cli_only"
PLAN = "configs/proposals/kiki_energy_cap_transfer_v1.json"
SOURCES = "configs/proposals/kiki_energy_cap_saved_sources_v1.json"
PROTOCOL = "docs/protocols/kiki_energy_cap_transfer_v1_zh.md"
CONFIG = "configs/experiments/kiki_energy_cap_transfer_v1.toml"
SCOPE = {
    "authorization": AUTH, "experiment": EXPERIMENT, "workflow": WORKFLOW,
    "stage": STAGE, "module": MODULE, "dataset": DATASET, "data_policy": DATA_POLICY,
    "max_executions": 1, "retry_allowed": False, "fit_attempts": 0, "optimizer_updates": 0,
    "allocated_events": 268, "eligible_events": 258, "seed_event_slots": 774,
    "source_payload_reads": 0, "internal_val_reads": 0, "source_val_reads": 0,
    "test_reads": 0, "label_reads": 0,
}
REQUIRED_BINDINGS = frozenset({
    CONFIG, PROTOCOL, SOURCES, "configs/project_manifest.json",
    "configs/workflows/kiki_energy_cap_transfer_v1.json", "requirements/constraints-py312.txt",
    "src/kron_reduce/data/energy_cap_saved_inputs.py",
    "src/kron_reduce/evaluation/encoder_energy_cap.py",
    "src/kron_reduce/evaluation/energy_cap_scores.py",
    "src/kron_reduce/evaluation/energy_cap_transfer.py",
    "src/kron_reduce/evaluation/energy_constrained_initializer.py",
    "src/kron_reduce/evaluation/constant_c_observability.py",
    "src/kron_reduce/evaluation/history_metrics.py",
    "src/kron_reduce/evaluation/train_mechanism_audit.py",
    "src/kron_reduce/evaluation/train_refinement_audit.py",
    "src/kron_reduce/evaluation/scoped_audit_journal.py",
    "src/kron_reduce/models/causal_latent.py", "src/kron_reduce/models/learned_latent.py",
    "src/kron_reduce/models/nonlinear_latent.py", "src/kron_reduce/physics/coordinates.py",
    "src/kron_reduce/workflows/constant_c_synthetic.py",
    "src/kron_reduce/workflows/kiki_energy_cap_transfer.py",
    "project_harness/evidence.py", "project_harness/git_state.py", "project_harness/loop.py",
    "project_harness/manifest.py", "project_harness/policy.py", "project_harness/runner.py",
    "tests/test_encoder_energy_cap.py", "tests/test_energy_cap_saved_inputs.py",
    "tests/test_energy_cap_transfer.py", "tests/test_energy_cap_workflow.py",
    "tests/test_energy_cap_scores.py",
})


def _registered_experiment(root):
    entry = load_manifest(root).get("experiments", {}).get(EXPERIMENT)
    expected = {
        "module": MODULE, "workflow": WORKFLOW, "stage": STAGE,
        "config": Path(CONFIG).name, "execution_mode": MODE, "dataset": DATASET,
        "data_policy": DATA_POLICY, "reads_validation_performance": False,
        "reads_test_performance": False, "spatial_node_count": 10, "spatial_clustering": False,
        "input_contract_id": "measured_history_100ms_zoh_v1",
    }
    if not isinstance(entry, dict) or any(
            type(entry.get(k)) is not type(v) or entry[k] != v for k, v in expected.items()):
        raise RuntimeError("ENERGY_CAP_EXPERIMENT_REGISTRATION_MISMATCH")
    validate_run_policy(root, EXPERIMENT, entry)
    return entry


def authorize(root):
    root = Path(root).resolve(strict=True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("ENERGY_CAP_REQUIRES_PYTHON_312")
    assert_clean(root)
    _registered_experiment(root)
    verify_bound_file(root, "configs/research_policy.json", sha256_file(root / "configs/research_policy.json"))
    approval = load_policy(root).get("development_evaluations", {}).get(AUTH)
    if not isinstance(approval, dict) or approval.get("status") != "user_approved":
        raise RuntimeError("ENERGY_CAP_AUTHORIZATION_NOT_ACTIVE")
    if any(type(approval.get(k)) is not type(v) or approval[k] != v for k, v in SCOPE.items()):
        raise RuntimeError("ENERGY_CAP_SCOPE_MISMATCH")
    if approval.get("plan_path") != PLAN or approval.get("protocol_path") != PROTOCOL:
        raise RuntimeError("ENERGY_CAP_BOUND_PATH_MISMATCH")
    review_path = approval.get("review_path")
    if (not isinstance(review_path, str) or not review_path.startswith("docs/research/")
            or not review_path.endswith(".json")):
        raise RuntimeError("ENERGY_CAP_REVIEW_PATH_INVALID")
    bindings = approval.get("implementation_bindings")
    if not isinstance(bindings, dict) or not REQUIRED_BINDINGS.issubset(bindings):
        raise RuntimeError("ENERGY_CAP_BINDINGS_INCOMPLETE")
    for path, digest in [(PLAN, approval.get("plan_sha256")),
                         (PROTOCOL, approval.get("protocol_sha256")),
                         (review_path, approval.get("review_sha256")), *bindings.items()]:
        verify_bound_file(root, path, digest)
    plan, review = _read_json(root, PLAN), _read_json(root, review_path)
    expected = {"scope": SCOPE, "protocol_path": PROTOCOL,
                "protocol_sha256": approval["protocol_sha256"], "implementation_bindings": bindings}
    if (any(plan.get(k) != v for k, v in expected.items()) or review.get("approved") is not True
            or review.get("plan_path") != PLAN or review.get("plan_sha256") != approval["plan_sha256"]
            or any(review.get(k) != v for k, v in expected.items())):
        raise RuntimeError("ENERGY_CAP_REVIEW_OR_PLAN_BINDING_MISMATCH")
    return plan


def _reserve(root, run_id):
    common = Path(subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve(strict=True)
    directory = common / "energy_cap_transfer_budgets" / AUTH
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    source_commit = git_head(root)
    _write_new_json(directory / "reservation.json", {
        "authorization": AUTH, "run_id": run_id, "source_commit": source_commit,
        "plan_sha256": sha256_file(root / PLAN), "spent": True, "retry_allowed": False,
    })
    return directory, source_commit


def run(*, root, experiment, stage, result_dir, run_id):
    root = Path(root).resolve(strict=True)
    if stage != STAGE or experiment != _registered_experiment(root):
        raise RuntimeError("ENERGY_CAP_DIRECT_RUN_IDENTITY_MISMATCH")
    plan = authorize(root)
    result_dir = Path(result_dir).resolve(strict=True)
    if (result_dir.parent != (root / "results").resolve(strict=True)
            or result_dir.name != run_id or any(result_dir.iterdir())):
        raise RuntimeError("ENERGY_CAP_RESULT_DIRECTORY_NOT_NEW")
    budget_dir, source_commit = _reserve(root, run_id)
    try:
        kernel = importlib.import_module("kron_reduce.evaluation.energy_cap_transfer")
        runtime = _verify_loaded_runtime(root, plan)
        summary = kernel.execute(root=root, result_dir=result_dir, plan=plan, runtime=runtime)
        authorize(root)
        if git_head(root) != source_commit:
            raise RuntimeError("ENERGY_CAP_SOURCE_HEAD_CHANGED")
        verify_artifact_directory(result_dir, strict=True)
        _write_new_json(budget_dir / "completed.json", {
            "run_id": run_id, "authorization_spent": True, "source_commit": source_commit,
            "artifact_manifest_sha256": sha256_file(result_dir / "artifact_hashes.json"),
            "scientific_verdict": summary["scientific_verdict"],
        })
        return summary
    except BaseException as error:
        _write_new_json(budget_dir / "failed.json", {
            "run_id": run_id, "authorization_spent": True, "error_type": type(error).__name__,
            "message": str(error), "retry_allowed": False,
        })
        raise


def execute_registered(root):
    root = Path(root).resolve(strict=True)
    authorize(root)
    with workflow_lock(root, WORKFLOW):
        return _run_experiment(root, EXPERIMENT, STAGE, experiment=_registered_experiment(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the one-use encoder-energy-cap train comparison")
    parser.add_argument("--root", type=Path, default=project_root())
    print(execute_registered(parser.parse_args(argv).root))


if __name__ == "__main__":
    main()
