"""Recover completed A05 evidence after a runner verdict-schema failure, with zero fits."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines


def validate_completed_source(source, manifest_sha256, failure_sha256):
    """The sole allowed failed-source repair is a completed verdict-schema closeout."""
    source = Path(source)
    manifest_path, failure_path = source / "artifact_hashes.json", source / "RUN_FAILED.json"
    if (source.is_symlink() or sha256_file(manifest_path) != manifest_sha256
        or sha256_file(failure_path) != failure_sha256):
        raise ValueError("A05 closeout source binding changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if failure["error_type"] != "EvidenceValidationError" or failure["message"] != (
        "EVIDENCE_VERDICT: unregistered scientific_verdict: 'REJECT_DIAGNOSTIC'"):
        raise ValueError("closeout does not authorize recovery of other failure types")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual != set(manifest) | {"artifact_hashes.json", "RUN_FAILED.json"}:
        raise ValueError("A05 failed source inventory changed")
    for relative, digest in manifest.items():
        path = (source / relative).resolve()
        if (not path.is_relative_to(source.resolve()) or any(p.is_symlink() for p in (source / relative).parents)
            or (source / relative).is_symlink() or sha256_file(path) != digest):
            raise ValueError("A05 failed source artifact changed")
    summary = json.loads((source / "learning_summary.json").read_text(encoding="utf-8"))
    if (summary["process_status"] != "PASS" or summary["scientific_verdict"] != "REJECT_DIAGNOSTIC"
        or not summary["checks"]["all_six_fits_completed"]
        or summary["budget"]["fit_attempts_reserved"] != 6
        or summary["budget"]["confirmed_optimizer_steps"] != 480
        or summary["model_promoted"] or summary["next_stage_research_supported"]):
        raise ValueError("source did not complete the exact rejected A05 diagnostic")
    return manifest, summary


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != "closeout" or status_lines(root):
        raise RuntimeError("A05 closeout needs a clean committed checkout")
    config_path = root / "configs/experiments" / experiment["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = root / "results" / config["source_run"]
    manifest, summary = validate_completed_source(source, config["source_manifest_sha256"],
                                                   config["source_failure_sha256"])
    renamed = {"artifact_hashes.json": "source_artifact_hashes.json",
               "RUN_FAILED.json": "source_runner_failure.json",
               "stage_evidence.json": "source_stage_evidence.json",
               "learning_summary.json": "source_learning_summary.json"}
    source_files = [*manifest, "artifact_hashes.json", "RUN_FAILED.json"]
    for relative in source_files:
        target = result_dir / renamed.get(relative, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("immutable closeout target exists")
        shutil.copyfile(source / relative, target)
        if sha256_file(target) != sha256_file(source / relative):
            raise ValueError("A05 recovery copy differs")
    summary.update(scientific_verdict="REJECT", closeout_source_run=source.name,
                   execution_status="COMPLETED_EXPERIMENT_WITH_RUNNER_CLOSEOUT_REPAIR",
                   source_run_failed=True, source_failure_preserved=True,
                   new_fit_attempts_in_closeout=0, new_optimizer_updates_in_closeout=0,
                   new_forecasts_in_closeout=0)
    _write(result_dir / "learning_summary.json", summary)
    _write(result_dir / "closeout_receipt.json", {"source_run": source.name,
           "source_manifest_sha256": config["source_manifest_sha256"],
           "source_failure_sha256": config["source_failure_sha256"],
           "all_original_artifacts_copied_byte_identically": True,
           "original_failed_run_unchanged": True, "source_run_failed": True,
           "new_fit_attempts": 0, "new_optimizer_updates": 0, "new_forecasts": 0,
           "repair_commit": git_head(root), "run_id": run_id})
    evidence = json.loads((source / "stage_evidence.json").read_text(encoding="utf-8"))
    evidence.update(scientific_verdict="REJECT", source_commit=git_head(root),
                    source_fit_commit=summary["records"][0]["fit_record"].get("source_commit",
                        json.loads((source/"scientific_plan.json").read_text(encoding="utf-8"))["source_commit"]),
                    source_run_failed=True, closeout_only=True,
                    new_fit_attempts=0, new_optimizer_updates=0, new_forecasts=0)
    _write(result_dir / "stage_evidence.json", evidence)
    validate_completed_source(source, config["source_manifest_sha256"], config["source_failure_sha256"])
    _write(result_dir / "artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
           for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
