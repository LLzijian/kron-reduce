from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from kron_reduce.workflows.unified_first_swing import _archive_asset
from project_harness.archive_storage import archive_path, read_run_artifact
from project_harness.audit import _stage_evidence_error
from project_harness.evidence import (
    EvidenceContext,
    EvidenceValidationError,
    validate_stage_evidence,
)


def _relocations(root: Path, locations: dict[str, str]) -> None:
    path = root / "configs/archive_relocations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(locations), encoding="utf-8")


def test_archive_path_maps_full_root_and_preserves_other_paths(tmp_path: Path) -> None:
    _relocations(tmp_path, {"results/backup/old_group": "results/backup/04_topic"})

    relative = "results/backup/old_group/run.zip"
    old = tmp_path / relative
    old.parent.mkdir(parents=True)
    old.write_bytes(b"legacy layout")
    assert archive_path(tmp_path, relative) == old
    current = tmp_path / "results/backup/04_topic/run.zip"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"numbered layout")
    assert archive_path(tmp_path, relative) == current
    similar_group = "results/backup/old_group_other/run.zip"
    assert archive_path(tmp_path, similar_group) == tmp_path / similar_group
    assert archive_path(tmp_path, "results/diagnostics/run/model.npz") == (
        tmp_path / "results/diagnostics/run/model.npz"
    )


def test_archive_path_rejects_traversal_mapping(tmp_path: Path) -> None:
    _relocations(tmp_path, {"results/backup/old_group": "results/backup/../escape"})

    with pytest.raises(RuntimeError):
        archive_path(tmp_path, "results/backup/old_group/run.zip")


def test_archive_asset_reads_relocated_zip_member_and_checks_sha(tmp_path: Path) -> None:
    old_archive = "results/backup/old_group/model.zip"
    new_archive = "results/backup/04_topic/model.zip"
    _relocations(tmp_path, {"results/backup/old_group": "results/backup/04_topic"})
    archive = tmp_path / new_archive
    archive.parent.mkdir(parents=True)
    member = "diagnostics/frozen-run/checkpoint.npz"
    payload = b"frozen checkpoint"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, payload)

    binding = {
        "path": "results/" + member,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert _archive_asset(tmp_path, binding, old_archive) == payload

    binding["sha256"] = hashlib.sha256(b"changed checkpoint").hexdigest()
    with pytest.raises(RuntimeError, match="frozen starting asset changed"):
        _archive_asset(tmp_path, binding, old_archive)


def test_archived_run_preserves_history_and_hash_bound_reads(tmp_path: Path) -> None:
    run_id = "frozen-run"
    relative = "metrics/summary.json"
    payload = b'{"status": "PASS"}'
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.dumps({relative: digest}).encode()
    archive_pathname = "results/backup/10_topic/runs.zip"
    archive = tmp_path / archive_pathname
    archive.parent.mkdir(parents=True)
    _relocations(tmp_path, {f"results/{run_id}": archive_pathname})
    workflow = {"evidence_contract_version": 2}
    evidence = {
        "status": "PASS",
        "result_dir": run_id,
        "artifact_manifest": {
            "run_id": run_id,
            "path": "artifact_hashes.json",
            "sha256": hashlib.sha256(manifest).hexdigest(),
        },
    }

    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{run_id}/artifact_hashes.json", manifest)
        stream.writestr(f"{run_id}/{relative}", payload)
        stream.writestr("another-run/summary.json", b"other result")

    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold") is None
    with pytest.raises(EvidenceValidationError, match="ARTIFACT_DIRECTORY"):
        validate_stage_evidence(EvidenceContext(tmp_path, workflow, "fit"), evidence)
    assert read_run_artifact(tmp_path, run_id, relative, digest) == payload

    live_run = tmp_path / "results" / run_id
    live_run.mkdir()
    with pytest.raises(EvidenceValidationError, match="ARTIFACT_MISSING"):
        read_run_artifact(tmp_path, run_id, relative, digest)
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_MANIFEST:")
    live_run.rmdir()

    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{run_id}/artifact_hashes.json", manifest)
        stream.writestr(f"{run_id}/{relative}", b"changed result")

    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_HASH:")
    with pytest.raises(EvidenceValidationError, match="ARTIFACT_HASH"):
        read_run_artifact(tmp_path, run_id, relative, digest)

    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr(f"{run_id}/../unsafe.json", b"extra")
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_ARCHIVE_PATH:")
