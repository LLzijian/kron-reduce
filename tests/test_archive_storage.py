from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from kron_reduce.workflows.unified_first_swing import _archive_asset
from project_harness.archive_storage import archive_path, read_run_artifact, verify_archived_run
from project_harness.audit import _stage_evidence_error
from project_harness.evidence import (
    EvidenceContext,
    EvidenceValidationError,
    _verify_report_bindings,
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


def test_pruned_archived_run_verifies_retained_members_and_fails_closed(tmp_path: Path) -> None:
    run_id = "pruned-run"
    kept_rel = "metrics/summary.json"
    dropped_rel = "predictions.npz"
    kept_bytes = b'{"status": "PASS"}'
    dropped_bytes = b"bulk predictions"
    manifest = json.dumps({
        kept_rel: hashlib.sha256(kept_bytes).hexdigest(),
        dropped_rel: hashlib.sha256(dropped_bytes).hexdigest(),
    }).encode()
    archive_pathname = "results/backup/04_topic/pruned.zip"
    archive = tmp_path / archive_pathname
    archive.parent.mkdir(parents=True)
    _relocations(tmp_path, {f"results/{run_id}": {"archive": archive_pathname, "pruned": True}})
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
        stream.writestr(f"{run_id}/{kept_rel}", kept_bytes)

    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold") is None
    with pytest.raises(EvidenceValidationError, match="ARTIFACT_DIRECTORY"):
        validate_stage_evidence(EvidenceContext(tmp_path, workflow, "fit"), evidence)

    # Unpruned string mapping rejects missing members.
    _relocations(tmp_path, {f"results/{run_id}": archive_pathname})
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_INVENTORY:")

    # Invalid mapping dict shape fails closed.
    _relocations(tmp_path, {f"results/{run_id}": {"archive": archive_pathname}})
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_ARCHIVE_PATH:")

    # Restored pruned mapping rejects extra unregistered files, tampered hashes, bad manifest SHA, or empty retained set.
    _relocations(tmp_path, {f"results/{run_id}": {"archive": archive_pathname, "pruned": True}})
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr(f"{run_id}/extra.txt", b"unregistered")
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_INVENTORY:")

    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{run_id}/artifact_hashes.json", manifest)
        stream.writestr(f"{run_id}/{kept_rel}", b'{"status": "TAMPERED"}')
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_HASH:")

    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{run_id}/artifact_hashes.json", b"{}")
        stream.writestr(f"{run_id}/{kept_rel}", kept_bytes)
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_MANIFEST_SHA:")

    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{run_id}/artifact_hashes.json", manifest)
    assert _stage_evidence_error(tmp_path, workflow, "fit", evidence, "scaffold").startswith("ARTIFACT_INVENTORY:")


def test_archived_report_acceptance_requires_acceptance_json(tmp_path: Path) -> None:
    acceptance_run = "acc-run"
    acceptance_bytes = b'{"accepted": true}'
    extra_bytes = b'{"stage": "acceptance"}'
    manifest = json.dumps({
        "acceptance.json": hashlib.sha256(acceptance_bytes).hexdigest(),
        "stage_evidence.json": hashlib.sha256(extra_bytes).hexdigest(),
    }).encode()
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    archive_rel = "results/backup/03_topic/acc.zip"
    archive = tmp_path / archive_rel
    archive.parent.mkdir(parents=True)
    _relocations(tmp_path, {f"results/{acceptance_run}": {"archive": archive_rel, "pruned": True}})

    report_rel = "docs/reports/acc_report.md"
    report_path = tmp_path / report_rel
    report_path.parent.mkdir(parents=True)
    report_bytes = b"# Report\n"
    report_path.write_bytes(report_bytes)

    workflow = {"evidence_contract_version": 2}
    acc_manifest = {"run_id": acceptance_run, "path": "artifact_hashes.json", "sha256": manifest_sha}
    state = {
        "history": [{
            "action": "advance",
            "from": "review",
            "to": "acceptance",
            "evidence": {"result_dir": acceptance_run, "artifact_manifest": acc_manifest},
        }]
    }
    report_evidence = {
        "status": "PASS",
        "acceptance_run": acceptance_run,
        "artifact_manifest": acc_manifest,
        "report_path": report_rel,
        "report_artifact": {"path": report_rel, "sha256": hashlib.sha256(report_bytes).hexdigest()},
    }

    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{acceptance_run}/artifact_hashes.json", manifest)
        stream.writestr(f"{acceptance_run}/acceptance.json", acceptance_bytes)

    assert verify_archived_run(tmp_path, acceptance_run, manifest_sha, required_members=("acceptance.json",))
    ctx_archived = EvidenceContext(tmp_path, workflow, "report", state=state, allow_archived_runs=True)
    validate_stage_evidence(ctx_archived, report_evidence)

    ctx_live = EvidenceContext(tmp_path, workflow, "report", state=state, allow_archived_runs=False)
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_ACCEPTANCE_RUN"):
        _verify_report_bindings(ctx_live, report_evidence)

    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{acceptance_run}/artifact_hashes.json", manifest)
        stream.writestr(f"{acceptance_run}/stage_evidence.json", extra_bytes)

    assert not verify_archived_run(tmp_path, acceptance_run, manifest_sha, required_members=("acceptance.json",))
    with pytest.raises(EvidenceValidationError, match="EVIDENCE_ACCEPTANCE_RUN"):
        validate_stage_evidence(ctx_archived, report_evidence)
